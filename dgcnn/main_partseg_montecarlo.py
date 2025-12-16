#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao, Pengliang Ji
@Contact: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
@File: main_partseg.py
@Time: 2021/7/20 7:49 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
#from data import ShapeNetPart
from ShapeNetDataLoader1 import My_H5Dataset
from model import DGCNN_partseg
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream, calculate_uncertainity_logits,CalculateUncertaintyLogits2
import sklearn.metrics as metrics
#from plyfile import PlyData, PlyElement
import hashlib
import h5py
import random
global class_cnts
class_indexs = np.zeros((1,), dtype=int)
global visual_warning
visual_warning = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
#class_choices = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
class_choices=['Wheat']
#seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
seg_num=[2]
#index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
index_start=[0]
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
    if not os.path.exists('outputs/'+args.exp_name+'/'+'visualization'):
        os.makedirs('outputs/'+args.exp_name+'/'+'visualization')
    os.system('cp main_partseg.py outputs'+'/'+args.exp_name+'/'+'main_partseg.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')

def hash_tensor(tensor):
    """
    Generates a hash for a tensor for easy comparison and detection of duplicates.
    This function converts the tensor to its byte representation and hashes it.
    """
    # Ensure the tensor is on CPU and convert to numpy to extract bytes
    tensor_bytes = tensor.cpu().numpy().tobytes()
    hash_obj = hashlib.sha256(tensor_bytes)
    return hash_obj.hexdigest()

def apply_leaf_crossover(points1, labels1, points2, labels2):
    """Swap ear points between two point clouds."""
    # Extract ear points based on labels
    ear_points1 = points1[labels1 == 1]
    non_ear_points1 = points1[labels1 == 0]

    ear_points2 = points2[labels2 == 1]
    non_ear_points2 = points2[labels2 == 0]

    # Swap ear points
    new_points1 = torch.cat((non_ear_points1, ear_points2))
    new_labels1 = torch.cat((torch.zeros_like(non_ear_points1[:, 0]), torch.ones_like(ear_points2[:, 0])))

    new_points2 = torch.cat((non_ear_points2, ear_points1))
    new_labels2 = torch.cat((torch.zeros_like(non_ear_points2[:, 0]), torch.ones_like(ear_points1[:, 0])))

    return new_points1, new_labels1, new_points2, new_labels2

def apply_scaling(point_cloud, scale_range=(0.98, 1.05)):
    """
    Apply small random scaling to the point cloud.
    
    Args:
    - point_cloud (torch.Tensor): The input point cloud (N, 3).
    - scale_range (tuple): The range of scaling factors.
    
    Returns:
    - scaled_point_cloud (torch.Tensor): The scaled point cloud.
    """
    # Generate random scale factor
    scale_factor = np.random.uniform(*scale_range)
    scaled_point_cloud = point_cloud * scale_factor
    #print(scale_factor)
    return scaled_point_cloud

def leaf_rotation(points, angle_range=(0, 180)):
    """Apply random rotation to the point cloud.
    
    Args:
        points: Tensor of shape [batch_size, num_points, 3]
        angle_range: Tuple of (min_angle, max_angle) for rotation in degrees.
    
    Returns:
        Rotated point cloud of the same shape.
    """
    batch_size, num_points, _ = points.shape
    
    # Generate random rotation angles for each batch
    angles = torch.rand(batch_size) * (angle_range[1] - angle_range[0]) + angle_range[0]
    angles = torch.deg2rad(angles)  # Convert to radians
    
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    
    # Create rotation matrices for rotation around the Y-axis
    rotation_matrices = torch.zeros((batch_size, 3, 3)).to(points.device)
    rotation_matrices[:, 0, 0] = cos_vals
    rotation_matrices[:, 0, 2] = -sin_vals
    rotation_matrices[:, 1, 1] = 1
    rotation_matrices[:, 2, 0] = sin_vals
    rotation_matrices[:, 2, 2] = cos_vals
    
    # Transpose points to shape [batch_size, 3, num_points] for matrix multiplication
    #print('points',points.shape)
    #points = points.transpose(1, 2)  # Now [batch_size, 3, num_points]
    
    # Apply the rotation using batch matrix multiplication
    rotated_points = torch.bmm(rotation_matrices, points)  # Result [batch_size, 3, num_points]
    print(rotated_points.shape)
    # Transpose back to original shape [batch_size, num_points, 3]
    #rotated_points = rotated_points.transpose(1, 2)  # Now [batch_size, num_points, 3]
    #print(rotated_points.shape)
    return rotated_points


def leaf_translation(points, translation_range=(-0.1, 0.1)):
    """Apply random translation to the point cloud."""
    batch_size, num_coords, num_points = points.shape
    
    # Generate random translations for each dimension (x, y, z)
    translations = torch.FloatTensor(batch_size, num_coords, 1).uniform_(*translation_range).to(points.device)  # Shape [batch_size, 3, 1]
    
    # Broadcast the translations to match the shape of points
    translated_points = points + translations  # Shape [batch_size, 3, num_points]
    
    return translated_points
def adjust_uncertainty_threshold(all_scores, percentile):
    """Adjust the threshold to the given percentile of recent scores."""
    # Convert PyTorch tensors to numpy arrays, handle CUDA device and gradient requirement
    if all_scores and isinstance(all_scores[0], torch.Tensor):
        all_scores = [score.detach().cpu().numpy() if score.requires_grad 
                      else score.cpu().numpy() if score.is_cuda 
                      else score.numpy() 
                      for score in all_scores]
    return np.percentile(all_scores, percentile) if all_scores else 0

def select_most_uncertain_samples2(uncertain_samples, labels, targets, scores, num_to_select, min_uncertainty_threshold):
    """ Selects the most uncertain samples while avoiding duplicates.
    
    Args:
        uncertain_samples (list): List of tensor samples.
        labels (list): Corresponding labels of the samples.
        targets (list): Corresponding targets of the samples.
        scores (list): Uncertainty scores of the samples.
        num_to_select (int): Number of samples to select.
        min_uncertainty_threshold (float): Minimum threshold for uncertainty to consider a sample.

    Returns:
        tuple: Selected samples, labels, targets, and scores.
    """
    paired_samples = list(zip(uncertain_samples, labels, targets, scores))
    paired_samples.sort(key=lambda x: x[3], reverse=True)

    selected_samples, selected_labels, selected_targets, selected_scores = [], [], [], []
    skipped_samples = 0

    for sample, label, target, score in paired_samples:
        # Ensure that the sample retains its original dimensions
        if sample.dim() == 2 and sample.shape[0] == 3:  # Likely squeezed inadvertently
            sample = sample.unsqueeze(0)  # Add the batch dimension back
        
        if score > min_uncertainty_threshold:
            sample_hash = tensor_hash(sample)
            if sample_hash in seen_samples:
                skipped_samples += 1
                continue

            seen_samples.add(sample_hash)
            selected_samples.append(sample)
            selected_labels.append(label)
            selected_targets.append(target)
            selected_scores.append(score)
            
            if len(selected_samples) == num_to_select:
                break

    print(f"Skipped {skipped_samples} duplicate samples out of {len(paired_samples)} total processed.")
    #return selected_samples, selected_labels, selected_targets, selected_scores
    return selected_samples[:num_to_select], selected_labels[:num_to_select], selected_targets[:num_to_select], selected_scores[:num_to_select]

def select_most_uncertain_samples(uncertain_samples, labels, targets, scores, num_to_select,min_uncertainty_threshold):
    paired_samples = list(zip(uncertain_samples, labels, targets, scores))
    # Sort by uncertainty score, assuming higher scores indicate higher uncertainty
    paired_samples.sort(key=lambda x: x[3], reverse=True)

    uncertain_samples_list = uncertain_samples.tolist() if isinstance(uncertain_samples, torch.Tensor) else uncertain_samples
    #print("Sorted paired samples:", paired_samples.shape)
    print("Uncertain samples list:", len(uncertain_samples_list))
    # Select top 'num_to_select' samples
    #selected_samples, selected_labels, selected_targets = zip(*[(x[0], x[1], x[2]) for x in paired_samples[:num_to_select]])
    selected_samples, selected_labels, selected_targets = [], [], []
    for sample, label, target, score in paired_samples:
        if score > min_uncertainty_threshold:  # Check if current score is higher than the threshold
            sample_found = False
            # Check if the sample is already in the selected list
            for s in selected_samples:
                if torch.equal(sample, s):
                    sample_found = True
                    break
            
            if not sample_found:
                selected_samples.append(sample)
                selected_labels.append(label)
                selected_targets.append(target)
                # Update the minimum uncertainty threshold with the new higher score
               # print('score',score)
                min_uncertainty_threshold = max(min_uncertainty_threshold, score)
                print('mn unceratin',min_uncertainty_threshold)
                if len(selected_samples) == num_to_select:
                    break

    return selected_samples[:num_to_select], selected_labels[:num_to_select], selected_targets[:num_to_select], min_uncertainty_threshold
        

   # return selected_samples, selected_labels, selected_targets
    return selected_samples[:num_to_select], selected_labels[:num_to_select], selected_targets[:num_to_select]

def compute_dynamic_k(epoch, start_epoch, end_epoch, max_k, batch_size):
    # Linearly increase k from 0 to max_k during start_epoch to end_epoch
    if epoch < start_epoch:
       return 0
    elif epoch < end_epoch:
      # print( int((epoch - start_epoch) / (end_epoch - start_epoch) * max_k ))
      # return int((epoch - start_epoch) / (end_epoch - start_epoch) * max_k )
      return math.ceil((epoch / (end_epoch - start_epoch) * max_k))
    else:
       return max_k
   

def compute_dynamic_k(current_accuracy, previous_accuracy, max_k=4, min_k=2):
     if current_accuracy < previous_accuracy:  # If current accuracy is worse than previous
        return min(max_k, max_k // 2 + int((previous_accuracy - current_accuracy) * 500))
     else:  # If current accuracy improves or stays the same
        return max(min_k, max_k // 2 - int((current_accuracy - previous_accuracy) * 500))




def calculate_shape_IoU(pred_np, seg_np, label, class_choice, visual=False):
    if not visual:
        label = label.squeeze()
    shape_ious = []
    parts=range(0,2)
    shape_ious0=[]
    shape_ious1=[]
    #for shape_idx in range(seg_np.shape[0]):
    #    if not class_choice:
    #        start_index = index_start[label[shape_idx]]
    #        num = seg_num[label[shape_idx]]
    #        parts = range(start_index, start_index + num)
    #        print(parts)
    #    else:
    #        parts = range(seg_num[label[0]])
    for shape_idx in range(seg_np.shape[0]):
        #print(seg_np.shape[0])

        part_ious = []
        part_ious0=[]
        part_ious1=[]
        for part in parts:
            #print('part',part)
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            #print(I,U)
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            if part== 0 :
                part_ious0.append(iou)
               # print(part_ious0)
            else :
                part_ious1.append(iou)
                #print(part_ious1)
            part_ious.append(iou)
            
    shape_ious.append(np.mean(part_ious))
    shape_ious0.append(np.mean(part_ious0))
    shape_ious1.append(np.mean(part_ious1))
    return shape_ious,shape_ious0,shape_ious1


def visualization(visu, visu_format, data, pred, seg, label, partseg_colors, class_choice):
    global class_indexs
    global visual_warning
    visu = visu.split('_')
    for i in range(0, data.shape[0]):
        RGB = []
        RGB_gt = []
        skip = False
        classname = class_choices[int(label[i])]
        class_index = class_indexs[int(label[i])]
        if visu[0] != 'all':
            if len(visu) != 1:
                if visu[0] != classname or visu[1] != str(class_index):
                    skip = True 
                else:
                    visual_warning = False
            elif visu[0] != classname:
                skip = True 
            else:
                visual_warning = False
        elif class_choice != None:
            skip = True
        else:
            visual_warning = False
        if skip:
            class_indexs[int(label[i])] = class_indexs[int(label[i])] + 1
        else:  
            if not os.path.exists('outputs/'+args.exp_name+'/'+'visualization'+'/'+classname):
                os.makedirs('outputs/'+args.exp_name+'/'+'visualization'+'/'+classname)
            for j in range(0, data.shape[2]):
                RGB.append(partseg_colors[int(pred[i][j])])
                RGB_gt.append(partseg_colors[int(seg[i][j])])
            pred_np = []
            seg_np = []
            pred_np.append(pred[i].cpu().numpy())
            seg_np.append(seg[i].cpu().numpy())
            xyz_np = data[i].cpu().numpy()
            xyzRGB = np.concatenate((xyz_np.transpose(1, 0), np.array(RGB)), axis=1)
            xyzRGB_gt = np.concatenate((xyz_np.transpose(1, 0), np.array(RGB_gt)), axis=1)
            IoU = calculate_shape_IoU(np.array(pred_np), np.array(seg_np), label[i].cpu().numpy(), class_choice, visual=True)
            IoU = str(round(IoU[0], 4))
            filepath = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+classname+'/'+classname+'_'+str(class_index)+'_pred_'+IoU+'.'+visu_format
            filepath_gt = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+classname+'/'+classname+'_'+str(class_index)+'_gt.'+visu_format
            if visu_format=='txt':
                np.savetxt(filepath, xyzRGB, fmt='%s', delimiter=' ') 
                np.savetxt(filepath_gt, xyzRGB_gt, fmt='%s', delimiter=' ') 
                print('TXT visualization file saved in', filepath)
                print('TXT visualization file saved in', filepath_gt)
            elif visu_format=='ply':
                xyzRGB = [(xyzRGB[i, 0], xyzRGB[i, 1], xyzRGB[i, 2], xyzRGB[i, 3], xyzRGB[i, 4], xyzRGB[i, 5]) for i in range(xyzRGB.shape[0])]
                xyzRGB_gt = [(xyzRGB_gt[i, 0], xyzRGB_gt[i, 1], xyzRGB_gt[i, 2], xyzRGB_gt[i, 3], xyzRGB_gt[i, 4], xyzRGB_gt[i, 5]) for i in range(xyzRGB_gt.shape[0])]
                vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(filepath)
                vertex = PlyElement.describe(np.array(xyzRGB_gt, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(filepath_gt)
                print('PLY visualization file saved in', filepath)
                print('PLY visualization file saved in', filepath_gt)
            else:
                print('ERROR!! Unknown visualization format: %s, please use txt or ply.' % \
                (visu_format))
                exit()
            class_indexs[int(label[i])] = class_indexs[int(label[i])] + 1


def train(args, io):
    #train_dataset = ShapeNetPart(partition='trainval', num_points=args.num_points, class_choice=args.class_choice)
    #if (len(train_dataset) < 100):
    #    drop_last = False
    #else:
    #    drop_last = True
    #train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=drop_last)
    #test_loader = DataLoader(ShapeNetPart(partition='test', num_points=args.num_points, class_choice=args.class_choice), 
    #                        num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    TRAIN_DATASET=My_H5Dataset(os.path.join(ROOT_DIR, 'wheat_a2r4k/TRAIN_A2R4K.h5'),normal_channel=False)
    VAL_DATASET=My_H5Dataset(os.path.join(ROOT_DIR, 'wheat_a2r4k/VAL_A2R4K.h5'),normal_channel=False)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
                 #TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
    print('lendataloader',len(trainDataLoader))
    valDataLoader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    


    device = torch.device("cuda" if args.cuda else "cpu")
    #device = torch.device("cuda:1")# if torch.cuda.is_available() else "cpu")
    #Try to load models
    #seg_num_all = train_loader.dataset.seg_num_all
    seg_num_all= 2
    seg_start_index = 0
    if args.model == 'dgcnn':
        model = DGCNN_partseg(args, seg_num_all).to(device)
    else:
        raise Exception("Not implemented")
    #print(str(model))
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)
    model = nn.DataParallel(model)
    #print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.5)

    criterion = cal_loss
    criterion2=CalculateUncertaintyLogits2().to(device)
    
    previous_accuracy = 0.0  # Start with zero or a baseline accuracy
    max_k = 4
    dynamic_k = 2
    
    baseline_accuracy = 0.0
    initial_min_uncertainty_threshold = 0
    best_test_iou = 0
    uncertain_samples = []
    uncertain_labels = []
    uncertain_targets = []
    unique_hashes = set()
    
    for epoch in range(args.epochs):
        #print(epoch)
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()

        all_hard_points = []
        all_hard_labels = []
        all_hard_targets = []
        all_uncertainty_scores=[]
        
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        for data, label, seg in trainDataLoader:
            seg = seg - seg_start_index 
            #print(data.shape,label.shape,seg.shape)
            #label=label.to(device)
            label_one_hot = np.zeros((label.shape[0], 1))
            #print(label_one_hot)
            for idx in range(label.shape[0]):
                
                #label_one_hot[idx, label[idx]] = 1
                label_one_hot[idx,0]=1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            #print('labels',label[:20],label_one_hot[:5])
            data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data, label_one_hot)
            #print(data.shape,label_one_hot.shape)
            #print(seg_pred.shape)
            #uncertainty=criterion2(seg_pred)
            uncertainty = criterion2.mc_dropout_variance(model,data,label_one_hot,num_samples=5)
            #print("Uncertainty Tensor:", uncertainty)
            #print("Tensor Type:", type(uncertainty))
            _, hard_samples_indices = uncertainty.topk(k=dynamic_k, largest=True)

            if hard_samples_indices.numel() > 0 :
               
               for idx in (hard_samples_indices):
        # Extract and append each sample and its corresponding data individually
                      #print('idx',idx)
                      #print(data[idx, :, :].shape)
                      #print(data.shape,label.shape,seg.shape)
                      single_sample = data[idx, :, :].unsqueeze(0)
                      single_label = label_one_hot[idx].unsqueeze(0) #if label.dim() > 0 else label  # Check dimension if label is a tensor
                      single_target = seg[idx].unsqueeze(0) #if seg.dim() > 0 else seg
                      single_score = uncertainty[idx].unsqueeze(0)
                      #print('single_sample',single_sample.shape,single_label.shape,single_score.shape)

                      all_hard_points.append(single_sample)
                      all_hard_labels.append(single_label)
                      all_hard_targets.append(single_target)
                      all_uncertainty_scores.append(single_score.item()) 
           
           

       
           # print('seg',seg_pred.shape,seg_pred.view(-1, seg_num_all).shape, seg.view(-1,1).squeeze().shape)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1,1).squeeze())
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
           # print(seg_np.shape)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
           # print('gh',len(train_true_seg),len(train_pred_seg))
            train_label_seg.append(label.reshape(-1))
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        #train_label_seg = [x.cpu().numpy() for x in train_label_seg] 
        train_label_seg = np.concatenate(train_label_seg)
        
        train_ious,train_ious0,train_ious1 = calculate_shape_IoU(train_pred_seg, train_true_seg, train_label_seg, args.class_choice)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f , train_iou0: %.6f,train _iou1:%.6f' % (epoch, 
                                                                                                  train_loss*1.0/count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious),np.mean(train_ious0),np.mean(train_ious1))

   
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        test_label_seg = []
        for data, label, seg in valDataLoader:
            seg = seg - seg_start_index
            label_one_hot = np.zeros((label.shape[0], 1))
            for idx in range(label.shape[0]):
                label_one_hot[idx, 0] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            seg_pred = model(data, label_one_hot)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1,1).squeeze())
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
            test_label_seg.append(label.reshape(-1))
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_label_seg = np.concatenate(test_label_seg)
        test_ious,test_ious0,test_ious1 = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f,test_iou0: %.6f, test_iou1: %.6f' % (epoch,
                                                                                              test_loss*1.0/count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              np.mean(test_ious),np.mean(test_ious0),np.mean(test_ious1))
        io.cprint(outstr)
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            torch.save(model.state_dict(), 'outputs/%s/models/model.t7' % args.exp_name)
        if test_acc >= baseline_accuracy :
              dynamic_k = compute_dynamic_k(test_acc, baseline_accuracy, max_k) 
              print('dynamic_k',dynamic_k)
              baseline_accuracy = test_acc
              should_oversample = False
              #print(baseline_accuracy)
                
        elif test_acc < baseline_accuracy and epoch % 5 == 0:
             print('Performing oversampling at epoch:', epoch)
             should_oversample = True  # Enable oversampling every 20 epochs
             print('oversample')
        else:
             should_oversample = False 
                
        if should_oversample:
           if all_uncertainty_scores:
             
              all_uncertainty_tensor = torch.tensor(all_uncertainty_scores)  # Convert the list to a tensor
              threshold = np.percentile(all_uncertainty_tensor, 10)
              #topk_indices = all_uncertainty_tensor.topk(k=dynamic_k, largest=True).indices  # Get indices of top k uncertain samples
              topk_indices = all_uncertainty_tensor.topk(k=dynamic_k, largest=True).indices
           selected_hard_points = torch.cat([all_hard_points[i] for i in topk_indices], dim=0)
           selected_hard_labels = torch.cat([all_hard_labels[i] for i in topk_indices], dim=0)
           selected_hard_targets = torch.cat([all_hard_targets[i] for i in topk_indices], dim=0)
           print('sel',selected_hard_points.shape)
           
        # Apply augmentation (rotation or translation) to selected samples
           # Apply leaf crossover on hard samples
           
           if torch.rand(1).item() > 0.5:
             #selected_hard_points = leaf_rotation(selected_hard_points)
              num_selected = len(selected_hard_points)
              if len(selected_hard_points) >= 2:  # Need at least two to perform crossover
                 indices = list(range(num_selected))
                 random.shuffle(indices)
                 print('leaf crossover')
        # Perform pairwise crossover on the shuffled points
                 for i in range(0, num_selected - 1, 2):
                     idx1, idx2 = indices[i], indices[i + 1]
                     points1, labels1 = selected_hard_points[idx1], selected_hard_labels[idx1]
                     points2, labels2 = selected_hard_points[idx2], selected_hard_labels[idx2]
              else: 
                  print('translation')
                  selected_hard_points = leaf_translation(selected_hard_points)
           else:
            
              print('scaling')
              #selected_hard_points = leaf_translation(selected_hard_points)
              selected_hard_points= apply_scaling(selected_hard_points)
           
        # Concatenate the augmented samples back into the dataset for the next epoch
           # Append the new hard samples to the existing features and labels
           selected_hard_points = selected_hard_points.permute(0, 2, 1).contiguous() 
           trainDataLoader.dataset.features = np.concatenate([trainDataLoader.dataset.features, selected_hard_points.cpu().numpy()], axis=0)
           trainDataLoader.dataset.labels = np.concatenate([trainDataLoader.dataset.labels, selected_hard_labels.cpu().numpy()], axis=0)
           trainDataLoader.dataset.index = np.concatenate([trainDataLoader.dataset.index, selected_hard_targets.cpu().numpy()], axis=0)
            

    
                
    

def test(args, io):
    #test_loader = DataLoader(ShapeNetPart(partition='test', num_points=args.num_points, class_choice=args.class_choice),
    #                         batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    TEST_DATASET=My_H5Dataset(os.path.join(ROOT_DIR, 'wheat_a2r4k/TEST_A2R4K.h5'),normal_channel=False)

    #trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
                          #TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    device = torch.device("cuda" if args.cuda else "cpu")
    
    #Try to load models
    seg_num_all =2
    seg_start_index = 0
    #partseg_colors = test_loader.dataset.partseg_colors
    if args.model == 'dgcnn':
        model = DGCNN_partseg(args, seg_num_all).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    #model = model.to(device)

    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true_cls = []
    test_pred_cls = []
    test_true_seg = []
    test_pred_seg = []
    test_label_seg = []
    for data, label, seg in testDataLoader:
        seg = seg - seg_start_index
        label_one_hot = np.zeros((label.shape[0], 1))
        for idx in range(label.shape[0]):
            label_one_hot[idx, 0] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        seg_pred = model(data, label_one_hot)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        pred = seg_pred.max(dim=2)[1]
        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        test_true_cls.append(seg_np.reshape(-1))
        test_pred_cls.append(pred_np.reshape(-1))
        test_true_seg.append(seg_np)
        test_pred_seg.append(pred_np)
        test_label_seg.append(label.reshape(-1))
        # visiualization
       # visualization(args.visu, args.visu_format, data, pred, seg, label, partseg_colors, args.class_choice) 
    if visual_warning and args.visu != '':
        print('Visualization Failed: You can only choose a point cloud shape to visualize within the scope of the test class')
    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    test_true_seg = np.concatenate(test_true_seg, axis=0)
    test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    test_label_seg = np.concatenate(test_label_seg)
    test_ious,test_ious0,test_ious1 = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f,test iou0: %.6f,test iou1: %.6f' % (test_acc,
                                                                             avg_per_class_acc,
                                                                             np.mean(test_ious), np.mean(test_ious0), np.mean(test_ious1))
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='None', metavar='N',
                        choices=['shapenetpart'])
    parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                        choices=['Wheat'])
    parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default=' ', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--visu', type=str, default='',
                        help='visualize the model')
    parser.add_argument('--visu_format', type=str, default='ply',
                        help='file format of visualization')
    args = parser.parse_args()

    _init_() 


    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    
   # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
