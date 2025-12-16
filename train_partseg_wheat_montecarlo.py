"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import torch.nn as nn
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
import math 
import hashlib
from pathlib import Path
import h5py
from tqdm import tqdm
import random
#from data_utils.ShapeNetDataLoader import PartNormalDataset
#from data import My_H5Dataset
from ShapeNetDataLoader1 import My_H5Dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Wheat': [0,1]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

#def hash_tensor(tensor):
    # A simple way to hash a tensor
 #   hash_obj = hashlib.sha1(tensor.byte())
  #  return hash_obj.hexdigest()

def hash_tensor(tensor):
        # Ensure the tensor is on CPU and convert to numpy to extract bytes
    tensor_bytes = tensor.cpu().numpy().tobytes()
    hash_obj = hashlib.sha256(tensor_bytes)
    return hash_obj.hexdigest()

def tensor_hash(tensor):
    """ Create a hash of a tensor for tracking purposes. """
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    #print('new_y',new_y)
    if (y.is_cuda):
        return new_y.cuda()
    return new_y



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


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_part_seg', help='model name')
    parser.add_argument('--batch_size', type=int, default=8, help='batch Size during training')
    parser.add_argument('--epoch', default=150, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=8192, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')

    return parser.parse_args()

def ensure_cpu(tensor):
    return tensor.cpu() if tensor.is_cuda else tensor

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg_a2r4k_montecarlo_t5_b8')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    #root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    #TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
    
    TRAIN_DATASET=My_H5Dataset(os.path.join(ROOT_DIR, 'wheat_a2r4k/TRAIN_A2R4K.h5'),normal_channel=False)
    TEST_DATASET=My_H5Dataset(os.path.join(ROOT_DIR, 'wheat_a2r4k/VAL_A2R4K.h5'),normal_channel=False)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    #TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_classes = 1
    num_part = 2

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #classifier=classifier.to(device)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
   # classifier= nn.DataParallel(classifier,device_ids = [0, 1])
   #classifier = nn.DataParallel(classifier)####new line
    classifier=classifier.to(device)
    #criterion = MODEL.get_loss().cuda()
    criterion=MODEL.get_loss().to(device)
    #criterion2=MODEL.calculate_uncertainity_logits().cuda()
    #criterion2=MODEL.calculate_uncertainity_logits().to(device)
    criterion3=MODEL.CalculateUncertaintyLogits2().to(device)
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size
    
    previous_accuracy = 0.0  # Start with zero or a baseline accuracy
    max_k = 4             ###### changaeble default 4
    dynamic_k = 2         ##### changebale  default 2
    
    baseline_accuracy = 0.0
    initial_min_uncertainty_threshold = 0
    initial_epoch=35
    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0
    best_cat0_avg_iou=0      #####
    best_cat1_avg_iou=0      #######
    uncertain_samples = []
    uncertain_labels = []
    uncertain_targets = []
    unique_hashes = set()
    seen_samples=set()

    for epoch in range(start_epoch, args.epoch):
        

        mean_correct = []
        #print(epoch)
        #print('acc',dynamic_k,previous_accuracy)
        all_hard_points = []
        all_hard_labels = []
        all_hard_targets = []
        all_uncertainty_scores=[]
        
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
           
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if  momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()
        '''learning one epoch'''
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
           
            #all_hard_points = []
            #all_hard_labels = []
            #all_hard_targets = []
            #print(all_hard_points)
            optimizer.zero_grad()
            points = points.data.numpy()    
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)
            points = points.transpose(2, 1) 
            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
            
           # print(i)
           # Calculate uncertainty
            #print('seg pred',seg_pred[0,0,:])

            uncertainty = criterion3.mc_dropout_variance(classifier, points, to_categorical(label, num_classes), num_samples=5)

            #print('un',uncertainty)
            _, hard_samples_indices = uncertainty.topk(k=dynamic_k, largest=True)

            if hard_samples_indices.numel() > 0 :
               
               for idx in (hard_samples_indices):
        # Extract and append each sample and its corresponding data individually
                      #print('idx',idx)
                      #print(data[idx, :, :].shape)
                      #print(data.shape,label.shape,seg.shape)
                      single_sample = points[idx, :, :].unsqueeze(0)
                      single_label = label[idx].unsqueeze(0) #if label.dim() > 0 else label  # Check dimension if label is a tensor
                      single_target = target[idx].unsqueeze(0) #if seg.dim() > 0 else seg
                      single_score = uncertainty[idx].unsqueeze(0)
                      #print('single_sample',single_sample.shape,single_label.shape,single_score.shape)

                      all_hard_points.append(single_sample)
                      all_hard_labels.append(single_label)
                      all_hard_targets.append(single_target)
                      all_uncertainty_scores.append(single_score.item()) 
           

            batch_size_new=points.shape[0]
               #print('batch',batch_size_new)
             
            


        # Compute loss with the augmented data
            #seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
            #print('seg2',seg_pred.shape)


            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (batch_size_new * args.npoint))
            loss = criterion(seg_pred, target, trans_feat)
            loss.backward()
            optimizer.step()

        
        train_instance_acc = np.mean(mean_correct)
        log_string('Train accuracy is: %.5f' % train_instance_acc)
                     
               

         
        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            shape_ious1 = {cat: [] for cat in seg_classes.keys()}
            shape_ious2 = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            classifier = classifier.eval()

            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()

                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))
                    shape_ious1[cat].append(np.mean(part_ious[0]))  ##new
                   
                    
                    shape_ious2[cat].append(np.mean(part_ious[1]))  ##new
                    #print('2 cat',shape_ious2[cat])

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
                shape_ious1[cat]=np.mean(shape_ious1[cat])    ###############
                
                shape_ious2[cat]=np.mean(shape_ious2[cat])  #################

            mean_shape_ious = np.mean(list(shape_ious.values()))
            mean_shape_ious1=np.mean(list(shape_ious1.values())) ###########
           
            mean_shape_ious2=np.mean(list(shape_ious2.values())) ################
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
            for cat in sorted(shape_ious.keys()):
                log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious

            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
            ################################################## part id mean           ###############
            test_metrics['class_cat0_avg_iou']=mean_shape_ious1                ##############
            test_metrics['class_cat1_avg_iou']=mean_shape_ious2   
            
  ##########################################################################################
            
        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f  Cat0 avg mIOU: %f    Cat1 avg mIOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou'],test_metrics['class_cat0_avg_iou'],test_metrics['class_cat1_avg_iou']))
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                 
                'class_cat0_avg_iou': test_metrics['class_cat0_avg_iou'],        
                'class_cat1_avg_iou': test_metrics['class_cat1_avg_iou'],      

                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')
        print('baseline',baseline_accuracy)    
        if test_metrics['accuracy'] >= baseline_accuracy :
              dynamic_k = compute_dynamic_k(test_metrics['accuracy'], baseline_accuracy, max_k) 
              print(dynamic_k)
              baseline_accuracy = test_metrics['accuracy']
              should_oversample = False
              #print(baseline_accuracy)
                
        elif test_metrics['accuracy'] < baseline_accuracy and epoch % 5 == 0:
             print('Performing oversampling at epoch:', epoch)
             should_oversample = True  # Enable oversampling every 20 epochs
        else:
             should_oversample = False 
                
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        if test_metrics['class_cat0_avg_iou'] > best_cat0_avg_iou:               #######
             best_cat0_avg_iou = test_metrics['class_cat0_avg_iou']
        if test_metrics['class_cat1_avg_iou'] > best_cat1_avg_iou:
             best_cat1_avg_iou = test_metrics['class_cat1_avg_iou']
       
        log_string('Best accuracy is: %.5f' % best_acc)
        log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        log_string('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
        
        log_string('Category 0 is: %.5f' % best_cat0_avg_iou)
        log_string('Category 1 is: %.5f' % best_cat1_avg_iou)
        
        logger.info('Best accuracy is: %.5f' % best_acc)
        logger.info('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        logger.info('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)

        logger.info('Category 0 is: %.5f' % best_cat0_avg_iou)
        logger.info('Category 1 is: %.5f' % best_cat1_avg_iou)
        
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

        # Perform pairwise crossover on the shuffled points
                 for i in range(0, num_selected - 1, 2):
                     idx1, idx2 = indices[i], indices[i + 1]
                     points1, labels1 = selected_hard_points[idx1], selected_hard_labels[idx1]
                     points2, labels2 = selected_hard_points[idx2], selected_hard_labels[idx2]
              else: 
                  selected_hard_points = leaf_translation(selected_hard_points)
           else:
            
              print('hi')
              #selected_hard_points = leaf_translation(selected_hard_points)
              selected_hard_points= apply_scaling(selected_hard_points)
           
        # Concatenate the augmented samples back into the dataset for the next epoch
           # Append the new hard samples to the existing features and labels
           selected_hard_points = selected_hard_points.permute(0, 2, 1).contiguous() 
           trainDataLoader.dataset.features = np.concatenate([trainDataLoader.dataset.features, selected_hard_points.cpu().numpy()], axis=0)
           trainDataLoader.dataset.labels = np.concatenate([trainDataLoader.dataset.labels, selected_hard_labels.cpu().numpy()], axis=0)
           trainDataLoader.dataset.index = np.concatenate([trainDataLoader.dataset.index, selected_hard_targets.cpu().numpy()], axis=0)
            

         
        global_epoch += 1
   

if __name__ == '__main__':
    args = parse_args()
    main(args)
