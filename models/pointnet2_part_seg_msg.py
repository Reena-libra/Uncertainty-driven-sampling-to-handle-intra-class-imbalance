import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        #self.fp3 = PointNetFeaturePropagation(in_channel=1521, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        #self.fp2 = PointNetFeaturePropagation(in_channel=561, mlp=[256, 128])
        #self.fp1 = PointNetFeaturePropagation(in_channel=150+additional_channel, mlp=[128, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=135+additional_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        #print('ddg',l0_xyz.shape, l0_points.shape)
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        #cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        cls_label_one_hot = cls_label.view(B,1,1).repeat(1,1,N)
        #print('cls',cls_label_one_hot.shape)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss

class calculate_uncertainity_logits(nn.Module):
    def __init__(self):
        super(calculate_uncertainity_logits, self).__init__()

    def forward(self, pred):
        #print("Shape before operations:", pred.shape)
        
        probabilities = F.softmax(pred, dim=2)
        #print("Shape after softmax operation:",probabilities ,probabilities.shape)

        uncertainty = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=2)
        #print("Uncertainty per point shape:",uncertainty, uncertainty.shape)
        
        uncertainty1 = torch.sum(uncertainty, axis=1)
        #uncertainty = torch.sum(uncertainty_per_point).unsqueeze(0)
        #print("Final uncertainty shape:", uncertainty1.shape, uncertainty1)
        


        return uncertainty1
    
class CalculateUncertaintyLogits(nn.Module):
    def __init__(self):
        super(CalculateUncertaintyLogits, self).__init__()

    def forward(self, pred):
        # Apply softmax along the class dimension to get probabilities
        probabilities = F.softmax(pred, dim=-1)  # Shape: [batch_size, num_points, num_classes]

        # Calculate entropy for each point
        uncertainty = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=-1)  # Shape: [batch_size, num_points]
        
        # Sum the uncertainty across all points for each batch
        uncertainty1 = torch.sum(uncertainty, dim=1)  # Shape: [batch_size]
        
        return uncertainty1

    def mc_dropout_entropy(self, model, inputs, cls_label, num_samples=5):
        model.train()  # Ensure dropout is active during inference
        predictions = []
        
        with torch.no_grad():
           for i in range(num_samples):
            # Perform forward pass with inputs and class labels
               seg_pred, _ = model(inputs, cls_label)
               predictions.append(seg_pred.unsqueeze(0))  # Add a new dimension for stacking
        

        predictions = torch.cat(predictions, dim=0)  # Shape: [num_samples, batch_size, num_points, num_classes]
        #print('predictons',predictions.shape)
        # Calculate the entropy-based uncertainty for the aggregated predictions
        entropy = self.forward(predictions.mean(dim=0))  # Shape: [batch_size, num_points]
        
        return entropy

    
class CalculateUncertaintyLogits2(nn.Module):
    def __init__(self):
        super(CalculateUncertaintyLogits2, self).__init__()

    def mc_dropout_variance(self, model, inputs, cls_label, num_samples=5):
        model.train()  # Ensure dropout is active during inference
        predictions = []
        
        with torch.no_grad():
            for i in range(num_samples):
                # Perform forward pass with inputs and class labels
                seg_pred, _ = model(inputs, cls_label)  # seg_pred: [batch_size, num_points, num_classes]
                predictions.append(seg_pred.unsqueeze(0))  # Add a new dimension for stacking
        
        # Stack predictions across num_samples
        predictions = torch.cat(predictions, dim=0)  # Shape: [num_samples, batch_size, num_points, num_classes]

        # Calculate mean and variance across MC-Dropout samples (across num_samples dimension)
        variance_pred = predictions.var(dim=0)  # Variance of predictions: [batch_size, num_points, num_classes]

        # Calculate the mean variance for each sample by averaging over points and classes
        mean_variance_per_sample = variance_pred.mean(dim=[1, 2])  # Shape: [batch_size]
        
        return mean_variance_per_sample
    