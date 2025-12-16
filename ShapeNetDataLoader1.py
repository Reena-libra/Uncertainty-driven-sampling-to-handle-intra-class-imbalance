import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')


import numpy as np
import os
from torch.utils.data import Dataset
import torch
#from model.pointnet_util import farthest_point_sample, pc_normalize
import json
import h5py


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
print(BASE_DIR)
#file_path = os.path.join(BASE_DIR, 'wheat_data/TRAIN_SPLIT_Wheat.h5')
#print('flie',file_path)
class My_H5Dataset(Dataset):

    def __init__(self, file_path,normal_channel=False):
        super(My_H5Dataset, self).__init__()
        #file_path = os.path.join(BASE_DIR, 'wheat_data/TRAIN_SPLIT_Wheat1024.h5')

        h5_file = h5py.File(file_path , 'r')
        self.features = h5_file['data'][()]

        #print('self.features',self.features)
        self.labels = h5_file['label'][()]

        self.index = h5_file['pid'][()]

        #self.ears = h5_file['ear'][()]
        
        self.normal_channel = normal_channel





        #self.labels_values = h5_file['labels_values']
        #self.index_values = h5_file['index_values']
    

    def __getitem__(self, index): 
        self.features[:, 0:3] = pc_normalize(self.features[:, 0:3])
        #print('self shape',self.features.shape)
        return (torch.from_numpy(self.features[index,:]).float(),
            torch.from_numpy(self.labels[index,:]).float(),
           torch.from_numpy(self.index[index,:]).float()
           #torch.from_numpy(np.array(self.labels_values[index])),
           #torch.from_numpy(np.array(self.index_values[index]))
            )

    def __len__(self):
        return self.features.shape[0]
 


if __name__ == '__main__':
   data = My_H5Dataset('wheat_data/TRAIN_SPLIT_Wheat1024.h5',normal_channel=False)
   DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
   for point,label,seg in DataLoader:
       print('model',point.shape)
       print(label.shape)
       print(seg.shape)

##train_dataset = My_H5Dataset('wheat_data/TRAIN_SPLIT_Wheat.h5')
#train_ms = MySampler(train_dataset)
#trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
#sampler=train_ms,num_workers=2)
