import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        #############################################
        # TODO Initialize  Dataset
        self.images     = h5py.File(path[0],'r')['data']
        self.masks      = h5py.File(path[1],'r')['data']
        self.labels     = np.load(path[2], allow_pickle=True)
        self.bboxes     = np.load(path[3], allow_pickle=True)

        self.masks_stacked = []
        self.preprocess()

        self.transform_images = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), transforms.Resize((800,1066)), transforms.Pad(padding=(11,0), fill=0, padding_mode='constant')])
        self.transform_masks  = transforms.Compose([transforms.Resize((800,1066)), transforms.Pad(padding=(11,0), fill=0, padding_mode='constant')])
        #############################################
    
    def preprocess(self):

        count=0
        gt_mask_indices=[]

        for number in self.labels:
            mask_list=[]
            for each_count in range(len(number)):
                mask_list.append(count)
                count=count+1
            gt_mask_indices.append(mask_list)


        for mask_indices in gt_mask_indices:
            self.masks_stacked.append(self.masks[mask_indices])


    def __getitem__(self, index):
        '''
        In this function for given index we rescale the image and the corresponding  masks, boxes
        and we return them as output
        output:
            transed_img
            label
            transed_mask
            transed_bbox
            index
        '''
        ################################
        # TODO return transformed images,labels,masks,boxes,index
        ################################
        image = self.images[index]        
        mask  = self.masks_stacked[index]                      
        label = self.labels[index]                                    
        bbox  = self.bboxes[index]                                 

        transed_img, transed_mask, transed_bbox = self.pre_process_batch(image, mask, bbox)

        assert transed_img.shape     == (3,800,1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]

        return transed_img, label, transed_mask, transed_bbox, index

    def pre_process_batch(self, img, mask, bbox):
        '''
        This function preprocess the given image, mask, box by rescaling them appropriately
        output:
               img: (3,800,1088)
               mask: (n_box,800,1088)
               box: (n_box,4)
        '''
        #######################################
        # TODO apply the correct transformation to the images,masks,boxes
        ######################################
        #Image Transform
        img_norm = img/255.
        img_norm = torch.tensor(img_norm, dtype = torch.float).unsqueeze(0)
        img_norm = self.transform_images(img_norm)
        # self.transform_images((img/255.).astype(np.float32).transpose())

        #Mask Transform
        mask_norm = torch.zeros((1, len(mask), 800, 1088))
        for idx in range(len(mask)):
            msk = mask[idx]/1.
            msk = torch.tensor(msk, dtype = torch.float).unsqueeze(0)
            msk = self.transform_masks(msk)
            msk[msk > 0.5] = 1
            msk[msk < 0.5] = 0
            mask_norm[:,idx] = msk

        #Box Transform
        scaled_box = np.zeros_like(bbox)
        scaled_box[:,1] = bbox[:,1] * (800/300)
        scaled_box[:,3] = bbox[:,3] * (800/300) 
        scaled_box[:,0] = (bbox[:,0] * (1066/400)) + 11
        scaled_box[:,2] = (bbox[:,2] * (1066/400)) + 11

        assert img_norm.squeeze(0).shape == (3, 800, 1088)
        assert scaled_box.shape[0] == mask_norm.squeeze(0).shape[0]

        return img_norm.squeeze(0), mask_norm.squeeze(0), scaled_box
    
    def __len__(self):
        return len(self.images)


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.num_workers = num_workers


    def collect_fn(self, batch):
        '''
        output:
         dict{images: (bz, 3, 800, 1088)
              labels: list:len(bz)
              masks: list:len(bz){(n_obj, 800,1088)}
              bbox: list:len(bz){(n_obj, 4)}
              index: list:len(bz)
        '''
        out_batch = {"images": [], "labels": [], "masks": [], "bbox": [], "index":[]}
        
        imgs = []
        for i in range(len(batch)):
            imgs.append(batch[i][0])
            out_batch["labels"].append(batch[i][1])
            out_batch["masks"] .append(batch[i][2])
            out_batch["bbox"]  .append(torch.from_numpy(batch[i][3]))
            out_batch["index"] .append(batch[i][4])

        out_batch["images"] = torch.stack(imgs[:])
        return out_batch


    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)


# if __name__ == '__main__':
