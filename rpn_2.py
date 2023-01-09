import torchvision

import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import trange
from sklearn import metrics
from torchvision.models.detection.image_list import ImageList
from functools import partial

from torch import nn, Tensor

from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks

from utils import *
from dataset import *


class RPNHead(pl.LightningModule):
    '''
    The input of the initialization of the RPN is:
    Input:
          computed_anchors: the anchors computed in the dataset
          num_anchors: the number of anchors that are assigned to each grid cell
          in_channels: number of channels of the feature maps that are outputed from the backbone
          device: the device that we will run the model
    '''
    def __init__(self, num_anchors=3, in_channels=256, device='cuda',
                 anchors_param=dict(ratio=[[1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2]],
                                    scale=[32, 64, 128, 256, 512],
                                    grid_size=[(200, 272), (100, 136), (50, 68), (25, 34), (13, 17)],
                                    stride=[4, 8, 16, 32, 64])
                 ):
        super(RPNHead,self).__init__()
        ######################################
        # TODO initialize RPN
        self.train_losses   = []
        self.val_losses     = []
        self.num_anchors    = num_anchors
        self.sy             = 800
        self.sx             = 1088
        
        self.grid_size  = anchors_param['grid_size']
        self.stride     = anchors_param['stride']
        self.scale      = anchors_param['scale']


        # TODO  Define Intermediate Layer
        self.intermediate_layer = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
            )

        # TODO  Define Proposal Classifier Head
        self.proposal_classifier_head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1*3, kernel_size=(1, 1), padding='same'),
            nn.Sigmoid()
            )

        # TODO Define Proposal Regressor Head
        self.proposal_regressor_head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=4*3, kernel_size=(1, 1), padding='same'),
            nn.Sigmoid()
            )

        #  find anchors
        self.anchors_param=anchors_param
        self.anchors=self.create_anchors(self.anchors_param['ratio'],self.anchors_param['scale'],self.anchors_param['grid_size'],self.anchors_param['stride'])
        # self.anchors_rohit=self.create_anchors_rohit(self.anchors_param['ratio'],self.anchors_param['scale'],self.anchors_param['grid_size'],self.anchors_param['stride'])
        self.ground_dict={}

        #######################################

    def forward(self, X):
        '''
        Forward each level of the FPN output through the intermediate layer and the RPN heads
        Input:
              X:            list:len(FPN){(bz,256,grid_size[0],grid_size[1])}
        Ouput:
              logits:       list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
              bbox_regs:    list:len(FPN){(bz,4*num_anchors, grid_size[0],grid_size[1])}
        '''
        logits      = []
        bbox_regs   = []
        # for i in range(len(X)):
        #     lgts, bbx = self.forward_single(X[i])   
        #     logits.append(lgts)
        #     bbox_regs.append(bbx)
        logits, bbox_regs = MultiApply(self.forward_single, X)

        return logits, bbox_regs

    def forward_single(self, feature):
        '''
        Forward a single level of the FPN output through the intermediate layer and the RPN heads
        Input:
              feature: (bz,256,grid_size[0],grid_size[1])}
        Ouput:
              logit: (bz,1*num_acnhors,grid_size[0],grid_size[1])
              bbox_regs: (bz,4*num_anchors, grid_size[0],grid_size[1])
        '''
        #TODO forward through the Intermediate layer
        inter_out = self.intermediate_layer(feature)

        #TODO forward through the Classifier Head
        logit = self.proposal_classifier_head(inter_out)

        #TODO forward through the Regressor Head
        bbox_reg  = self.proposal_regressor_head(inter_out)

        return logit, bbox_reg


    def create_anchors(self, aspect_ratio, scale, grid_size, stride):
        '''
        This function creates the anchor boxes for all FPN level
        Input:
              aspect_ratio: list:len(FPN){list:len(number_of_aspect_ratios)}
              scale:        list:len(FPN)
              grid_size:    list:len(FPN){tuple:len(2)}
              stride:        list:len(FPN)
        Output:
              anchors_list: list:len(FPN){(grid_size[0]*grid_size[1]*num_anchors,4)}
        '''
        anchors_list = []
        for i in range(len(aspect_ratio)):
            anchors_list.append(self.create_anchors_single(aspect_ratio[i], scale[i], grid_size[i], stride[i]))
        return anchors_list


    def create_anchors_single(self, aspect_ratio, scale, grid_sizes, stride):
        '''
        This function creates the anchor boxes for one FPN level
        Input:
             aspect_ratio: list:len(number_of_aspect_ratios)
             scale: scalar
             grid_size: tuple:len(2)
             stride: scalar
        Output:
              anchors: (grid_size[0]*grid_size[1]*num_acnhors,4)
        '''
        anchors = torch.zeros((grid_sizes[0]*grid_sizes[1]*len(aspect_ratio), 4))

        #For each Aspect Ratio
        for idx in range(len(aspect_ratio)):
            temp_anchors        = torch.zeros((grid_sizes[0], grid_sizes[1],4))
            xx, yy              = torch.meshgrid(torch.arange(grid_sizes[1]), torch.arange(grid_sizes[0]))
            
            temp_anchors[:,:,0] = ((xx.T * stride) + (stride/2))
            temp_anchors[:,:,1] = ((yy.T * stride) + (stride/2))
            
            h = scale/np.sqrt(aspect_ratio[idx])
            w = h * aspect_ratio[idx]

            w_grid = torch.ones((grid_sizes[0], grid_sizes[1])) * w
            h_grid = torch.ones((grid_sizes[0], grid_sizes[1])) * h

            temp_anchors[:,:,2] = w_grid
            temp_anchors[:,:,3] = h_grid

            temp_anchors = temp_anchors.reshape((grid_sizes[0] * grid_sizes[1],4))
            anchors[grid_sizes[0]*grid_sizes[1]*idx : grid_sizes[0]*grid_sizes[1]*(idx+1)] = temp_anchors

        ######################################
        assert anchors.shape == (grid_sizes[0]*grid_sizes[1]*len(aspect_ratio), 4)
        return anchors


    def get_anchors(self):
        return self.anchors

    def create_batch_truth(self, bboxes_list, indexes, image_shape):
        '''
        This function creates the ground truth for a batch of images
        Input:
             bboxes_list: list:len(bz){(number_of_boxes,4)}
             indexes: list:len(bz)
             image_shape: list:len(bz){tuple:len(2)}
        Ouput:
             ground: list: len(FPN){(bz,num_anchors,grid_size[0],grid_size[1])}
             ground_coord: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
        '''
        ground_clas_list  = []
        ground_coord_list = []


        for num in range(len(bboxes_list)):
          ground_clas,ground_coord = self.create_ground_truth(bboxes_list[num], indexes[num], self.anchors_param['grid_size'], self.anchors, image_shape)  
          ground_clas_list .append(ground_clas)
          ground_coord_list.append(ground_coord)

        ground          = []
        ground_coord    = []
        for idx2 in range(5):
            ground_temp = []
            ground_coord_temp = []
            for idx in range(len(ground_clas_list)):
                ground_temp        .append(ground_clas_list[idx][idx2])
                ground_coord_temp  .append(ground_coord_list[idx][idx2])
            
            ground.      append(torch.stack(ground_temp[:]))
            ground_coord.append(torch.stack(ground_coord_temp[:]))


        return ground, ground_coord


    def create_ground_truth(self, bboxes, index, grid_sizes, anchors, image_size):
        '''
        This function create the ground truth for one image for all the FPN levels
        It also caches the ground truth for the image using its index
        Input:
              bboxes:      (n_boxes,4)
              index:       scalar (the index of the image in the total dataset)
              grid_size:   list:len(FPN){tuple:len(2)}
              anchor_list: list:len(FPN){(num_anchors*grid_size[0]*grid_size[1],4)}
        Output:
              ground_clas:  list:len(FPN){(num_anchors,grid_size[0],grid_size[1])}
              ground_coord: list:len(FPN){(4*num_anchors,grid_size[0],grid_size[1])}
        '''
        key = str(index)
        if key in self.ground_dict:
            groundt, ground_coord = self.ground_dict[key]
            return groundt, ground_coord

        #####################################################
        num_anchors  = 3
        ground_clas  = []
        ground_coord = []
        # TODO create ground truth for a single image
        for level in range(len(grid_sizes)):
            temp_ground_clas  = torch.ones((num_anchors,  grid_sizes[level][0], grid_sizes[level][1]), dtype=torch.double) * (-1)
            temp_ground_coord = torch.ones((4*num_anchors,grid_sizes[level][0], grid_sizes[level][1]), dtype=torch.double)
            # temp_anchors = anchors[i]
            for j in range(num_anchors): 
                 
                temp_anchors = anchors[level][j*grid_sizes[level][0]*grid_sizes[level][1]:(j+1)*grid_sizes[level][0]*grid_sizes[level][1]]
                temp_anchors_reshaped = temp_anchors.reshape((grid_sizes[level][0], grid_sizes[level][1] , 4))
                # Anchors
                # anchors_xywh = anchors.permute(2,0,1).flatten(start_dim=1,end_dim=-1).T
                anchors_xywh = temp_anchors.flatten().reshape(-1,4)

                ax      = anchors_xywh[:,0]
                ay      = anchors_xywh[:,1]
                aw      = anchors_xywh[:,2]
                ah      = anchors_xywh[:,3]

                anc_x1  = ax - (aw/2.0)
                anc_y1  = ay - (ah/2.0)
                anc_x2  = ax + (aw/2.0)
                anc_y2  = ay + (ah/2.0)


                #Boxes
                bbox_xy     = bboxes
                box_x1      = bbox_xy[:,0]
                box_y1      = bbox_xy[:,1]
                box_x2      = bbox_xy[:,2]
                box_y2      = bbox_xy[:,3]
                
                bx          = (box_x1 + box_x2)/2.0
                by          = (box_y1 + box_y2)/2.0
                bw          = (box_x2 - box_x1)
                bh          = (box_y2 - box_y1) 
                bbox_xywh   = torch.vstack((bx,by,bw,bh)).T

                #Removing Cross boundary Anchor boxes
                invalid_list = torch.tensor([])
                invalid      = torch.where((anc_x1 < 0) | (anc_y1 < 0) |(anc_x2 >= 1088) | (anc_y2 >= 800))[0]
                row          = invalid // grid_sizes[level][1]
                col          = invalid % grid_sizes[level][1]
                invalid_idx  = torch.vstack((row, col)).T
                invalid_list = torch.cat((invalid_list, invalid_idx))
                temp_ground_clas[j, row, col] = -1

                valid_anchor_idx = torch.where((anc_x1 >= 0) & (anc_y1 >= 0) & (anc_x2 < 1088) & (anc_y2 < 800))[0]
                row_anc          = valid_anchor_idx // grid_sizes[level][1]
                col_anc          = valid_anchor_idx % grid_sizes[level][1]

                valid_anchor     = temp_anchors_reshaped[row_anc, col_anc, :].reshape(-1,4)

                anchors_xywh    = valid_anchor.flatten().reshape(-1,4)

                ax      = anchors_xywh[:,0]
                ay      = anchors_xywh[:,1]
                aw      = anchors_xywh[:,2]
                ah      = anchors_xywh[:,3]

                anc_x1  = ax - (aw/2.0)
                anc_y1  = ay - (ah/2.0)
                anc_x2  = ax + (aw/2.0)
                anc_y2  = ay + (ah/2.0)

                anchor_xy = torch.vstack((anc_x1, anc_y1, anc_x2, anc_y2)).T
                #######################################################
                assigned    = torch.tensor([])
                bbox_dict   = {} 

                for idx in range(len(bboxes)):

                    bbox_xy     = bboxes[idx].reshape(1, -1)
                    box_x1      = bbox_xy[:,0]
                    box_y1      = bbox_xy[:,1]
                    box_x2      = bbox_xy[:,2]
                    box_y2      = bbox_xy[:,3]
                    
                    bx          = (box_x1 + box_x2)/2.0
                    by          = (box_y1 + box_y2)/2.0
                    bw          = (box_x2 - box_x1)
                    bh          = (box_y2 - box_y1) 
                    bbox_xywh   = torch.vstack((bx,by,bw,bh)).T
                    
                    #Calculate IOU

                    IOU_matrix = torchvision.ops.box_iou(bbox_xy.to(device), anchor_xy.to(device))
                    IOU_matrix = IOU_matrix.detach().cpu()
                    IOU_max    = torch.max(IOU_matrix).item()
                    
                    #Threshold somewhere other than >0.7
                    true_box = torch.logical_or(IOU_matrix == IOU_max, IOU_matrix >= 0.7)
                    true_idx = torch.where(true_box)[1]

                    val_row  = row_anc[true_idx] #//68
                    val_col  = col_anc[true_idx] #% 68
                    valid    = torch.vstack((val_row, val_col)).T

                    temp_ground_clas[j,val_row, val_col] = 1

                    bbox_dict[idx]  = [valid, IOU_matrix[0,true_idx] , bbox_xywh]


                    #Threshold less than 0.3 and invalid
                    less_thresh_idx = torch.where((IOU_matrix < 0.3) & (~true_box))[1]
                    less_thresh_row = row_anc[less_thresh_idx]#//68
                    less_thresh_col = col_anc[less_thresh_idx]#%68
                    less_thresh     = torch.vstack((less_thresh_row, less_thresh_col)).T
                    dont_del_idx    = torch.where(~torch.isin(less_thresh, assigned).all(axis = 1))[0]
                    assigned        = torch.cat((assigned,valid))

                    temp_ground_clas[j,less_thresh[dont_del_idx,0], less_thresh[dont_del_idx,1]]  = 0
                            
                indicies    = torch.where(temp_ground_clas[j] == 1) 
                # xy_ind      = indicies[1:]
                xy_ind      = indicies[:]
                xy_ind      = torch.vstack(xy_ind[:]).T
                # ind_flat    = (indicies[1] * 68) + indicies[2]
                ind_flat    = (indicies[0] * grid_sizes[level][1]) + indicies[1]
                
                bbox_xy     = bboxes.reshape(1,-1)
            
                anchors_xywh = temp_anchors.flatten().reshape(-1,4)

                ax      = anchors_xywh[:,0]
                ay      = anchors_xywh[:,1]
                aw      = anchors_xywh[:,2]
                ah      = anchors_xywh[:,3]

                anc_x1  = ax - (aw/2.0)
                anc_y1  = ay - (ah/2.0)
                anc_x2  = ax + (aw/2.0)
                anc_y2  = ay + (ah/2.0)

                new_f_anchors = torch.vstack((anc_x1, anc_y1, anc_x2, anc_y2)).T


                IOU_matrix  = torchvision.ops.box_iou(bboxes.to(device), new_f_anchors[ind_flat].to(device))
                IOU_matrix  = IOU_matrix.detach().cpu() 
                IOU_max     = torch.max(IOU_matrix, dim=0)
                
                bbox_xy     = bboxes
                box_x1      = bbox_xy[:,0]
                box_y1      = bbox_xy[:,1]
                box_x2      = bbox_xy[:,2]
                box_y2      = bbox_xy[:,3]
                
                bx          = (box_x1 + box_x2)/2.0
                by          = (box_y1 + box_y2)/2.0
                bw          = (box_x2 - box_x1)
                bh          = (box_y2 - box_y1) 

                bbox_xywh   = torch.vstack((bx,by,bw,bh)).T


                for i, each_ind in enumerate(IOU_max[1]):

                    temp_ground_coord[(j*4)+0, indicies[0][i], indicies[1][i]] = (bbox_xywh[each_ind][0] - temp_anchors_reshaped[indicies[0][i], indicies[1][i]][0])/  temp_anchors_reshaped[indicies[0][i], indicies[1][i]][2]
                    temp_ground_coord[(j*4)+1, indicies[0][i], indicies[1][i]] = (bbox_xywh[each_ind][1] - temp_anchors_reshaped[indicies[0][i], indicies[1][i]][1])/  temp_anchors_reshaped[indicies[0][i], indicies[1][i]][3]
                    
                    temp_ground_coord[(j*4)+2, indicies[0][i], indicies[1][i]] = torch.log(bbox_xywh[each_ind][2] /  temp_anchors_reshaped[indicies[0][i], indicies[1][i]][2])
                    temp_ground_coord[(j*4)+3, indicies[0][i], indicies[1][i]] = torch.log(bbox_xywh[each_ind][3] /  temp_anchors_reshaped[indicies[0][i], indicies[1][i]][3])
            
            ground_clas.append(temp_ground_clas)
            ground_coord.append(temp_ground_coord)
        
        #####################################################

        self.ground_dict[key] = (ground_clas, ground_coord)

        # assert ground_clas.shape ==(1,grid_sizes[0],grid_sizes[1])
        # assert ground_coord.shape==(4,grid_sizes[0],grid_sizes[1])
        return ground_clas, ground_coord

    def loss_class(self, p_out, n_out):
        '''
        Compute the loss of the classifier
        Input:
             p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
             n_out:     (negatives_on_mini_batch) (output of the classifier for sampled anchors with negative gt labels
        '''
        # torch.nn.BCELoss()
        # TODO compute classifier's loss
        loss    = torch.nn.BCELoss(reduction="mean")
        t_loss  = loss(p_out, n_out.float())
        return t_loss


    def loss_reg(self, pos_target_coord, pos_out_r, non_zero = False):
        '''
        Compute the loss of the regressor
        Input:
              pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
              pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
        '''
        # torch.nn.SmoothL1Loss()
        # TODO compute regressor's loss
        loss        = torch.nn.SmoothL1Loss()
        loss_r = sum([loss(pos_target_coord[i], pos_out_r[i]) for i in range(4)]) if non_zero==False else torch.tensor(0)
        return loss_r


    def compute_loss(self, clas_out_list, regr_out_list, targ_clas_list, targ_regr_list, l=1, effective_batch=150):
        '''
        Compute the total loss for the FPN heads
        Input:
              clas_out_list: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
              regr_out_list: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
              targ_clas_list: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
              targ_regr_list: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
              l: weighting lambda between the two losses
        Output:
              loss: scalar
              loss_c: scalar
              loss_r: scalar
        '''

        loss   = 0
        loss_c = 0
        loss_r = 0
        for idx in range(len(clas_out_list)):
            targ_clas_list_temp = torch.stack(targ_clas_list[idx][:]).to(device)
            targ_regr_list_temp = torch.stack(targ_regr_list[idx][:]).to(device)

            M = effective_batch
            
            positives_indexes   = torch.where(targ_clas_list_temp == 1)
            negative_indexes    = torch.where(targ_clas_list_temp == 0)

            p_idx               = int(min(positives_indexes[0].shape[0], effective_batch/2))
            n_idx               = int(effective_batch - p_idx)

            rand_p_idx          = torch.randperm(positives_indexes[0].shape[0])
            rand_n_idx          = torch.randperm(negative_indexes[0].shape[0])

            final_pos_indexes   = (positives_indexes[0][rand_p_idx[:p_idx]], positives_indexes[1][rand_p_idx[:p_idx]], positives_indexes[2][rand_p_idx[:p_idx]], positives_indexes[3][rand_p_idx[:p_idx]])            
            final_neg_indexes   = (negative_indexes[0][rand_n_idx[:n_idx]], negative_indexes[1][rand_n_idx[:n_idx]], negative_indexes[2][rand_n_idx[:n_idx]], negative_indexes[3][rand_n_idx[:n_idx]])

            final_gt        = targ_clas_list_temp [final_pos_indexes]
            pos_class_pred  = clas_out_list[idx][final_pos_indexes]
            loss1           = self.loss_class(pos_class_pred, final_gt)

            ng_class_gt     = targ_clas_list_temp [final_neg_indexes]
            ng_class_pred   = clas_out_list[idx][final_neg_indexes]
            loss2           = self.loss_class(ng_class_pred, ng_class_gt)
                        
            loss_c          += loss1 + loss2

            ###################################################################
            targ_class      = targ_clas_list_temp.reshape(-1)
            # clas_out_2    = clas_out_list[idx].reshape(-1)
            pos_class       = (targ_class==1).nonzero()

            positives       = int( min(pos_class.shape[0], M/2))

            pos_class       = pos_class[torch.randperm(pos_class.shape[0]), :]
            pos_class       = pos_class[:positives, :]

            non_zero        = False

            if(pos_class.shape[0]==0):
                non_zero= True

            mask_targ_regr  = targ_regr_list_temp.permute(1,0,2,3).reshape(4,-1)[:, pos_class]
            mask_regr_pred  = regr_out_list[idx].permute(1,0,2,3).reshape(4,-1)[:, pos_class]
            
            loss_r          += self.loss_reg(mask_targ_regr, mask_regr_pred, non_zero)

            loss += loss_c + l*loss_r

        return loss, loss_c, loss_r


    def postprocess(self, out_c, out_r, IOU_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=3):
        '''
        Post process for the outputs for a batch of images
        Input:
              out_c: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
              out_r: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
              IOU_thresh: scalar that is the IOU threshold for the NMS
              keep_num_preNMS: number of masks we will keep from each image before the NMS
              keep_num_postNMS: number of masks we will keep from each image after the NMS
        Output:
              nms_clas_list: list:len(bz){(Post_NMS_boxes)} (the score of the boxes that the NMS kept)
              nms_prebox_list: list:len(bz){(Post_NMS_boxes,4)} (the coordinate of the boxes that the NMS kept)
        '''
        pre_nms_matrix_list = []
        scores_sorted_list  = []
        nms_clas_list       = []
        nms_prebox_list     = []

        for idx in range(len(out_c)):
            for idx2 in range(len(out_c[idx])):
                scores_sorted, pre_nms_matrix, nms_clas, nms_prebox = self.postprocessImg(out_c[idx][idx2],out_r[idx][idx2], IOU_thresh,keep_num_preNMS, keep_num_postNMS)
                scores_sorted_list. append(scores_sorted)
                pre_nms_matrix_list.append(pre_nms_matrix)
                nms_clas_list.      append(nms_clas)
                nms_prebox_list.    append(nms_prebox)

        return scores_sorted_list, pre_nms_matrix_list, nms_clas_list, nms_prebox_list


    def postprocessImg(self, mat_clas, mat_coord, IOU_thresh, keep_num_preNMS, keep_num_postNMS):
        '''
        Post process the output for one image
        Input:
             mat_clas:  list:len(FPN){(1,1*num_anchors,grid_size[0],grid_size[1])}  (score of the output boxes)
             mat_coord: list:len(FPN){(1,4*num_anchors,grid_size[0],grid_size[1])} (encoded coordinates of the output boxess)
        Output:
              nms_clas:   (Post_NMS_boxes)
              nms_prebox: (Post_NMS_boxes,4)
        '''
        anchors     = self.get_anchors()
        mat_coord   = mat_coord.permute(1,2,0)
        
        # x = (tx * wa) + xa
        x = (mat_coord[:,:, 0] * anchors[:,:,2]) + anchors[:,:,0]  

        # y = (ty * ha) + ya
        y = (mat_coord[:,:, 1] * anchors[:,:,3]) + anchors[:,:,1]  
        
        # w = wa * torch.exp(tw) 
        w = anchors[:,:,2] * torch.exp(mat_coord[:,:, 2])

        # h = ha * torch.exp(th)
        h = anchors[:,:,3] * torch.exp(mat_coord[:,:, 3])

        x1                 = x - (w/2.0)
        y1                 = y - (h/2.0)
        x2                 = x + (w/2.0)
        y2                 = y + (h/2.0) 
        decoded_boxes_xywh = torch.stack((x,y,w,h))
        decoded_boxes_xy   = torch.stack((x1,y1,x2,y2))
        decoded_boxes_xy_reshape = decoded_boxes_xy.permute(2,1,0).reshape(-1,4)

        
        valid_dec_box_idx   = torch.where((decoded_boxes_xy_reshape[:,0] >= 0) & (decoded_boxes_xy_reshape[:,1] >= 0) & (decoded_boxes_xy_reshape[:,2] < 1088) & (decoded_boxes_xy_reshape[:,3] < 800))[0]
        valid_dec_boxes     = decoded_boxes_xy_reshape[valid_dec_box_idx]
        mat_clas_flat       = mat_clas.flatten().reshape(-1)
        sorted_indicies     = torch.sort( mat_clas_flat[valid_dec_box_idx], descending=True)
        scores_sorted       = sorted_indicies[0][:keep_num_preNMS]
        pre_nms_matrix      = valid_dec_boxes[sorted_indicies[1][:keep_num_preNMS]]  
        nms_clas,nms_prebox = self.NMS(sorted_indicies[0][:keep_num_preNMS], pre_nms_matrix, thresh = 0.5)
        

        return scores_sorted, pre_nms_matrix, nms_clas[:keep_num_postNMS], nms_prebox[:keep_num_postNMS]



    def NMS(self, clas, prebox, thresh):
        '''
        Input:
              clas:   (top_k_boxes)   (scores of the top k boxes)
              prebox: (top_k_boxes,4) (coordinate of the top k boxes)
        Output:
              nms_clas:   (Post_NMS_boxes)
              nms_prebox: (Post_NMS_boxes,4)
        '''
        prebox_cp = torch.clone(prebox)
        clas_cp   = torch.clone(clas)

        ped_iou = torchvision.ops.box_iou(prebox_cp, prebox_cp)
        i = 0
        rows_to_delete = []
        for i in range(ped_iou.shape[0]):
            if i not in rows_to_delete:
                j = i+1
                while j < ped_iou.shape[1]:
                    if (ped_iou[i,j] > thresh):
                        rows_to_delete.append(j)
                    j+=1
        prebox_cp = torch.tensor(np.delete(prebox_cp.cpu().detach().numpy(), list(set(rows_to_delete)), axis = 0))
        clas_cp   = torch.tensor(np.delete(clas_cp.cpu().detach().numpy(), list(set(rows_to_delete))))

        return clas_cp, prebox_cp


    ####################################################################################################################################################

    def training_step(self, batch, batch_idx):
        # print('Here')
        images, bounding_boxes, indexes     = batch['images'], batch['bbox'], batch['index']
        with torch.no_grad():
            fpnout                          = resnet50_fpn(images)
            
        fpn_list                            = [fpnout['0'],fpnout['1'],fpnout['2'],fpnout['3'],fpnout['pool']]
        logits, bbox_regs                   = self.forward(fpn_list)
        gt,ground_coord                     = self.create_batch_truth(bounding_boxes, indexes, (800, 1088))  

        loss, loss_c, loss_r = self.compute_loss(logits, bbox_regs, gt, ground_coord)
        self.log("train_loss",              loss,       prog_bar=True)
        self.log("train_class_loss",        loss_c,     prog_bar=True)
        self.log("train_regression_loss",   loss_r,     prog_bar=True)

        # del images, labels, masks, bounding_boxes
        # del mask_ground_truths, category_ground_truths, active_masks
        # torch.cuda.empty_cache()

        return {"loss": loss, "train_class_loss": loss_c, "train_regression_loss": loss_r}

    def training_epoch_end(self, outputs):
        # print
        train_loss              = torch.tensor([output["loss"]                  for output in outputs]).mean().item()
        train_class_loss        = torch.tensor([output["train_class_loss"]      for output in outputs]).mean().item()
        train_regression_loss   = torch.tensor([output["train_regression_loss"] for output in outputs]).mean().item()
        # print(train_loss, train_class_loss, train_regression_loss)
        self.train_losses.append((train_loss, train_class_loss, train_regression_loss))

    def validation_step(self, batch, batch_idx):
        images, bounding_boxes, indexes     = batch['images'], batch['bbox'], batch['index']
        # images_re                           = torch.stack(images[:])
        with torch.no_grad():
            fpnout                          = resnet50_fpn(images)
        
        fpn_list                            = [fpnout['0'],fpnout['1'],fpnout['2'],fpnout['3'],fpnout['pool']]
        logits, bbox_regs                   = self.forward(fpn_list)
        gt,ground_coord                     = self.create_batch_truth(bounding_boxes, indexes, (800, 1088))  

        val_loss, loss_c, loss_r = self.compute_loss(logits, bbox_regs, gt, ground_coord)
        self.log("val_loss",              val_loss,   prog_bar=True)
        self.log("val_class_loss",        loss_c,     prog_bar=True)
        self.log("val_regression_loss",   loss_r,     prog_bar=True)

        # print(float(logits.flatten().shape[0]))
        # print(float(logits.flatten().shape[0]))
        logits = torch.where(logits >= 0.5, 1, 0)
        point_wise_accuracy = torch.sum(gt.flatten() == logits.flatten())/float(logits.flatten().shape[0])
        print('point_wise_accuracy', point_wise_accuracy)
        # del images, labels, masks, bounding_boxes
        # del mask_ground_truths, category_ground_truths, active_masks
        # torch.cuda.empty_cache()

        return {"val_loss": val_loss, "val_class_loss": loss_c, "val_regression_loss": loss_r}
        
    def validation_epoch_end(self, outputs):
        # print
        val_loss                = torch.tensor([output["val_loss"]            for output in outputs]).mean().item()
        val_class_loss          = torch.tensor([output["val_class_loss"]      for output in outputs]).mean().item()
        val_regression_loss     = torch.tensor([output["val_regression_loss"] for output in outputs]).mean().item()
        # print(val_loss, val_class_loss, val_regression_loss)
        self.val_losses.append((val_loss, val_class_loss, val_regression_loss))


    def configure_optimizers(self):
        opt     = torch.optim.SGD(self.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9)
        sched   = {"scheduler": torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[26, 32], gamma=0.1)}

        return {"optimizer": opt, "lr_scheduler": sched}