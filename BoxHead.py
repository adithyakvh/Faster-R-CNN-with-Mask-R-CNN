import torch
import torch.nn.functional as F
import torchvision
import copy
import random

from torch import nn

from dataset import *
from utils   import *

from pretrained_models import *

from tqdm import trange
from sklearn import metrics
from torchvision.models.detection.image_list import ImageList


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



class BoxHead(torch.nn.Module):
    def __init__(self,Classes=3,P=7):
        super(BoxHead, self).__init__()
        self.C      = Classes
        self.P      = P
        self.device = device
        # TODO initialize BoxHead
        self.intermediate_layer = nn.Sequential(nn.Linear(in_features=256*self.P*self.P, out_features = 1024),
                                                nn.ReLU(),
                                                nn.Linear(in_features=1024, out_features = 1024),
                                                nn.ReLU())
        self.classifier         = nn.Sequential(nn.Linear(in_features=1024, out_features = self.C+1))
        self.regressor          = nn.Sequential(nn.Linear(in_features=1024, out_features = 4*self.C))
    

    def create_ground_truth(self,proposals, gt_labels, bbox):
        '''
        This function assigns to each proposal either a ground truth box or the background class (we assume background class is 0)
        Input:
            proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
            gt_labels: list:len(bz) {(n_obj)}
            bbox: list:len(bz){(n_obj, 4)}
        Output: (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)
            labels: (total_proposals,1) (the class that the proposal is assigned)
            regressor_target: (total_proposals,4) (target encoded in the [t_x,t_y,t_w,t_h] format)
        '''
        device           = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        labels_in        = []
        regressor_target = []
        #Iterate over length of proposals/gt/bbox
        for i, each_img in enumerate(zip(bbox, proposals, gt_labels)):
            box         = each_img[0]
            props       = each_img[1]
            gt_lab      = each_img[2]
            #Find the IOU between the boxes and the proposals and sort them according to max values.
            iou                     = IOU(props, box.to(device))
            #Sorted[0] == values, sorted[1] == indeicies
            sorted                  = torch.max(iou, dim=1)
            #Find all IOU > 0.5 
            iou_match_idx           = torch.where(sorted[0] > 0.5)[0]
            to_be_assigned          = torch.zeros((len(props))) 
            regressor_target_tba    = torch.zeros((len(props),4))
            #Loop over the sorted IOU
            for j in range(len(sorted[1])):
                # If we find that the IOU value is greater than the threshold above:
                if j in iou_match_idx:
                    #Store the ground truth labels at the corresponding index
                    to_be_assigned[j] = gt_lab.tolist()[sorted[1][j]]
                    
                    x_a = (props[j][0] + props[j][2]) / 2
                    y_a = (props[j][1] + props[j][3]) / 2
                    w_a = (props[j][2] - props[j][0])
                    h_a = (props[j][3] - props[j][1])
                    
                    x_bbox = (box[sorted[1][j]][0] + box[sorted[1][j]][2]) / 2
                    y_bbox = (box[sorted[1][j]][1] + box[sorted[1][j]][3]) / 2
                    w_bbox = (box[sorted[1][j]][2] - box[sorted[1][j]][0])
                    h_bbox = (box[sorted[1][j]][3] - box[sorted[1][j]][1])
                    
                    #tx = (x −xp)/wp, ty = (y −yp)/hp, tw = log(w/wp), th = log(h/hp)
                    #Regressor targets are of [tx,ty,tw,th]
                    regressor_target_tba[j,0] = (x_bbox - x_a)/w_a 
                    regressor_target_tba[j,1] = (y_bbox - y_a)/h_a
                    regressor_target_tba[j,2] = torch.log(w_bbox/w_a)
                    regressor_target_tba[j,3] = torch.log(h_bbox/h_a)

            labels_in.append(to_be_assigned.reshape(-1,1))
            regressor_target.append(regressor_target_tba)
        
        return torch.vstack(labels_in[:]),torch.vstack(regressor_target[:])


    def MultiScaleRoiAlign(self, fpn_feat_list,proposals,P=7):
        '''
        This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
        a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
        Input:
             fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
             proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
             P: scalar
        Output:
             feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
        '''
        # Here you can use torchvision.ops.RoIAlign check the docs
        device              = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        feature_vectors     = []
        for i in range(len(proposals)):
            k = torch.floor(4 + torch.log2(1.0/224*torch.sqrt((proposals[i][:,2] - proposals[i][:,0])*(proposals[i][:,3] - proposals[i][:,1])))) 
            k = torch.clamp(k, 2, 5)
            k = [int(item) for item in k]

            for j in range(len(proposals[i])):
                rescale_feat_map_x = 1088/fpn_feat_list[k[j]-2].shape[3]
                rescale_feat_map_y = 800/ fpn_feat_list[k[j]-2].shape[2]
                scaled_proposal = torch.zeros(4)

                scaled_proposal[0] = proposals[i][j][0]/rescale_feat_map_x
                scaled_proposal[1] = proposals[i][j][1]/rescale_feat_map_y
                scaled_proposal[2] = proposals[i][j][2]/rescale_feat_map_x
                scaled_proposal[3] = proposals[i][j][3]/rescale_feat_map_y

                proposal_feature_vector = torchvision.ops.roi_align(fpn_feat_list[k[j]-2][i].unsqueeze(0),[scaled_proposal.unsqueeze(0).to(device)],output_size=(P,P),spatial_scale=1.0)
                feature_vectors.append(proposal_feature_vector.reshape(-1))
        feature_vectors  = torch.stack(feature_vectors,dim=0)

        return feature_vectors



    def non_max_suppression(self, boxes_batch, scores_batch, labels_batch, keep_post_nms = 20):
        '''
        boxes_batch : {total_proposals for batch, 4} [x1, y1, x2, y2 format]
        scores_batch : {total_proposals for batch, 1}
        labels_batch : {total_proposals for batch, 1}
        '''
        final_batch_label_list = []
        final_labels_list = []
        final_scores_list = []
        for counter, each_img in enumerate(labels_batch): # eacg_img is each image in the batch
            ## convert to tensor here

            each_img        = each_img.cpu().detach().numpy()
            new_label       = np.vstack(np.stack(each_img,axis = -1))

            pedestrian_idx  = np.where(new_label[:,0] == 1)[0]
            pedestrian      = new_label[pedestrian_idx]

            trfc_lts_idx    = np.where(new_label[:,0] == 2)[0]
            trfc_lts        = new_label[trfc_lts_idx]
            
            cars_idx        = np.where(new_label[:,0] == 3)[0]
            cars            = new_label[cars_idx]

            ped_rec, trf_rec, crs_rec           = np.array([]), np.array([]), np.array([])
            label_to_add                        = np.array([])
            ped_scores, trf_scores, crs_scores  = np.array([]), np.array([]), np.array([])

            thresh = 0.5
            if pedestrian.shape[0]!=0:
                ped_scores  = scores_batch[counter][pedestrian_idx].detach().cpu()
                ped_rec     = boxes_batch[counter][pedestrian_idx].detach().cpu()
                ped_iou     = torchvision.ops.box_iou(ped_rec.clone(), ped_rec.clone())
                i = 0
                rows_to_delete = []
                for i in range(ped_iou.shape[0]):
                    if i not in rows_to_delete:
                        j = i+1
                        while j < ped_iou.shape[1]:
                            if (ped_iou[i,j] > thresh):
                                rows_to_delete.append(j)
                            j+=1
                ped_rec     = np.delete(ped_rec,    list(set(rows_to_delete)), axis = 0)
                ped_scores  = np.delete(ped_scores, list(set(rows_to_delete)), axis = 0)
                if len(ped_rec) > keep_post_nms:
                    ped_rec = ped_rec[:keep_post_nms]
                    ped_scores = ped_scores[:keep_post_nms]
            
            if trfc_lts.shape[0]!=0:
                trf_scores  = scores_batch[counter][trfc_lts_idx].detach().cpu()
                trf_rec     = boxes_batch[counter][trfc_lts_idx].detach().cpu()
                trf_iou     = torchvision.ops.box_iou(trf_rec.clone(), trf_rec.clone())
                i = 0
                rows_to_delete = []
                for i in range(trf_iou.shape[0]):
                    if i not in rows_to_delete:
                        j = i+1
                        while j < trf_iou.shape[1]:
                            if (trf_iou[i,j] > thresh):
                                rows_to_delete.append(j)
                            j+=1
                trf_rec     = np.delete(trf_rec,    list(set(rows_to_delete)), axis = 0)
                trf_scores  = np.delete(trf_scores, list(set(rows_to_delete)), axis = 0)
                if len(ped_rec) > keep_post_nms:
                    trf_rec     = trf_rec[:keep_post_nms]
                    trf_scores  = trf_scores[:keep_post_nms]


            if cars.shape[0]!=0:
                crs_scores  = scores_batch[counter][cars_idx].detach().cpu()
                crs_rec     = boxes_batch[counter][cars_idx].detach().cpu()
                crs_iou     = torchvision.ops.box_iou(crs_rec.clone(), crs_rec.clone())
                i = 0
                rows_to_delete = []
                for i in range(crs_iou.shape[0]):
                    if i not in rows_to_delete:
                        j = i+1
                        while j < crs_iou.shape[1]:
                            if (crs_iou[i,j] > thresh):
                                rows_to_delete.append(j)
                            j+=1
                crs_rec     = np.delete(crs_rec,    list(set(rows_to_delete)), axis = 0)
                crs_scores  = np.delete(crs_scores, list(set(rows_to_delete)), axis = 0)
                if len(ped_rec) > keep_post_nms:
                    crs_rec     = crs_rec[:keep_post_nms]
                    crs_scores  = crs_scores[:keep_post_nms]


            inner_labels = np.hstack(((np.ones((len(ped_rec)))*1).astype(int), (np.ones((len(trf_rec)))*2).astype(int), (np.ones((len(crs_rec)))*3).astype(int)))
            inner_scores = np.hstack((ped_scores,   trf_scores,   crs_scores))

            if ped_rec.shape[0]!= 0 and trf_rec.shape[0]!= 0 and crs_rec.shape[0]!= 0:
                label_to_add = np.vstack((ped_rec, trf_rec, crs_rec))

            if ped_rec.shape[0]!= 0 and trf_rec.shape[0]!= 0 and crs_rec.shape[0]== 0:
                label_to_add = np.vstack((ped_rec, trf_rec))

            if ped_rec.shape[0]!= 0 and trf_rec.shape[0]== 0 and crs_rec.shape[0]!= 0:
                label_to_add = np.vstack((ped_rec, crs_rec))

            if ped_rec.shape[0]== 0 and trf_rec.shape[0]!= 0 and crs_rec.shape[0]!= 0:
                label_to_add = np.vstack((crs_rec, trf_rec))

            if ped_rec.shape[0]!= 0 and trf_rec.shape[0]== 0 and crs_rec.shape[0]== 0:
                label_to_add = ped_rec.cpu().detach().numpy()

            if ped_rec.shape[0]== 0 and trf_rec.shape[0]!= 0 and crs_rec.shape[0]== 0:
                label_to_add = trf_rec.cpu().detach().numpy()

            if ped_rec.shape[0]== 0 and trf_rec.shape[0]== 0 and crs_rec.shape[0]!= 0:
                label_to_add = crs_rec.cpu().detach().numpy()

            if label_to_add.shape[0] !=0:
                final_batch_label_list.append(torch.from_numpy(label_to_add))
            final_labels_list.append(torch.from_numpy(inner_labels))
            final_scores_list.append(torch.from_numpy(inner_scores))

        return final_batch_label_list,final_labels_list, final_scores_list


    def postprocess_detections(self, class_logits, box_regression, proposals, conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=50):
        '''
        This function does the post processing for the results of the Box Head for a batch of images
        Use the proposals to distinguish the outputs from each image
        Input:
              class_logits: (total_proposals,(C+1))
              box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
              proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
              conf_thresh: scalar
              keep_num_preNMS: scalar (number of boxes to keep pre NMS)
              keep_num_postNMS: scalar (number of boxes to keep post NMS)
        Output:
              boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
              scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
              labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
        '''
        # conf_thresh = 0.2
        scores_pre_NMS_batch = []
        labels_pre_NMS_batch = []
        boxes_pre_NMS_batch  = []

        for i, each_image in enumerate(proposals):
            
            boxes_image         = box_regression[i*200:(i+1)*200]
            class_logits_image  = class_logits[i*200:(i+1)*200]
            class_scores, class_labels = torch.max(class_logits_image, dim=1)
            class_labels        = class_labels.to(torch.int32)
            class_labels        = class_labels - 1
            non_bg_labels       = torch.where(class_labels >= 0)[0]
            
            if len(non_bg_labels) != 0:
                class_labels        = class_labels[non_bg_labels]
                boxes_image         = boxes_image[non_bg_labels]
                boxes_image         = torch.stack([boxes_image[i, x*4:(x+1)*4] for i, x in enumerate(class_labels)])
                boxes_x1y1x2y2      = output_decodingd(boxes_image, each_image[non_bg_labels])

                valid_boxes_idx     = torch.where((boxes_x1y1x2y2[:,0] >= 0) & (boxes_x1y1x2y2[:,1] >= 0) & (boxes_x1y1x2y2[:,2] < 1088) & (boxes_x1y1x2y2[:,3] < 800))[0]
                valid_boxes         = boxes_x1y1x2y2[valid_boxes_idx]
                class_logits_image  = class_logits[valid_boxes_idx]

                keep_boxes_thresh_idx       = torch.where(class_logits_image[:, 1:] > conf_thresh)
                valid_boxes_after_thresh    = valid_boxes[keep_boxes_thresh_idx[0]]
                class_logits_image          = class_logits_image[keep_boxes_thresh_idx[0]]
                class_scores, class_labels  = torch.max(class_logits_image, dim=1)
                # class_labels = class_labels - 1
                sorted_scores, sorted_scores_idx = torch.sort(class_scores, descending=True)

                if len(sorted_scores) > keep_num_preNMS:
                    sorted_scores       = sorted_scores[:keep_num_preNMS]
                    sorted_scores_idx   = sorted_scores_idx[:keep_num_preNMS]
                    class_labels        = class_labels[sorted_scores_idx[:keep_num_preNMS]]
                    boxes_sorted_image  = valid_boxes_after_thresh[sorted_scores_idx[:keep_num_preNMS]]
                    if len(class_labels) > 0:
                        scores_pre_NMS_batch.append(sorted_scores)
                        labels_pre_NMS_batch.append(class_labels)
                        boxes_pre_NMS_batch .append(boxes_sorted_image)
                else:
                    sorted_scores       = sorted_scores
                    sorted_scores_idx   = sorted_scores_idx
                    class_labels        = class_labels[sorted_scores_idx]
                    boxes_sorted_image  = valid_boxes_after_thresh[sorted_scores_idx]
                    if len(class_labels) > 0:
                        scores_pre_NMS_batch.append(sorted_scores)
                        labels_pre_NMS_batch.append(class_labels)
                        boxes_pre_NMS_batch .append(boxes_sorted_image)
        
        if type(labels_pre_NMS_batch) == list and len(labels_pre_NMS_batch) != 0:
                if torch.stack(labels_pre_NMS_batch[:]).shape[1] != 0:
                    final_boxes, final_label, final_scores = self.non_max_suppression(boxes_pre_NMS_batch, scores_pre_NMS_batch, labels_pre_NMS_batch)
                else:
                    final_boxes  = [torch.tensor([]) for x in range(len(proposals))]
                    final_label  = [torch.tensor([]) for x in range(len(proposals))]
                    final_scores = [torch.tensor([]) for x in range(len(proposals))]
        
        else:
            final_boxes  = [torch.tensor([]) for x in range(len(proposals))]
            final_label  = [torch.tensor([]) for x in range(len(proposals))]
            final_scores = [torch.tensor([]) for x in range(len(proposals))]

        return final_boxes, final_scores, final_label, boxes_pre_NMS_batch, labels_pre_NMS_batch

    def postprocess_detections_map_scores(self, class_logits, box_regression, proposals, conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=50):
        '''
        This function does the post processing for the results of the Box Head for a batch of images
        Use the proposals to distinguish the outputs from each image
        Input:
              class_logits: (total_proposals,(C+1))
              box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
              proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
              conf_thresh: scalar
              keep_num_preNMS: scalar (number of boxes to keep pre NMS)
              keep_num_postNMS: scalar (number of boxes to keep post NMS)
        Output:
              boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
              scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
              labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
        '''
        # conf_thresh = 0.2
        scores_pre_NMS_batch = []
        labels_pre_NMS_batch = []
        boxes_pre_NMS_batch  = []

        for i, each_image in enumerate(proposals):
            
            boxes_image         = box_regression[i*200:(i+1)*200]
            class_logits_image  = class_logits[i*200:(i+1)*200]
            class_scores, class_labels = torch.max(class_logits_image, dim=1)
            class_labels        = class_labels.to(torch.int32)
            class_labels        = class_labels - 1
            non_bg_labels       = torch.where(class_labels >= 0)[0]


            
            if len(non_bg_labels) != 0:
                class_labels        = class_labels[non_bg_labels]
                boxes_image         = boxes_image[non_bg_labels]
                boxes_image         = torch.stack([boxes_image[i, x*4:(x+1)*4] for i, x in enumerate(class_labels)])
                boxes_x1y1x2y2      = output_decodingd(boxes_image, each_image[non_bg_labels])

                valid_boxes_idx     = torch.where((boxes_x1y1x2y2[:,0] >= 0) & (boxes_x1y1x2y2[:,1] >= 0) & (boxes_x1y1x2y2[:,2] < 1088) & (boxes_x1y1x2y2[:,3] < 800))[0]
                valid_boxes         = boxes_x1y1x2y2[valid_boxes_idx]
                class_logits_image  = class_logits[valid_boxes_idx]

                keep_boxes_thresh_idx       = torch.where(class_logits_image[:, 1:] > conf_thresh)
                valid_boxes_after_thresh    = valid_boxes[keep_boxes_thresh_idx[0]]
                class_logits_image          = class_logits_image[keep_boxes_thresh_idx[0]]
                class_scores, class_labels  = torch.max(class_logits_image, dim=1)
                # class_labels = class_labels - 1
                sorted_scores, sorted_scores_idx = torch.sort(class_scores, descending=True)

                if len(sorted_scores) > keep_num_preNMS:
                    sorted_scores       = sorted_scores[:keep_num_preNMS]
                    sorted_scores_idx   = sorted_scores_idx[:keep_num_preNMS]
                    class_labels        = class_labels[sorted_scores_idx[:keep_num_preNMS]]
                    boxes_sorted_image  = valid_boxes_after_thresh[sorted_scores_idx[:keep_num_preNMS]]
                    if len(class_labels) > 0:
                        scores_pre_NMS_batch.append(sorted_scores)
                        labels_pre_NMS_batch.append(class_labels)
                        boxes_pre_NMS_batch .append(boxes_sorted_image)
                else:
                    sorted_scores       = sorted_scores
                    sorted_scores_idx   = sorted_scores_idx
                    class_labels        = class_labels[sorted_scores_idx]
                    boxes_sorted_image  = valid_boxes_after_thresh[sorted_scores_idx]
                    if len(class_labels) > 0:
                        scores_pre_NMS_batch.append(sorted_scores)
                        labels_pre_NMS_batch.append(class_labels)
                        boxes_pre_NMS_batch .append(boxes_sorted_image)

            else:
                continue

        final_boxes, final_label, final_scores = self.non_max_suppression(boxes_pre_NMS_batch, scores_pre_NMS_batch, labels_pre_NMS_batch)

        return final_label, final_scores, final_boxes


    def compute_loss(self,class_logits, box_preds, labels, regression_targets,l=1,effective_batch=150):
        '''
        Compute the total loss of the classifier and the regressor
        Input:
             class_logits: (total_proposals,(C+1)) (as outputed from forward, not passed from softmax so we can use CrossEntropyLoss)
             box_preds: (total_proposals,4*C)      (as outputed from forward)
             labels: (total_proposals,1)
             regression_targets: (total_proposals,4)
             l: scalar (weighting of the two losses)
             effective_batch: scalar
        Outpus:
             loss: scalar
             loss_class: scalar
             loss_regr: scalar
        '''
        M = effective_batch
   
        positive_indexes = torch.where(labels >  0)[0]
        negative_indexes  = torch.where(labels == 0)[0]

        if labels.shape[0] >= (3*effective_batch)/4:
            positive_indexes  = positive_indexes[:int((3*effective_batch)/4)]
            negative_indexes  = negative_indexes[:int((effective_batch)/4)]
           
        p_idx             = int(min(positive_indexes.shape[0], effective_batch/2))
        n_idx             = int(effective_batch - p_idx)

        positive_indexes  = positive_indexes[:p_idx]
        negative_indexes  = negative_indexes[:n_idx]

        rand_p_idx        = torch.randperm(positive_indexes.shape[0])
        rand_n_idx        = torch.randperm(negative_indexes.shape[0])
        final_pos_indexes = (positive_indexes[rand_p_idx])            
        final_neg_indexes = (negative_indexes[rand_n_idx])
        criterion         = nn.CrossEntropyLoss()
        
        final_gt_pos      = labels[final_pos_indexes]
        final_gt_neg      = labels[final_neg_indexes]
        final_logits_pos  = class_logits[final_pos_indexes]
        final_logits_neg  = class_logits[final_neg_indexes]

        loss_class = 0

        final_gt_pos_new  = torch.nn.functional.one_hot(final_gt_pos.to(torch.int64).flatten(), num_classes = 4)
        final_gt_neg_new  = torch.nn.functional.one_hot(final_gt_neg.to(torch.int64).flatten(), num_classes = 4)
        
        loss_class_pos    = criterion(final_logits_pos, final_gt_pos_new.to(torch.float32).to(device))
        loss_class_neg    = criterion(final_logits_neg, final_gt_neg_new.to(torch.float32).to(device))
        
        loss_class        = loss_class_pos + loss_class_neg
        ###############################################################################################
        ###################################   Regression Loss       ###################################

        regress_pos             = box_preds[final_pos_indexes]
        regress_pos             = regress_pos.reshape((regress_pos.shape[0], self.C, 4))
        target_regression_pos   = regression_targets[final_pos_indexes]
        
        loss_regr        = 0
        l1_loss          = nn.SmoothL1Loss(reduction='sum')

        for i, label_in in enumerate(labels[final_pos_indexes]):
            loss_regr+=l1_loss(regress_pos[i, int(label_in)-1, :], target_regression_pos[i].to(device))

        loss = loss_class + (l * loss_regr)

        # print('loss: {}, loss_c: {}, loss_r :{} '.format(loss, loss_class, loss_regr))
        return loss, loss_class, loss_regr

    ########################################################################################################

    def forward(self, feature_vectors, evaluate = False):
        '''
        # Outputs:
        #        class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus background, notice if you want to use
        #                                               CrossEntropyLoss you should not pass the output through softmax here)
        #        box_pred:     (total_proposals,4*C)
        '''
        x            = self.intermediate_layer(feature_vectors)
        class_logits = self.classifier(x)
        box_pred     = self.regressor(x)

        if evaluate:
            softmax = torch.nn.Softmax(dim = 1)
            class_logits = softmax(class_logits)

        return class_logits, box_pred


# if __name__ == '__main__':
# #     ##########################################################################################################################################################################
# #     ###############################################################      INITIALIZATION              #########################################################################
# #     ##########################################################################################################################################################################

#     pretrained_path='/home/josh/Desktop/CIS680/HW4/FasterRCNN/HW4_PartB_Code_Template/checkpoint680.pth'
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     backbone, rpn = pretrained_models_680(pretrained_path)

#     # we will need the ImageList from torchvision
    


#     imgs_path   = '/home/josh/Desktop/CIS680/HW4/data/hw3_mycocodata_img_comp_zlib.h5'
#     masks_path  = '/home/josh/Desktop/CIS680/HW4/data/hw3_mycocodata_mask_comp_zlib.h5'
#     labels_path = "/home/josh/Desktop/CIS680/HW4/data/hw3_mycocodata_labels_comp_zlib.npy"
#     bboxes_path = "/home/josh/Desktop/CIS680/HW4/data/hw3_mycocodata_bboxes_comp_zlib.npy"

#     paths = [imgs_path, masks_path, labels_path, bboxes_path]
#     # load the data into data.Dataset
#     dataset = BuildDataset(paths)

#     # Standard Dataloaders Initialization
#     full_size  = len(dataset)
#     train_size = int(full_size * 0.8)
#     test_size  = full_size - train_size

#     torch.random.manual_seed(1)
#     train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

#     batch_size = 1
#     # print("batch size:", batch_size)
#     train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
#     train_loader = train_build_loader.loader()
#     test_build_loader = BuildDataLoader(test_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
#     test_loader = test_build_loader.loader()

#     box_head      = BoxHead().to(device)
#     # # opt           = torch.optim.SGD(box_head.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9)
#     # # opt             = torch.optim.SGD(box_head.parameters(),lr = 0.002,weight_decay=1.0e-4,momentum=0.90)
#     # # scheduler       = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[25,35], gamma=0.20)
#     # # lr            = 0.0007
#     # # opt           = torch.optim.Adam(box_head.parameters(), lr = 0.0001)
#     keep_topK     = 200

#     ##########################################################################################################################################################################
#     ###############################################################      TRAINING LOOP               #########################################################################
#     ##########################################################################################################################################################################

#     train_loss       = []
#     train_loss_class = []
#     train_loss_regr  = []

#     test_loss       = []
#     test_loss_class = []
#     test_loss_regr  = []

    
#     t = trange(40, desc='Dataset', leave=True)
#     for epoch in t:
#         train_l  = 0
#         train_lc = 0
#         train_lr = 0
#         #####################################################################################
#         ##############################TRAIN CASE ############################################
#         #####################################################################################
#         box_head.train(True)
#         print("Epoch {}".format(epoch))
#         for iter, batch in enumerate(train_loader, 0):
#             opt.zero_grad()
#             images    = batch['images'].to(device)
#             labels    = batch['labels']
#             bboxes    = batch['bbox']
            
#             # Take the features from the backbone
#             backout = backbone(images)

        
#             # The RPN implementation takes as first argument the following image list
#             im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
#             # Then we pass the image list and the backbone output through the rpn
#             rpnout = rpn(im_lis, backout)

#             #The final output is
#             # A list of proposal tensors: list:len(bz){(keep_topK,4)}
        

#             proposals=[proposal[0:keep_topK,:] for proposal in rpnout[0]]
#             # A list of features produces by the backbone's FPN levels: list:len(FPN){(bz,256,H_feat,W_feat)}
#             fpn_feat_list= list(backout.values())
            
#             ##########################################################################################################################################################################
#             new_labels,regressor_target  = box_head.create_ground_truth(proposals,labels, bboxes)
#             feature_vectors              = box_head.MultiScaleRoiAlign(fpn_feat_list,  proposals)
#             class_logits, box_preds      = box_head.forward(feature_vectors.detach())
#             loss, loss_c, loss_r         = box_head.compute_loss(class_logits, box_preds, new_labels, regressor_target, l=0.2, effective_batch=150)
#             ##########################################################################################################################################################################
            
#             train_l  += loss.item()
#             train_lc += loss_c.item()
#             train_lr += loss_r.item()

            
#             loss.backward()
#             opt.step()
#             box_head.train(False)

#             del feature_vectors, new_labels, regressor_target, class_logits, box_preds, loss, loss_c, loss_r
#             del images, labels, bboxes, proposals, fpn_feat_list, im_lis, rpnout, backout


#         train_loss      .append(train_l / len(train_loader))
#         train_loss_class.append(train_lc/ len(train_loader))
#         train_loss_regr .append(train_lr/ len(train_loader))

#         # print("Loss: {}, Class_loss : {}, Reg_loss:{}".format(train_loss[epoch],train_loss_class[epoch],train_loss_regr[epoch]))
        
#         #####################################################################################
#         ##############################TEST CASE #############################################
#         #####################################################################################

#         test_l = 0
#         test_lc = 0
#         test_lr = 0

#         box_head.train(False)
#         box_head.eval()

#         for i, batch in enumerate(test_loader):
#             images = batch['images'].to(device)
#             bboxes = batch['bbox']
#             labels = batch['labels']
#             #####################################################################################
#             with torch.no_grad():
#                 # Take the features from the backbone
#                 backout = backbone(images)

#                 # The RPN implementation takes as first argument the following image list
#                 im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
#                 # Then we pass the image list and the backbone output through the rpn
#                 rpnout = rpn(im_lis, backout)

#                 #The final output is
#                 # A list of proposal tensors: list:len(bz){(keep_topK,4)}
#                 proposals=[proposal[0:keep_topK,:] for proposal in rpnout[0]]
#                 # A list of features produces by the backbone's FPN levels: list:len(FPN){(bz,256,H_feat,W_feat)}
#                 fpn_feat_list= list(backout.values())

#                 new_labels,regressor_target  = box_head.create_ground_truth(proposals,labels, bboxes)
#                 feature_vectors              = box_head.MultiScaleRoiAlign(fpn_feat_list,  proposals)
#                 class_logits,box_preds=box_head(feature_vectors)
#                 loss, loss_c, loss_r = box_head.compute_loss(class_logits, box_preds, new_labels, regressor_target, l=0.2, effective_batch=150)
            
#             test_l  += loss
#             test_lc += loss_c
#             test_lr += loss_r

#             del feature_vectors, new_labels, regressor_target, class_logits, box_preds, loss, loss_c, loss_r
#             del images, labels, bboxes, proposals, fpn_feat_list, im_lis, rpnout, backout

#         test_loss      .append(test_l.cpu().item() / len(test_loader))
#         test_loss_class.append(test_lc.cpu().item()/ len(test_loader))
#         test_loss_regr .append(test_lr.cpu().item()/ len(test_loader))
        
#         if(epoch%3==0):
#             torch.save(box_head.state_dict(), "./model_4B_v6_2_epoch{}.pth".format(epoch))

#         t.set_description("Epoch: {}, Train_Loss: {}, Test_loss : {}".format(epoch, train_loss[epoch],test_loss[epoch]), refresh=True)

#     torch.save(box_head.state_dict(), './model_4B_v6_2.pth')


#     ##########################################################################################################################################################################
#     ###############################################################       MAP SCORE                  #########################################################################
#     ##########################################################################################################################################################################



    # train_model_path="/home/josh/Desktop/CIS680/HW4/FasterRCNN/HW4_PartB_Code_Template/model_1.pth"
    # checkpoint = torch.load(train_model_path)
    # # reload models
    # box_head.load_state_dict(checkpoint)

    # keep_topK = 200
    # ##########################################################################################################################################################################
    # annotations = ["vehicle", "person", "animal"]
    # box_head.eval()
    # for i, batch in enumerate(test_loader,0):
    #     with torch.no_grad():
    #         images = batch['images'].to(device)
    #         #####################################################################################
    #         bboxes = batch['bbox']
    #         labels = batch['labels']

    #         # Take the features from the backbone
    #         backout = backbone(images)

    #         # The RPN implementation takes as first argument the following image list
    #         im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
    #         # Then we pass the image list and the backbone output through the rpn
    #         rpnout = rpn(im_lis, backout)

    #         #The final output is
    #         # A list of proposal tensors: list:len(bz){(keep_topK,4)}
    #         proposals     =[proposal[0:keep_topK,:] for proposal in rpnout[0]]
    #         # A list of features produces by the backbone's FPN levels: list:len(FPN){(bz,256,H_feat,W_feat)}
    #         fpn_feat_list = list(backout.values())

    #         feature_vectors                 = box_head.MultiScaleRoiAlign(fpn_feat_list,proposals)
    #         class_logits,box_pred,          = box_head.forward(feature_vectors, evaluate = True)
    #         new_labels,regressor_target     = box_head.create_ground_truth(proposals, labels, bboxes)
    #         boxes,scores,labels,final_boxes, final_labels = box_head.postprocess_detections(class_logits,box_pred,proposals,conf_thresh=0.5, keep_num_preNMS=200, keep_num_postNMS=3)

    #     proposal_new  = torch.stack(proposals,dim=0)
    #     proposal_new  = proposal_new.reshape((-1, proposal_new.shape[2])).to(device)
    #     new_boxes = output_decodingd(regressor_target.to(device), proposal_new.to(device), device=device)

    #     img_squeeze = transforms.functional.normalize(images[0].to('cpu'),
    #                                                     [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    #                                                     [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
    #     fig,ax=plt.subplots(1,1)
    #     ax.imshow(img_squeeze.permute(1,2,0))
        
    #     for bboxes_list, labels_list in zip(final_boxes, final_labels):
    #         bboxes_list = bboxes_list.cpu().detach().numpy()
    #         for idx in range(len(bboxes_list)):
    #             cmap = ""
    #             col = ''
    #             if labels_list[idx] == 1:
    #                 cmap = "jet"
    #                 col = 'b'
    #             elif labels_list[idx] == 2:
    #                 col = 'g'
    #                 cmap = "ocean"
    #             elif labels_list[idx] == 3:
    #                 col = 'r'
    #                 cmap = "prism"

    #             rect = patches.Rectangle((bboxes_list[idx][0],bboxes_list[idx][1]),bboxes_list[idx][2]-bboxes_list[idx][0],bboxes_list[idx][3]-bboxes_list[idx][1],fill=False,color=col)
    #             ax.add_patch(rect)
                # ax.annotate(annotations[labels_list[idx] - 1], (bboxes_list[idx][0] + 40, bboxes_list[idx][1] - 50), color=col, weight='bold', fontsize=14, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5,edgecolor="white"))

        # for box_final, labels_final in zip(new_boxes[torch.where(new_labels > 1)[0]], new_labels[torch.where(new_labels > 1)[0]]):
        #     box_final = box_final.cpu().detach().numpy()
        #     labels_final = int(labels_final.cpu().detach().numpy())

        #     cmap = ""
        #     col = ''
        #     if labels_final == 3:
        #         cmap = "jet"
        #         col = 'b'
        #     elif labels_final == 2:
        #         col = 'g'
        #         cmap = "ocean"
        #     elif labels_final == 1:
        #         col = 'r'
        #         cmap = "prism"
            
        #     rect=patches.Rectangle((box_final[0],box_final[1]),box_final[2]-box_final[0],box_final[3]-box_final[1],fill=False,color="white")
        #     ax.add_patch(rect)
        
        if i == 10:
            break
    # predictions_list = []
    # ground_truth_list = []
    # box_head.eval()
    # with torch.no_grad():

    #     for i, batch in enumerate(test_loader):
    #         images = batch['images'].to(device)
    #         bboxes = batch['bbox']
    #         labels = batch['labels']
    #         #####################################################################################

    #         # Take the features from the backbone
    #         backout = backbone(images)

    #         # The RPN implementation takes as first argument the following image list
    #         im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
    #         # Then we pass the image list and the backbone output through the rpn
    #         rpnout = rpn(im_lis, backout)

    #         #The final output is
    #         # A list of proposal tensors: list:len(bz){(keep_topK,4)}
    #         proposals=[proposal[0:keep_topK,:] for proposal in rpnout[0]]
    #         # A list of features produces by the backbone's FPN levels: list:len(FPN){(bz,256,H_feat,W_feat)}
    #         fpn_feat_list= list(backout.values())

    #         new_labels,regressor_target  = box_head.create_ground_truth(proposals,labels, bboxes)
    #         feature_vectors              = box_head.MultiScaleRoiAlign(fpn_feat_list,  proposals)
    #         # feature_vecs                 = feature_vectors.detach()

    #         class_logits,box_preds=box_head.forward(feature_vectors)
            
    #         loss, loss_c, loss_r = box_head.compute_loss(class_logits, box_preds, new_labels, regressor_target)
            
    #         labels_nms, scores_nms, boxes_nms = box_head.postprocess_detections_map_scores(class_logits, box_preds, proposals, conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=50)
    #         if len(boxes_nms) > 0:
    #             labels_nms = torch.hstack((labels_nms)).reshape(-1, 1)
    #             scores_nms = torch.hstack((scores_nms)).reshape(-1, 1)
    #             boxes_nms = torch.vstack(boxes_nms)
                
                
    #             predictions_tensor = torch.hstack((labels_nms, scores_nms, boxes_nms))
    #             predictions_list.append(predictions_tensor)

    #             ground_truth_tensor = torch.hstack((torch.tensor(np.hstack((labels[:]))).reshape(-1, 1), 
    #                                                 torch.vstack(bboxes)))
    #             ground_truth_list.append(ground_truth_tensor)
                
    #     map_ = mean_average_precision(predictions_list, ground_truth_list)
    #     print(map_)

#     ##########################################################################################################################################################################