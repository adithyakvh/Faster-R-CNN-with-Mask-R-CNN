import torch
import torch.nn.functional as F
from torch import nn
from utils import *
import torchvision

from torch import nn

from dataset import *
from utils import *
from pretrained_models import *
from BoxHead import *

from tqdm import trange
from sklearn import metrics
from torchvision.models.detection.image_list import ImageList

class MaskHead(torch.nn.Module):
    def __init__(self,Classes=3,P=14):
        super(MaskHead, self).__init__()
        self.C=Classes
        self.P=P
        # TODO initialize MaskHead
        self.MaskHead_network = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding='same'),
        nn.ReLU(True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding='same'),
        nn.ReLU(True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding='same'),
        nn.ReLU(True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding='same'),
        nn.ReLU(True),
        nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(True),
        nn.Conv2d(in_channels=256, out_channels=self.C, kernel_size=(1,1), padding='same'),
        nn.Sigmoid()
        )


    def preprocess_ground_truth_creation(self, class_logits, box_regression, gt_labels, proposals, bbox ,masks , IOU_thresh=0.5, keep_num_preNMS=1000, keep_num_postNMS=100):
        '''
        This function does the pre-prossesing of the proposals created by the Box Head (during the training of the Mask Head)
        and create the ground truth for the Mask Head
        
        Input:
              class_logits: (total_proposals,(C+1))
              box_regression: (total_proposal,4*C)  ([t_x,t_y,t_w,t_h])
              gt_labels: list:len(bz) {(n_obj)}
              bbox: list:len(bz){(n_obj, 4)}
              masks: list:len(bz){(n_obj,800,1088)}
              IOU_thresh: scalar (threshold to filter regressed with low IOU with a bounding box)
              keep_num_preNMS: scalar (number of boxes to keep pre NMS)
              keep_num_postNMS: scalar (number of boxes to keep post NMS)
        Output:
              boxes: list:len(bz){(post_NMS_boxes_per_image,4)} ([x1,y1,x2,y2] format)
              scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
              labels: list:len(bz){(post_NMS_boxes_per_image)}  (top category of each regressed box)
              gt_masks: list:len(bz){(post_NMS_boxes_per_image,2*P,2*P)}
        '''
        ####################################################
        final_boxes_postNMS, final_scores_postNMS, final_label_postNMS, final_masks_postNMS  = mask_head.postprocess_detections_masks(class_logits,box_regression, proposals, masks, bbox, conf_thresh=0.5, keep_num_preNMS=100, keep_num_postNMS=100)
                
        return final_boxes_postNMS, final_scores_postNMS, final_label_postNMS, final_masks_postNMS 
        # return final_boxes, scores, final_labels, gt_masks


    def flatten_inputs(self,input_list):
        '''
        general function that takes the input list of tensors and concatenates them along the first tensor dimension
        Input:
             input_list: list:len(bz){(dim1,?)}
        Output:
             output_tensor: (sum_of_dim1,?)
        '''
        output_tensor = torch.cat(input_list, dim=0)
        return output_tensor


    def postprocess_mask(self, masks_outputs, boxes, labels, image_size=(800,1088)):
        '''
        This function does the post processing for the result of the Mask Head for a batch of images. It project the predicted mask
        back to the original image size
        Use the regressed boxes to distinguish between the images
        Input:
              masks_outputs: (total_boxes,C,2*P,2*P)
              boxes: list:len(bz){(post_NMS_boxes_per_image,4)} ([x1,y1,x2,y2] format)
              labels: list:len(bz){(post_NMS_boxes_per_image)}  (top category of each regressed box)
              image_size: tuple:len(2)
        Output:
              projected_masks: list:len(bz){(post_NMS_boxes_per_image,image_size[0],image_size[1]
        '''
        projected_masks = []
        count = 0
        for i, each_label in enumerate(labels):
            
            each_masks = masks_outputs[count:(count+each_label.shape[0])]
            fin_each_masks = torch.stack([each_masks[l,one_each_label.item()] for l,one_each_label in enumerate(each_label)])
            count += each_label.shape[0]
            one_projected_mask = F.interpolate(fin_each_masks.unsqueeze(0), size=(800,1088), mode="bilinear").squeeze(0)
            one_projected_mask[one_projected_mask >= 0.5] = 1
            one_projected_mask[one_projected_mask < 0.5] = 0
            projected_masks.append(one_projected_mask)     

        return projected_masks


    def non_max_suppression(self, boxes_batch, scores_batch, labels_batch, masks_batch, keep_post_nms = 20):
        '''
        boxes_batch : {total_proposals for batch, 4} [x1, y1, x2, y2 format]
        scores_batch : {total_proposals for batch, 1}
        labels_batch : {total_proposals for batch, 1}
        '''
        final_batch_label_list = []
        final_labels_list = []
        final_scores_list = []
        final_masks_list  = []

        
        for counter, each_img in enumerate(labels_batch): # eacg_img is each image in the batch
            if each_img.shape[0] == 0:
                final_batch_label_list.append(torch.tensor([]))
                final_labels_list.append(torch.tensor([]))
                final_scores_list.append(torch.tensor([]))
                final_masks_list.append(torch.tensor([]))

                continue

            each_img        = each_img.cpu().detach().numpy()
            new_label       = np.vstack(np.stack(each_img,axis = -1))

            pedestrian_idx  = np.where(new_label[:,0] == 0)[0]
            pedestrian      = new_label[pedestrian_idx]

            trfc_lts_idx    = np.where(new_label[:,0] == 1)[0]
            trfc_lts        = new_label[trfc_lts_idx]
            
            cars_idx        = np.where(new_label[:,0] == 2)[0]
            cars            = new_label[cars_idx]

            ped_rec, trf_rec, crs_rec           = np.array([]), np.array([]), np.array([])
            label_to_add                        = np.array([])
            ped_scores, trf_scores, crs_scores  = np.array([]), np.array([]), np.array([])
            ped_masks, trf_masks, crs_masks     = np.array([]), np.array([]), np.array([])
            inner_masks                         = np.array([])

            thresh = 0.5
            if pedestrian.shape[0]!=0:
                ped_scores  = scores_batch[counter][pedestrian_idx].detach().cpu()
                ped_rec     = boxes_batch[counter][pedestrian_idx].detach().cpu()
                ped_masks   = masks_batch[counter][pedestrian_idx].detach().cpu()
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
                ped_masks   = np.delete(ped_masks, list(set(rows_to_delete)), axis = 0)
                if len(ped_rec) > keep_post_nms:
                    ped_rec = ped_rec[:keep_post_nms]
                    ped_scores = ped_scores[:keep_post_nms]
                    ped_masks = ped_masks[:keep_post_nms]
            
            if trfc_lts.shape[0]!=0:
                trf_scores  = scores_batch[counter][trfc_lts_idx].detach().cpu()
                trf_rec     = boxes_batch[counter][trfc_lts_idx].detach().cpu()
                trf_masks   = masks_batch[counter][trfc_lts_idx].detach().cpu()
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
                trf_masks   = np.delete(trf_masks, list(set(rows_to_delete)), axis = 0)
                if len(trf_rec) > keep_post_nms:
                    trf_rec     = trf_rec[:keep_post_nms]
                    trf_scores  = trf_scores[:keep_post_nms]
                    trf_masks   = trf_masks[:keep_post_nms]


            if cars.shape[0]!=0:
                crs_scores  = scores_batch[counter][cars_idx].detach().cpu()
                crs_rec     = boxes_batch[counter][cars_idx].detach().cpu()
                crs_masks   = masks_batch[counter][cars_idx].detach().cpu()
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
                crs_masks   = np.delete(crs_masks,  list(set(rows_to_delete)), axis = 0)
                if len(crs_rec) > keep_post_nms:
                    crs_rec     = crs_rec[:keep_post_nms]
                    crs_scores  = crs_scores[:keep_post_nms]
                    crs_masks   = crs_masks[:keep_post_nms]

            # print('ped',ped_rec)
            # print('car',crs_rec)
            # print('trf',trf_rec)
            inner_labels = np.hstack(((np.ones((len(ped_rec)))*1).astype(int), (np.ones((len(trf_rec)))*2).astype(int), (np.ones((len(crs_rec)))*3).astype(int)))
            inner_scores = np.hstack((ped_scores,   trf_scores,   crs_scores))
            # inner_masks  = np.hstack((ped_masks,   trf_masks,   crs_masks))

            if ped_rec.shape[0]!= 0 and trf_rec.shape[0]!= 0 and crs_rec.shape[0]!= 0:
                label_to_add = np.vstack((ped_rec, trf_rec, crs_rec))
                inner_masks  = np.vstack((ped_masks,   trf_masks,   crs_masks))

            if ped_rec.shape[0]!= 0 and trf_rec.shape[0]!= 0 and crs_rec.shape[0]== 0:
                label_to_add = np.vstack((ped_rec, trf_rec))
                inner_masks  = np.vstack((ped_masks, trf_masks))

            if ped_rec.shape[0]!= 0 and trf_rec.shape[0]== 0 and crs_rec.shape[0]!= 0:
                label_to_add = np.vstack((ped_rec, crs_rec))
                inner_masks  = np.vstack((ped_masks, crs_masks))

            if ped_rec.shape[0]== 0 and trf_rec.shape[0]!= 0 and crs_rec.shape[0]!= 0:
                label_to_add = np.vstack((crs_rec, trf_rec))
                inner_masks  = np.vstack((crs_masks, trf_masks))

            if ped_rec.shape[0]!= 0 and trf_rec.shape[0]== 0 and crs_rec.shape[0]== 0:
                label_to_add = ped_rec.cpu().detach().numpy()
                inner_masks  = ped_masks.cpu().detach().numpy()

            if ped_rec.shape[0]== 0 and trf_rec.shape[0]!= 0 and crs_rec.shape[0]== 0:
                label_to_add = trf_rec.cpu().detach().numpy()
                inner_masks  = trf_masks.cpu().detach().numpy()

            if ped_rec.shape[0]== 0 and trf_rec.shape[0]== 0 and crs_rec.shape[0]!= 0:
                label_to_add = crs_rec.cpu().detach().numpy()
                inner_masks  = crs_masks.cpu().detach().numpy()

                

            if label_to_add.shape[0] !=0:
                final_batch_label_list.append(torch.from_numpy(label_to_add))
                final_masks_list.append(torch.from_numpy(inner_masks))

                

            final_labels_list.append(torch.from_numpy(inner_labels))
            final_scores_list.append(torch.from_numpy(inner_scores))

        return final_batch_label_list, final_scores_list, final_labels_list, final_masks_list


    def postprocess_detections_masks(self, class_logits, box_regression, proposals, masks, bboxes, conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=100):
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
        conf_thresh = 0
        IOU_thresh  = 0.8
        scores_pre_NMS_batch = []
        labels_pre_NMS_batch = []
        boxes_pre_NMS_batch  = []
        masks_pre_NMS_batch  = []
        total_masks          = []
        for i, each_image in enumerate(proposals):
            
            boxes_image         = box_regression[i*200:(i+1)*200]
            class_logits_image  = class_logits[i*200:(i+1)*200]
            class_scores, class_labels = torch.max(class_logits_image, dim=1)
            class_labels        = class_labels.to(torch.int32)
            class_labels        = class_labels - 1
            non_bg_labels       = torch.where(class_labels >= 0)[0]
            # gt_masks            = torch.tensor([]).to(device)
            
            if len(non_bg_labels) != 0:
                class_labels        = class_labels[non_bg_labels]
                boxes_image         = boxes_image[non_bg_labels]
                boxes_image         = torch.stack([boxes_image[i, x*4:(x+1)*4] for i, x in enumerate(class_labels)])
                boxes_x1y1x2y2      = output_decodingd(boxes_image, each_image[non_bg_labels])
                each_proposal       = each_image[non_bg_labels]


                valid_boxes_idx     = torch.where((boxes_x1y1x2y2[:,0] >= 0) & (boxes_x1y1x2y2[:,1] >= 0) & (boxes_x1y1x2y2[:,2] < 1088) & (boxes_x1y1x2y2[:,3] < 800))[0]
                valid_boxes         = boxes_x1y1x2y2[valid_boxes_idx]
                class_logits_image  = class_logits[non_bg_labels][valid_boxes_idx]

                # keep_boxes_thresh_idx       = torch.where(class_logits_image[:, 1:] > conf_thresh)
                # valid_boxes_after_thresh    = valid_boxes[keep_boxes_thresh_idx[0]]
                # class_logits_image          = class_logits_image[keep_boxes_thresh_idx[0]]
                class_scores, class_labels  = torch.max(class_logits_image, dim=1)
                # class_labels = class_labels - 1
                sorted_scores, sorted_scores_idx = torch.sort(class_scores, descending=True)

                # if len(sorted_scores) > keep_num_preNMS:
                # sorted_scores       = sorted_scores[:keep_num_preNMS]
                # sorted_scores_idx   = sorted_scores_idx[:keep_num_preNMS]
                # class_labels        = class_labels[sorted_scores_idx[:keep_num_preNMS]]
                # boxes_sorted_image  = valid_boxes_after_thresh[sorted_scores_idx[:keep_num_preNMS]]

                # sorted_scores       = sorted_scores[:]
                # sorted_scores_idx   = sorted_scores_idx[:]
                class_labels        = class_labels[sorted_scores_idx[:]]
                boxes_sorted_image  = valid_boxes[sorted_scores_idx[:]]


                gt_boxes = torch.zeros_like(bboxes[i])
                gt_boxes[:, 0] = bboxes[i][:,0]
                gt_boxes[:, 1] = bboxes[i][:,1]
                gt_boxes[:, 2] = bboxes[i][:,2]
                gt_boxes[:, 3] = bboxes[i][:,3]

                iou = IOU(boxes_sorted_image, gt_boxes)
                iou_index = (iou > IOU_thresh).nonzero()
                prop_index = iou_index[:, 0]
                mask_index = iou_index[:, 1]


                sorted_scores       = sorted_scores[prop_index] 
                class_labels        = class_labels[prop_index]
                boxes_sorted_image  = boxes_sorted_image[prop_index]
                
                if (len(class_scores) == 0) or (len(prop_index) == 0):
                    # scores_pre_NMS_batch.append(torch.tensor([]))
                    # labels_pre_NMS_batch.append(torch.tensor([]))
                    # boxes_pre_NMS_batch .append(torch.tensor([]))
                    # masks_pre_NMS_batch .append(torch.tensor([]))
                    return [], [], [], []
                
                else:
                    # process masks
                    temp_masks = masks[i][mask_index]
                    class_masks = []
                    
                    for i in range(len(temp_masks)): #6
                        process_mask = torch.zeros((800, 1088), device='cuda:0')
                        single_box = boxes_sorted_image[i] #x1, y1, x2, y2
                        single_gt_mask = temp_masks[i]
                        process_mask[int(single_box[1]):int(single_box[3]), int(single_box[0]):int(single_box[2])] = 1
                        intersection = torch.logical_and(single_gt_mask.to(device), process_mask.to(device)).type(torch.float) 
                        intersection = torch.nn.functional.interpolate(intersection.unsqueeze(dim=0).unsqueeze(dim=0), size=(28,28), mode='nearest')
                        class_masks.append(intersection.squeeze().squeeze())
                    class_masks = torch.stack(class_masks)


                if len(class_labels) > 0:
                    scores_pre_NMS_batch.append(sorted_scores)
                    labels_pre_NMS_batch.append(class_labels)
                    boxes_pre_NMS_batch .append(boxes_sorted_image)
                    masks_pre_NMS_batch.append(class_masks)

                    
        final_boxes, final_scores, final_label, final_masks = self.non_max_suppression(boxes_pre_NMS_batch, scores_pre_NMS_batch, labels_pre_NMS_batch, masks_pre_NMS_batch)


        return final_boxes, final_scores, final_label, final_masks 
        # return final_boxes, final_scores, final_label, boxes_pre_NMS_batch, labels_pre_NMS_batch

    def compute_loss(self,mask_output,labels,gt_masks):
        '''
        Compute the total loss of the Mask Head
        Input:
             mask_output: (total_boxes,C,2*P,2*P)
             labels: (total_boxes)
             gt_masks: (total_boxes,2*P,2*P)
        Output:
             mask_loss
        '''
        mask_output_new = []
        # mask_output = torch.zeros(1,3,28,28)
        for i in range(len(labels)):
            one_mask_output = mask_output[i]
            # print('one_mask_output')
            # print(torch.where(one_mask_output == 0)[0].shape)
            mask_output_new.append(one_mask_output[int(labels[i].item()) - 1, :, :])
            # print('gt_masks')
            # print(torch.where(gt_masks[i] == 0)[0].shape)
         
        criterion = nn.BCELoss()

        mask_loss = criterion(torch.stack(mask_output_new), gt_masks)
        return mask_loss


    def forward(self, features):
        '''
        # Forward the pooled feature map Mask Head
        # Input:
        #        features: (total_boxes, 256,P,P)
        # Outputs:
        #        mask_outputs: (total_boxes,C,2*P,2*P)
        '''
        # feature_list = []
        # for i in range(len(features)):
        mask_outputs = self.MaskHead_network(features)
            # feature_list.append(mask_outputs)
        return mask_outputs



# if __name__ == '__main__':
#     pretrained_path='/home/josh/Desktop/CIS680/HW4/FasterRCNN/HW4_PartB_Code_Template/checkpoint680.pth'
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     backbone, rpn = pretrained_models_680(pretrained_path)
#     # we will need the ImageList from torchvision

#     imgs_path   = '/home/josh/Desktop/CIS680/HW4/data/hw3_mycocodata_img_comp_zlib.h5'
#     masks_path  = '/home/josh/Desktop/CIS680/HW4/data/hw3_mycocodata_mask_comp_zlib.h5'
#     labels_path = "/home/josh/Desktop/CIS680/HW4/data/hw3_mycocodata_labels_comp_zlib.npy"
#     bboxes_path = "/home/josh/Desktop/CIS680/HW4/data/hw3_mycocodata_bboxes_comp_zlib.npy"
#     paths = [imgs_path, masks_path, labels_path, bboxes_path]

#     dataset = BuildDataset(paths)

#     # Standard Dataloaders Initialization
#     full_size  = len(dataset)
#     train_size = int(full_size * 0.8)
#     test_size  = full_size - train_size

#     torch.random.manual_seed(3)
#     train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

#     # batch_size = 2
#     # # print("batch size:", batch_size)
#     # train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
#     # train_loader = train_build_loader.loader()
#     # test_build_loader = BuildDataLoader(test_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
#     # test_loader = test_build_loader.loader()

#     trainsize = 100
#     testsize  = 10

#     train_dataset = torch.utils.data.Subset(dataset, range(0,trainsize))
#     test_dataset  = torch.utils.data.Subset(dataset, range(trainsize, trainsize + testsize))

#     batch_size         = 2
#     train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#     train_loader       = train_build_loader.loader()
#     test_build_loader  = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
#     test_loader        = test_build_loader.loader()

#     # keep_topK = 200

#     # box_head      = BoxHead().to(device)
#     # mask_head     = MaskHead().to(device)

#     # for i, batch in enumerate(test_loader,0):
#     #     with torch.no_grad():
#     #         images = batch['images'].to(device)
#     #         #####################################################################################
#     #         bboxes = batch['bbox']
#     #         labels = batch['labels']
#     #         masks  = batch['masks']

#     #         # Take the features from the backbone
#     #         backout = backbone(images)

#     #         # The RPN implementation takes as first argument the following image list
#     #         im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
#     #         # Then we pass the image list and the backbone output through the rpn
#     #         rpnout = rpn(im_lis, backout)

#     #         #The final output is
#     #         # A list of proposal tensors: list:len(bz){(keep_topK,4)}
#     #         proposals     =[proposal[0:keep_topK,:] for proposal in rpnout[0]]
#     #         # A list of features produces by the backbone's FPN levels: list:len(FPN){(bz,256,H_feat,W_feat)}
#     #         fpn_feat_list = list(backout.values())

#     #         feature_vectors                                   = box_head.MultiScaleRoiAlign(fpn_feat_list,proposals)
#     #         class_logits,box_pred,                            = box_head.forward(feature_vectors, evaluate = True)
#     #         new_labels,regressor_target                       = box_head.create_ground_truth(proposals, labels, bboxes)

#     #         mask_boxes, mask_scores, mask_final_labels, mask_gt_masks = mask_head.preprocess_ground_truth_creation(class_logits, box_pred, labels, proposals, bboxes ,masks , IOU_thresh=0.5, keep_num_preNMS=200, keep_num_postNMS=100)


#     #         if len(mask_boxes[0]) == 0:
#     #             continue

#     #         mask_feature_vectors = MultiScaleRoiAlign_MaskHead(fpn_feat_list, mask_boxes, P=14).squeeze(1)
#     #         print()

#             #################################################################################################################################################################################################################################
#     box_head      = BoxHead().to(device)
#     boxhead_path='/home/josh/Desktop/CIS680/Final Project/MaskRCNN_project_extension_code/Boxhead_model/model_1_27.pth'
#     checkpoint = torch.load(boxhead_path)
#     box_head.load_state_dict(checkpoint)
#     box_head.train(False)

#     mask_head     = MaskHead().to(device)
#     opt           = torch.optim.Adam(box_head.parameters(), lr=0.01)
#     epochs        = 30

#     keep_topK     = 200

#     train_loss       = []
#     train_loss_class = []
#     train_loss_regr  = []

#     test_loss        = []
#     test_loss_class  = []
#     test_loss_regr   = []

#     t = trange(epochs, desc='Dataset', leave=True)
#     for epoch in t:
#     # for epoch in range(10):
#         train_l  = 0
#         train_lc = 0
#         train_lr = 0

#         mask_head.train(True)
#         for iter, batch in enumerate(train_loader):
#             opt.zero_grad()
#             images    = batch['images'].to(device)
#             labels    = batch['labels']
#             bboxes    = batch['bbox']
#             masks     = batch['masks']

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
#                 # print('here1')

#                 #####################################################################################
#                 feature_vectors              = box_head.MultiScaleRoiAlign(fpn_feat_list,  proposals)      
#                 class_logits, box_pred       = box_head.forward(feature_vectors, evaluate = True)
#                 # new_labels,regressor_target  = box_head.create_ground_truth(proposals,labels, bboxes)
            
#             mask_boxes, mask_scores, mask_final_labels, mask_gt_masks = mask_head.preprocess_ground_truth_creation(class_logits, box_pred, labels, proposals, bboxes ,masks , IOU_thresh=0.5, keep_num_preNMS=200, keep_num_postNMS=100)
            
#             if len(mask_boxes) == 0 :
#                 continue
            
#             # print('here2')
#             mask_feature_vectors = MultiScaleRoiAlign_MaskHead(fpn_feat_list, mask_boxes, P=14)
#             mask_feature_vectors_flatten = mask_head.flatten_inputs(mask_feature_vectors)
#             mask_output = mask_head.forward(mask_feature_vectors_flatten)
            

#             # mask_dummy = torch.zeros(1,3,28,28).to(device)
#             # mask_output.append(mask_dummy)
#             # calculate loss
#             # print('---------------------------------')

            
#             mask_labels = mask_head.flatten_inputs(mask_final_labels)
#             mask_gt_mks = mask_head.flatten_inputs(mask_gt_masks)
#             # mask_output = mask_head.flatten_inputs(mask_output)
#             if isinstance(mask_output, list):
#                 print("your object is a list !")
#             # try:
#             #     # print('trainloader')
#             #     # print('mask_output',mask_output.shape)
#             #     # print('mask_labels',mask_labels.shape)
#             #     # print('mask_gt_mks',mask_gt_mks.shape)
#             loss = mask_head.compute_loss(mask_output.to(device), mask_labels.to(device), mask_gt_mks.to(device))
#             # except:
#             #     pass

#             train_l  += loss
            
#             loss.backward()
#             opt.step()

#             del feature_vectors, class_logits, loss
#             del images, labels, bboxes, proposals, fpn_feat_list, im_lis, rpnout, backout

#         train_loss      .append(train_l.cpu().item() / len(train_loader))

#         print("Loss: {}".format(train_loss[epoch]))
        
#         #####################################################################################
#         ##############################TEST CASE #############################################
#         #####################################################################################

#         test_l  = 0
#         test_lc = 0
#         test_lr = 0

#         mask_head.eval()

#         for i, batch in enumerate(test_loader):
#             images = batch['images'].to(device)
#             bboxes = batch['bbox']
#             labels = batch['labels']
#             masks  = batch['masks']
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

#                 feature_vectors              = box_head.MultiScaleRoiAlign(fpn_feat_list,  proposals)      
                
#                 class_logits, box_pred       = box_head.forward(feature_vectors, evaluate = True)

#                 # new_labels,regressor_target  = box_head.create_ground_truth(proposals,labels, bboxes)
            
#             mask_boxes, mask_scores, mask_final_labels, mask_gt_masks = mask_head.preprocess_ground_truth_creation(class_logits, box_pred, labels, proposals, bboxes ,masks , IOU_thresh=0.5, keep_num_preNMS=200, keep_num_postNMS=100)

#             if len(mask_boxes) == 0 :
#                 continue

#             mask_feature_vectors = MultiScaleRoiAlign_MaskHead(fpn_feat_list, mask_boxes, P=14)
#             mask_feature_vectors_flatten = mask_head.flatten_inputs(mask_feature_vectors)
#             mask_output = mask_head.forward(mask_feature_vectors_flatten)
#             # mask_output = mask_head.forward(mask_feature_vectors)

#             # calculate loss
#             mask_labels = mask_head.flatten_inputs(mask_final_labels)
#             mask_gt_mks = mask_head.flatten_inputs(mask_gt_masks)
#             # mask_output = mask_head.flatten_inputs(mask_output)

#             if isinstance(mask_output, list):
#                 print("your object is a list !")

#             loss = mask_head.compute_loss(mask_output.to(device), mask_labels.to(device), mask_gt_mks.to(device))
            
#             test_l  += loss

#             del feature_vectors, class_logits, loss
#             del images, labels, bboxes, proposals, fpn_feat_list, im_lis, rpnout, backout

#         test_loss      .append(test_l.cpu().item() / len(test_loader))

#         print("Test: Loss: {}".format(test_loss[epoch]))
        
#         if(epoch%3==0):
#             torch.save(box_head.state_dict(), "./model_maskhead_{}.pth".format(epoch))

#         t.set_description("Epoch: {}, Train_Loss: {}, Test_loss : {}".format(epoch, train_loss[epoch],test_loss[epoch]), refresh=True)

#     torch.save(box_head.state_dict(), './model_1.pth')