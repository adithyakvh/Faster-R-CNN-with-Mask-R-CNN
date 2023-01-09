import numpy as np
import torch
from functools import partial
import torchvision
from sklearn import metrics

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def pretrained_models_680(checkpoint_file,eval=True):
    import torchvision
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    if(eval):
        model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    backbone = model.backbone
    rpn = model.rpn

    if(eval):
        backbone.eval()
        rpn.eval()

    rpn.nms_thresh=0.6
    checkpoint = torch.load(checkpoint_file)

    backbone.load_state_dict(checkpoint['backbone'])
    rpn.load_state_dict(checkpoint['rpn'])

    return backbone, rpn

def resnet50_fpn(images,eval=True):
    eval = True

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    if(eval):
        model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    backbone = model.backbone

    if(eval):
        backbone.eval()

    fpnout = backbone(images.to(device))
    
    return fpnout

def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))

# This function compute the IOU between two set of boxes 
def IOU(boxA, boxB):
    iou = torchvision.ops.box_iou(boxA, boxB.to(device))
    return iou


def output_flattening_old(out_r,out_c,anchors):
    #######################################
    # TODO flatten the output tensors and anchors
    #######################################

    bz = out_r.squeeze(0).shape[0]
    flatten_regr = out_r.squeeze(0).permute(0,2,3,1).reshape(-1,4)
    flatten_clas = out_c.squeeze(1).reshape(-1)
    flatten_anchors = anchors.reshape(-1,4).repeat(bz,1)

    return flatten_regr, flatten_clas, flatten_anchors

def output_flattening(out_r, out_c, anchors):
    '''
    This function flattens the output of the network and the corresponding anchors
    in the sense that it concatenate  the outputs and the anchors from all the grid cells from all
    the FPN levels from all the images into 2D matrices
    Each row correspond of the 2D matrices corresponds to a specific grid cell
    Input:
          out_r: list:len(FPN){(bz,num_anchors*4,grid_size[0],grid_size[1])}
          out_c: list:len(FPN){(bz,num_anchors*1,grid_size[0],grid_size[1])}
          anchors: list:len(FPN){(num_anchors*grid_size[0]*grid_size[1],4)}
    Output:
          flatten_regr: (total_number_of_anchors*bz,4)
          flatten_clas: (total_number_of_anchors*bz)
          flatten_anchors: (total_number_of_anchors*bz,4)
    '''
    # flatten_regr_all    = []
    # flatten_clas_all    = []
    # flatten_anchors_all = []
    
    # for level_idx in range(5):

    #     bz = out_r[level_idx].squeeze(0).shape[0]
    #     flatten_regr    = out_r[level_idx].reshape(bz,3,4,out_r[level_idx].shape[-2],out_r[level_idx].shape[-1]).permute(0,1,3,4,2).reshape(-1,4)
    #     flatten_clas    = out_c[level_idx].reshape(-1)
    #     flatten_anchors = anchors[level_idx].reshape(-1,4).repeat(bz,1)
    #     flatten_regr_all.append(flatten_regr)
    #     flatten_clas_all.append(flatten_clas)
    #     flatten_anchors_all.append(flatten_anchors)
    
    # flatten_regr_all    = torch.cat(flatten_regr_all)
    # flatten_clas        = torch.cat(flatten_clas_all)
    # flatten_anchors     = torch.cat(flatten_anchors_all)

    flatten_regr_all    = []
    flatten_clas_all    = []
    flatten_anchors_all = []

    for level_idx in range(5):
        bz = out_r[level_idx].shape[0]
        flatten_regr = out_r[level_idx].permute(0,2,3,1).reshape(-1,4)
        flatten_clas = out_c[level_idx].reshape(-1)
        flatten_anchors = anchors[level_idx].reshape(-1,4).repeat(bz,1)

        flatten_regr_all.append(flatten_regr)
        flatten_clas_all.append(flatten_clas)
        flatten_anchors_all.append(flatten_anchors)

    flatten_regr    = torch.cat(flatten_regr_all)
    flatten_clas        = torch.cat(flatten_clas_all)
    flatten_anchors     = torch.cat(flatten_anchors_all)

    return flatten_regr, flatten_clas, flatten_anchors

    # return flatten_regr_all, flatten_clas, flatten_anchors  


def output_decodingd(regressed_boxes_t,flatten_proposals):
    '''
    This function decodes the output of the box head that are given in the [t_x,t_y,t_w,t_h] format
    into box coordinates where it return the upper left and lower right corner of the bbox
    Input:
          regressed_boxes_t: (total_proposals,4) ([t_x,t_y,t_w,t_h] format)
          flatten_proposals: (total_proposals,4) ([x1,y1,x2,y2] format)
    Output:
          box: (total_proposals,4) ([x1,y1,x2,y2] format)
    '''
    #COnvert Proposals from x1,y1,x2,y2 to x,y,w,h for decoding
    prop_x = (flatten_proposals[:,0] + flatten_proposals[:,2])/2.0
    prop_y = (flatten_proposals[:,1] + flatten_proposals[:,3])/2.0
    prop_w = (flatten_proposals[:,2] - flatten_proposals[:,0])
    prop_h = (flatten_proposals[:,3] - flatten_proposals[:,1]) 

    prop_xywh = torch.vstack((prop_x,prop_y,prop_w,prop_h)).T
    
    # x = (tx * wa) + xa
    x = (regressed_boxes_t[:,0] * prop_xywh[:,2]) + prop_xywh[:,0]  
    # y = (ty * ha) + ya
    y = (regressed_boxes_t[:,1] * prop_xywh[:,3]) + prop_xywh[:,1]    
    # w = wa * torch.exp(tw) 
    w = prop_xywh[:,2] * torch.exp(regressed_boxes_t[:,2])
    # h = ha * torch.exp(th)
    h = prop_xywh[:,3] * torch.exp(regressed_boxes_t[:,3])
    
    x1 = x - (w/2.0)
    y1 = y - (h/2.0)
    x2 = x + (w/2.0)
    y2 = y + (h/2.0) 

    box = torch.vstack((x1,y1,x2,y2)).T

    return box


def output_decoding(flatten_out, flatten_anchors, device='cpu'):
    '''
    This function decodes the output that are given in the [t_x,t_y,t_w,t_h] format
    into box coordinates where it returns the upper left and lower right corner of the bbox
    Input:
          flatten_out: (total_number_of_anchors*bz,4)
          flatten_anchors: (total_number_of_anchors*bz,4)
    Output:
          box: (total_number_of_anchors*bz,4)
    '''
    conv_box = torch.zeros_like(flatten_anchors)
    box = torch.zeros_like(flatten_anchors)
    conv_box[:,3] = torch.exp(flatten_out[:,3]) * flatten_anchors[:,3]
    conv_box[:,2] = torch.exp(flatten_out[:,2]) * flatten_anchors[:,2]

    conv_box[:,1] = (flatten_out[:,1] * flatten_anchors[:,3]) + flatten_anchors[:,1]
    conv_box[:,0] = (flatten_out[:,0] * flatten_anchors[:,2]) + flatten_anchors[:,0]

    box[:,0] = conv_box[:,0] - (conv_box[:,2]/2)
    box[:,1] = conv_box[:,1] - (conv_box[:,3]/2)
    box[:,2] = conv_box[:,0] + (conv_box[:,2]/2)
    box[:,3] = conv_box[:,1] + (conv_box[:,3]/2)
    return box


def MultiScaleRoiAlign_BoxHead(fpn_feat_list,proposals,P=7):
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


###################### PROPOSALS ARE INPUT FROM BOXHEAD #########################################################
def MultiScaleRoiAlign_MaskHead(fpn_feat_list, proposals,P=7):
    '''
    This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
    a (256,P,P) feature map.
    Input:
            fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
            proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
            P: scalar
    Output:
            feature_vectors: (total_proposals, 256,P,P)  (make sure the ordering of the proposals are the same as the ground truth creation)
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
            scaled_proposal    = torch.zeros(4)

            scaled_proposal[0] = proposals[i][j][0]/rescale_feat_map_x
            scaled_proposal[1] = proposals[i][j][1]/rescale_feat_map_y
            scaled_proposal[2] = proposals[i][j][2]/rescale_feat_map_x
            scaled_proposal[3] = proposals[i][j][3]/rescale_feat_map_y

            proposal_feature_vector = torchvision.ops.roi_align(fpn_feat_list[k[j]-2][i].unsqueeze(0),[scaled_proposal.unsqueeze(0).to(device)],output_size=(P,P),spatial_scale=1.0)
            feature_vectors.append(proposal_feature_vector)
    # feature_vectors  = torch.stack(feature_vectors,dim=0)
    # feature_vectors_list= torch.stack(feature_vectors,dim=0)

    return feature_vectors

def precision_recall_curve(predictions, targets, target_class):

    AP_sum = 0
    tp_fp_array               = []
    total_ground_truth_boxes  = 0
    for i in range(len(predictions)): # i represents batch number
        if len(predictions[i]) != 0:
            predictions_class       = predictions[i][torch.where(predictions[i][:, 0] == target_class)[0]]
            targets_class           = targets[i][torch.where(targets[i][:, 0] == target_class)[0]]
            iou = torchvision.ops.box_iou(predictions_class[:, 2:], targets_class[:, 1:])
            thresh                  = 0.5

            if (iou.numpy()).size > 0:
                ground_truth_boxes_tracker = []
                for row in range(iou.shape[0]):
                    max_thresh = torch.max(iou[row, :], 0)
                    ground_truth_matched = max_thresh[1]

                    if max_thresh[0].item() > thresh:
                            if ground_truth_matched.item() not in ground_truth_boxes_tracker:
                                tp_fp_array.append(torch.tensor([predictions_class[row, 1], 1, 0]))
                                ground_truth_boxes_tracker.append(ground_truth_matched.item())                  
                            else:
                                tp_fp_array.append(torch.tensor([predictions_class[row, 1], 0, 1]))
                    else:
                        tp_fp_array.append(torch.tensor([predictions_class[row, 1], 0, 1]))


        total_ground_truth_boxes += targets_class.shape[0]

    if len(tp_fp_array) == 0:
        return torch.tensor([0, 0]), torch.tensor([0, 0])


    tp_fp_array = torch.stack(tp_fp_array)
    tp_fp_array_sorted = torch.flip(tp_fp_array[tp_fp_array[:, 0].sort()[1]], dims=(0,))
    precision_list    = []
    recall_list       = []

    true_positive     = 0
    for i, each in enumerate(tp_fp_array_sorted):

        if each[1] == 1:
            true_positive += 1

        precision       = true_positive / (i+1)
        recall          = true_positive / total_ground_truth_boxes
        precision_list.append(torch.tensor(precision))
        recall_list.append(torch.tensor(recall))


    recall       = torch.stack(recall_list)
    precision    = torch.stack(precision_list)


    if len(recall) == 1 or len(precision) == 1:
        return torch.hstack((recall,torch.tensor(0))), torch.hstack((precision,torch.tensor(0)))
        
    return recall, precision



def average_precision(predictions, targets, target_class):
    recall, precision = precision_recall_curve(predictions, targets, target_class)
    AP = metrics.auc(recall, precision)
    return AP


def mean_average_precision(predictions, targets):
    classes = [1, 2, 3]
    mean_average_precision = [average_precision(predictions, targets, x) for x in classes]
    return mean_average_precision



##############