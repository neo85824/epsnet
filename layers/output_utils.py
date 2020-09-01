""" Contains functions used to sanitize and prepare the output of EPSNet. """


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from data import cfg, mask_type, MEANS, STD, activation_func
from utils.augmentations import Resize
from utils import timer
import time
from .box_utils import crop, sanitize_coordinates

def postprocess(det_output, w, h, batch_idx=0, interpolation_mode='bilinear',
                visualize_lincomb=False, crop_masks=True, score_threshold=0, mask_score=True):
    """
    Postprocesses the output of EPSNet on testing mode into a format that makes sense,
    accounting for all the possible configuration settings.

    Args:
        - det_output: The lost of dicts that Detect outputs.
        - w: The real with of the image.
        - h: The real height of the image.
        - batch_idx: If you have multiple images for this batch, the image's index in the batch.
        - interpolation_mode: Can be 'nearest' | 'area' | 'bilinear' (see torch.nn.functional.interpolate)

    Returns 4 torch Tensors (in the following order):
        - classes [num_det]: The class idx for each detection.
        - scores  [num_det]: The confidence score for each detection.
        - boxes   [num_det, 4]: The bounding box for each detection in absolute point form.
        - masks   [num_det, h, w]: Full image masks for each detection.
    """
    
    dets = det_output[batch_idx]
    
    if not 'score' in dets :
        return [torch.Tensor()] * 4 # Warning, this is 4 copies of the same thing

    if score_threshold > 0:
        keep = dets['score'] > score_threshold

        for k in dets:
            if k != 'proto' and k!='segm':
                dets[k] = dets[k][keep]
        
        if dets['score'].size(0) == 0:
            return [torch.Tensor()] * 4

    # im_w and im_h when it concerns bboxes. This is a workaround hack for preserve_aspect_ratio
    b_w, b_h = (w, h)

    # Undo the padding introduced with preserve_aspect_ratio
    if cfg.preserve_aspect_ratio:
        r_w, r_h = Resize.faster_rcnn_scale(w, h, cfg.min_size, cfg.max_size)

        # Get rid of any detections whose centers are outside the image
        boxes = dets['box']
        boxes = center_size(boxes)
        s_w, s_h = (r_w/cfg.max_size, r_h/cfg.max_size)
        
        not_outside = ((boxes[:, 0] > s_w) + (boxes[:, 1] > s_h)) < 1 # not (a or b)
        for k in dets:
            if k != 'proto':
                dets[k] = dets[k][not_outside]

        # A hack to scale the bboxes to the right size
        b_w, b_h = (cfg.max_size / r_w * w, cfg.max_size / r_h * h)
    
    # Actually extract everything from dets now
    classes = dets['class']
    boxes   = dets['box']
    scores  = dets['score']
    masks   = dets['mask']
    
    if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
        # At this points masks is only the coefficients
        proto_data = dets['proto']
        
        # Test flag, do not upvote
        if cfg.mask_proto_debug:
            np.save('scripts/proto.npy', proto_data.cpu().numpy())


        if visualize_lincomb:
            display_lincomb(proto_data, masks)
        masks = torch.matmul(proto_data, masks.t())
        if mask_score:
            masks = cfg.mask_proto_mask_activation(masks)

        # Crop masks before upsampling because you know why
        if crop_masks:
            masks = crop(masks, boxes)

        # Permute into the correct output shape [num_dets, proto_h, proto_w]
        masks = masks.permute(2, 0, 1).contiguous()

        # Scale masks up to the full image
        if cfg.preserve_aspect_ratio:
            # Undo padding
            masks = masks[:, :int(r_h/cfg.max_size*proto_data.size(1)), :int(r_w/cfg.max_size*proto_data.size(2))]
        
        masks = F.interpolate(masks.unsqueeze(0), (h, w), mode=interpolation_mode, align_corners=False).squeeze(0)
        # Binarize the masks
        if mask_score:
            masks.gt_(0.5)

    if mask_score is True:
        boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], b_w, cast=False)
        boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], b_h, cast=False)
        boxes = boxes.long()

    if cfg.mask_type == mask_type.direct and cfg.eval_mask_branch:
        # Upscale masks
        full_masks = torch.zeros(masks.size(0), h, w)

        for jdx in range(masks.size(0)):
            x1, y1, x2, y2 = boxes[jdx, :]

            mask_w = x2 - x1
            mask_h = y2 - y1

            # Just in case
            if mask_w * mask_h <= 0 or mask_w < 0:
                continue
            
            mask = masks[jdx, :].view(1, 1, cfg.mask_size, cfg.mask_size)
            mask = F.interpolate(mask, (mask_h, mask_w), mode=interpolation_mode, align_corners=False)
            if mask_score:
                mask = mask.gt(0.5).float()
            full_masks[jdx, y1:y2, x1:x2] = mask
        
        masks = full_masks
    

    return classes, scores, boxes, masks

import matplotlib.pyplot as plt


def instance_logit(dets, w, h, interpolation_mode='bilinear',
                visualize_lincomb=False, crop_masks=True, score_threshold=0, overlap_thr=0.5, mask_prune=False):
    with timer.env('Postprocess'):
        classes, scores, boxes, masks = postprocess(dets, w, h, score_threshold=score_threshold, mask_score=False)

    if classes.size(0) == 0: #no predicted mask
        return None, None, None

    classes = classes.cpu().numpy().astype(int)
    scores = scores.cpu().numpy().astype(float)
    masks = masks.view(-1, h, w).cuda()
    boxes = boxes
    
    used = np.zeros((np.max(classes)+1, h, w), dtype=np.uint8)
    # used = np.zeros((h,w), dtype=np.uint8)

    keep_masks = []
    keep_boxes = []
    keep_classes = []
    # mask_prune = True
    # if mask_prune is False:
    #     return masks, boxes, classes
    # else:
    with timer.env('things mask pruning'):
        org_boxes = boxes.clone()  #after sanitization, the bbox became absolute coord, but we want to keep it relative to apply in crop function
        boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, cast=False)
        boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, cast=False)
        boxes = boxes.cpu().long().numpy()

        for i in range(masks.size(0)):

            if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) <= 0:
                continue         
            
            mask_crop = masks[i, boxes[i, 1]:boxes[i, 3], boxes[i, 0]:boxes[i, 2]].cpu().numpy() #mask logit , before activation
            mask_crop = np.array(mask_crop>0,  dtype=np.uint8)
            used_crop = used[classes[i], boxes[i, 1]:boxes[i, 3], boxes[i, 0]:boxes[i, 2]]
            

            area = mask_crop.sum()                 
            if area == 0 or (np.logical_and(used_crop >= 1, mask_crop == 1).sum() / area > overlap_thr):
                continue
           

            used[classes[i], boxes[i, 1]:boxes[i, 3], boxes[i, 0]:boxes[i, 2]] += mask_crop 
            keep_masks.append(masks[i, :, :])
            keep_boxes.append(org_boxes[i,:])
            keep_classes.append(classes[i])

        if len(keep_masks) > 0:
            ins_logits = torch.stack(keep_masks, dim=0)
            keep_boxes = torch.stack(keep_boxes, dim=0)
            return ins_logits, keep_boxes, np.array(keep_classes)
        else:
            return None, None, None

def panoptic_logit(dets, w, h, batch_idx=0, interpolation_mode='bilinear',
                visualize_lincomb=False, crop_masks=True, score_threshold=0, overlap_thr=0.5):
    ins_logit, keep_boxes, keep_classes = instance_logit(dets, w, h, score_threshold=score_threshold, overlap_thr=overlap_thr)
    n_thing = cfg.num_classes-1

    if ins_logit is not None:
        sem_logit = semantic_logit(dets[batch_idx], h, w)
        sem_logit[0] = sem_logit[-1] - torch.max(ins_logit, dim=0)[0] # unknown prediction
        return torch.cat((sem_logit[:-1], cfg.panoptic_loss_k*ins_logit), dim=0), keep_classes
    else:
        sem_logit = semantic_logit(dets[batch_idx], h, w)
        return sem_logit[:-1], None
    

def semantic_logit(dets, h ,w, interpolation_mode='bilinear', crop_boxes=None, class_idx=None):
    proto_out = dets['proto']  # (h, w , mask_dim)
    segment_coef = dets['segm'] # (channels ,h ,w)
    proto_h, proto_w, mask_dim = proto_out.size()

    with torch.no_grad():
        segment_coef = segment_coef.reshape(segment_coef.size(0), -1).mean(dim=1)
        segment_data = torch.matmul( proto_out, segment_coef.reshape(mask_dim, -1))
        segment_data = segment_data.permute(2, 0, 1).contiguous()
        upsampled_mask = F.interpolate(segment_data.unsqueeze(0) , (h, w),
                                                mode=interpolation_mode, align_corners=False).squeeze()
     
    if crop_boxes is not None:
        things_to_stuff = cfg.dataset.things_to_stuff_map
        things_conf = torch.tensor([things_to_stuff[label] for label in class_idx]).long().cuda()

        things_logit = upsampled_mask[things_conf, :, :]
        things_logit_crop = crop(things_logit.permute(1, 2, 0), crop_boxes).permute(2, 0, 1)

        return upsampled_mask, things_logit_crop
    else:
        return upsampled_mask
 

def undo_image_transformation(img, w, h):
    """
    Takes a transformed image tensor and returns a numpy ndarray that is untransformed.
    Arguments w and h are the original height and width of the image.
    """
    img_numpy = img.permute(1, 2, 0).cpu().numpy()
    img_numpy = img_numpy[:, :, (2, 1, 0)] # To BRG

    if cfg.backbone.transform.normalize:
        img_numpy = (img_numpy * np.array(STD) + np.array(MEANS)) / 255.0
    elif cfg.backbone.transform.subtract_means:
        img_numpy = (img_numpy / 255.0 + np.array(MEANS) / 255.0).astype(np.float32)
        
    img_numpy = img_numpy[:, :, (2, 1, 0)] # To RGB
    img_numpy = np.clip(img_numpy, 0, 1)

    if cfg.preserve_aspect_ratio:
        # Undo padding
        r_w, r_h = Resize.faster_rcnn_scale(w, h, cfg.min_size, cfg.max_size)
        img_numpy = img_numpy[:r_h, :r_w]

        # Undo resizing
        img_numpy = cv2.resize(img_numpy, (w,h))

    else:
        return cv2.resize(img_numpy, (w,h))


def display_lincomb(proto_data, masks):
    out_masks = torch.matmul(proto_data, masks.t())
    # out_masks = cfg.mask_proto_mask_activation(out_masks)

    for kdx in range(1):
        jdx = kdx + 0
        import matplotlib.pyplot as plt
        coeffs = masks[jdx, :].cpu().numpy()
        idx = np.argsort(-np.abs(coeffs))
        # plt.bar(list(range(idx.shape[0])), coeffs[idx])
        # plt.show()
        
        coeffs_sort = coeffs[idx]
        arr_h, arr_w = (4,8)
        proto_h, proto_w, _ = proto_data.size()
        arr_img = np.zeros([proto_h*arr_h, proto_w*arr_w])
        arr_run = np.zeros([proto_h*arr_h, proto_w*arr_w])
        test = torch.sum(proto_data, -1).cpu().numpy()

        for y in range(arr_h):
            for x in range(arr_w):
                i = arr_w * y + x

                if i == 0:
                    running_total = proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]
                else:
                    running_total += proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]

                running_total_nonlin = running_total
                if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                    running_total_nonlin = (1/(1+np.exp(-running_total_nonlin)))

                arr_img[y*proto_h:(y+1)*proto_h, x*proto_w:(x+1)*proto_w] = (proto_data[:, :, idx[i]] / torch.max(proto_data[:, :, idx[i]])).cpu().numpy() * coeffs_sort[i]
                arr_run[y*proto_h:(y+1)*proto_h, x*proto_w:(x+1)*proto_w] = (running_total_nonlin > 0.5).astype(np.float)
        plt.imshow(arr_img)
        plt.show()
        # plt.imshow(arr_run)
        # plt.show()
        # plt.imshow(test)
        # plt.show()
        plt.imshow(out_masks[:, :, jdx].cpu().numpy())
        plt.show()
