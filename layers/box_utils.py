# -*- coding: utf-8 -*-
import torch
from utils import timer
import math
from data import cfg

@torch.jit.script
def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


@torch.jit.script
def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat(( (boxes[:, 2:] + boxes[:, :2])/2,     # cx, cy
                        boxes[:, 2:] - boxes[:, :2]  ), 1)  # w, h

@torch.jit.script
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,A,4].
      box_b: (tensor) bounding boxes, Shape: [n,B,4].
    Return:
      (tensor) intersection area, Shape: [n,A,B].
    """
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, :, 0] * inter[:, :, :, 1]


def jaccard(box_a, box_b, iscrowd:bool=False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2]-box_a[:, :, 0]) *
              (box_a[:, :, 3]-box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2]-box_b[:, :, 0]) *
              (box_b[:, :, 3]-box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)



def change(gt, priors):
    """
    Compute the d_change metric proposed in Box2Pix:
    https://lmb.informatik.uni-freiburg.de/Publications/2018/UB18/paper-box2pix.pdf
    
    Input should be in point form (xmin, ymin, xmax, ymax).

    Output is of shape [num_gt, num_priors]
    Note this returns -change so it can be a drop in replacement for 
    """
    num_priors = priors.size(0)
    num_gt     = gt.size(0)

    gt_w = (gt[:, 2] - gt[:, 0])[:, None].expand(num_gt, num_priors)
    gt_h = (gt[:, 3] - gt[:, 1])[:, None].expand(num_gt, num_priors)

    gt_mat =     gt[:, None, :].expand(num_gt, num_priors, 4)
    pr_mat = priors[None, :, :].expand(num_gt, num_priors, 4)

    diff = gt_mat - pr_mat
    diff[:, :, 0] /= gt_w
    diff[:, :, 2] /= gt_w
    diff[:, :, 1] /= gt_h
    diff[:, :, 3] /= gt_h

    return -torch.sqrt( (diff ** 2).sum(dim=2) )

def center_distance(box_a, box_b):
    # computer the l2 distance between box_a & box_b
    # box_a, box_b : (xmin, ymin, xmax, ymax) Shape[num, 4]
    
    # compute center : Shape[num, 2]
    box_a_cen = (box_a[:, :2] + box_a[:, 2:])/2
    box_b_cen = (box_b[:, :2] + box_b[:, 2:])/2

    # L2 Distance
    a_norm = (box_a_cen**2).sum(1).view(-1, 1)
    b_norm = (box_b_cen**2).sum(1).view(1, -1)
   
    dist = a_norm + b_norm - 2.0 * torch.mm(box_a_cen, torch.transpose(box_b_cen, 0, 1))

    # Shape[num_a, num_b]
    return dist

def is_in_box(box_a, box_b):
    # cheack if box_a's center is in box_b
    # box_a : (xmin, ymin, xmax, ymax) Shape[num_a, 4]
    # box_b : (xmin, ymin, xmax, ymax) Shape[4]

    box_a_cen = (box_a[:, :2] + box_a[:, 2:])/2

    x_min_c = (box_a_cen[:, 0] - box_b[0]) >= 0
    y_min_c = (box_a_cen[:, 1] - box_b[1]) >= 0
    x_max_c = (box_a_cen[:, 0] - box_b[2]) <= 0
    y_max_c = (box_a_cen[:, 1] - box_b[3]) <= 0

    res = x_min_c * y_min_c * x_max_c * y_max_c
    # Shape[num_a] , 1 represents box_a in box_b
    return res

def atss_match(pos_thresh, neg_thresh, truths, priors, labels, crowd_boxes, loc_t, conf_t, idx_t, idx_t_pan, idx, loc_data, num_priors, k=9):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        pos_thresh: (float) IoU > pos_thresh ==> positive.
        neg_thresh: (float) IoU < neg_thresh ==> negative.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        crowd_boxes: (tensor) All the crowd box annotations or None if there are none.
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds. Note: -1 means neutral.
        idx_t: (tensor) Tensor to be filled w/ the index of the matched gt box for each prior.
        idx: (int) current batch index.
        loc_data: (tensor) The predicted bbox regression coordinates for this batch.
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    decoded_priors = decode(loc_data, priors, cfg.use_yolo_regressors) if cfg.use_prediction_matching else point_form(priors)
    
    # Size [num_objects, num_priors]
    overlaps = jaccard(truths, decoded_priors) if not cfg.use_change_matching else change(truths, decoded_priors)

    # Size [num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1)
    # Size [num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0)
    
    # # SIze [num_objects, num_priors]
    dist = center_distance(truths, priors)

    pos_idx = []
    neg_idx = []
    # for each gt box , find out top-k close(center) pred boxes for each layer
    for g_i, _ in enumerate(range(overlaps.size(0))):
        start_idx = 0
        C_g = []
        for n_i in num_priors: 
            # top k close pred boxes at level_i
            # topk_val , topk_idx = torch.topk(dist[g_i, start_idx:start_idx+n_i], k, largest=False)

            # use iou threshold choose top k
            topk_val , topk_idx = torch.topk(overlaps[g_i, start_idx:start_idx+n_i], k, largest=True)

            C_g.append(topk_idx+start_idx)
            start_idx = start_idx+n_i
        C_g = torch.cat((C_g), dim=-1).reshape(-1)
        D_g = overlaps[g_i, C_g]
        m_g = D_g.mean()
        v_g = D_g.std()
        t_g = m_g + v_g

        pos_idx_i = C_g[D_g >= t_g] 

        in_gt_idx = is_in_box( decoded_priors[pos_idx_i, :], truths[g_i, :])
        pos_idx_i = pos_idx_i[in_gt_idx]
        if len(pos_idx_i) == 0:
            # pos_idx_i = C_g[D_g.max(0)[1]].reshape(-1)  
            pos_idx_i = best_prior_idx[g_i].reshape(-1)
        dist[:, C_g] = 10000 # prevent pred box get considered by other gt box
        neg_idx_i = torch.tensor(list( set(C_g.tolist()) - set(pos_idx_i.tolist()))).long()
        pos_idx.append(pos_idx_i)
        neg_idx.append(neg_idx_i)

        best_truth_idx[pos_idx_i] = g_i # match the pred box to current gt box
        
        overlaps[:, pos_idx_i] = -1 # prevent pred box get considered by other gt box

        
    pos_idx = torch.cat(pos_idx, dim=0)
    neg_idx = torch.cat(neg_idx, dim=0)


    matches = truths[best_truth_idx]            # Shape: [num_priors,4]
    conf = torch.zeros_like(best_truth_idx)          # Shape: [num_priors]
    conf[:] = -1 # Default: neutral
    conf[pos_idx] = labels[best_truth_idx[pos_idx]] + 1
    conf[neg_idx] = 0

 
    # Deal with crowd annotations for COCO
    if crowd_boxes is not None and cfg.crowd_iou_threshold < 1:
        # Size [num_priors, num_crowds]
        crowd_overlaps = jaccard(decoded_priors, crowd_boxes, iscrowd=True)
        # Size [num_priors]
        best_crowd_overlap, best_crowd_idx = crowd_overlaps.max(1)
        # Set non-positives with crowd iou of over the threshold to be neutral.
        conf[(conf <= 0) & (best_crowd_overlap > cfg.crowd_iou_threshold)] = -1

    loc = encode(matches, priors, cfg.use_yolo_regressors)
    loc_t[idx]  = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf   # [num_priors] top class label for each prior
    idx_t[idx]  = best_truth_idx # [num_priors] indices for lookup
    idx_t_pan[idx]  = idx_t[idx]  #only the prior that match gt will be 1 

def match(pos_thresh, neg_thresh, truths, priors, labels, crowd_boxes, loc_t, conf_t, idx_t, idx_t_pan, idx, loc_data):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        pos_thresh: (float) IoU > pos_thresh ==> positive.
        neg_thresh: (float) IoU < neg_thresh ==> negative.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        crowd_boxes: (tensor) All the crowd box annotations or None if there are none.
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds. Note: -1 means neutral.
        idx_t: (tensor) Tensor to be filled w/ the index of the matched gt box for each prior.
        idx: (int) current batch index.
        loc_data: (tensor) The predicted bbox regression coordinates for this batch.
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    decoded_priors = decode(loc_data, priors, cfg.use_yolo_regressors) if cfg.use_prediction_matching else point_form(priors)
    
    # Size [num_objects, num_priors]
    overlaps = jaccard(truths, decoded_priors) if not cfg.use_change_matching else change(truths, decoded_priors)

    # Size [num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1)
    # Size [num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0)
    
    # For the best prior for each gt object, set its overlap to 2. This ensures
    # that it won't get thresholded out in the threshold step even if the IoU is
    # under the negative threshold. This is because we want at least one anchor
    # to match with each ground truth or else we'd be wasting training data.
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)

    # Set the index of the pair (prior, gt) we set the overlap for above.
    for j in range(best_prior_idx.size(0)):  #ensure that every gt bbox have corresponding prior
        best_truth_idx[best_prior_idx[j]] = j
   
    
    best_gt_idx = torch.zeros_like(best_truth_idx).fill_(-1)
    best_prior_idx[best_prior_overlap < pos_thresh] = -1
    gt_select = best_prior_overlap >= pos_thresh #choose the set of matched gt bboxes
    gt_idx = 0
    for i, prior in enumerate(best_prior_idx):
        if best_gt_idx[prior] == -1 and prior != -1:
            best_gt_idx[prior] = gt_idx
            gt_idx += 1
        
    idx_t_pan[idx]  = best_gt_idx  # only the prior that match gt will be 1 
    
    matches = truths[best_truth_idx]            # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1           # Shape: [num_priors]

    conf[best_truth_overlap < pos_thresh] = -1  # label as neutral
    conf[best_truth_overlap < neg_thresh] =  0  # label as background

    # Deal with crowd annotations for COCO
    if crowd_boxes is not None and cfg.crowd_iou_threshold < 1:
        # Size [num_priors, num_crowds]
        crowd_overlaps = jaccard(decoded_priors, crowd_boxes, iscrowd=True)
        # Size [num_priors]
        best_crowd_overlap, best_crowd_idx = crowd_overlaps.max(1)
        # Set non-positives with crowd iou of over the threshold to be neutral.
        conf[(conf <= 0) & (best_crowd_overlap > cfg.crowd_iou_threshold)] = -1

    loc = encode(matches, priors, cfg.use_yolo_regressors)
    loc_t[idx]  = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf   # [num_priors] top class label for each prior
    idx_t[idx]  = best_truth_idx # [num_priors] indices for lookup


@torch.jit.script
def encode(matched, priors, use_yolo_regressors:bool=False):
    """
    Encode bboxes matched with each prior into the format
    produced by the network. See decode for more details on
    this format. Note that encode(decode(x, p), p) = x.
    
    Args:
        - matched: A tensor of bboxes in point form with shape [num_priors, 4]
        - priors:  The tensor of all priors with shape [num_priors, 4]
    Return: A tensor with encoded relative coordinates in the format
            outputted by the network (see decode). Size: [num_priors, 4]
    """

    if use_yolo_regressors:
        # Exactly the reverse of what we did in decode
        # In fact encode(decode(x, p), p) should be x
        boxes = center_size(matched)

        loc = torch.cat((
            boxes[:, :2] - priors[:, :2],
            torch.log(boxes[:, 2:] / priors[:, 2:])
        ), 1)
    else:
        variances = [0.1, 0.2]

        # dist b/t match center and prior's center
        g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
        # encode variance
        g_cxcy /= (variances[0] * priors[:, 2:])
        # match wh / prior wh
        g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
        g_wh = torch.log(g_wh) / variances[1]
        # return target for smooth_l1_loss
        loc = torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]
        
    return loc

@torch.jit.script
def decode(loc, priors, use_yolo_regressors:bool=False):
    """
    Decode predicted bbox coordinates using the same scheme
    employed by Yolov2: https://arxiv.org/pdf/1612.08242.pdf

        b_x = (sigmoid(pred_x) - .5) / conv_w + prior_x
        b_y = (sigmoid(pred_y) - .5) / conv_h + prior_y
        b_w = prior_w * exp(loc_w)
        b_h = prior_h * exp(loc_h)
    
    Note that loc is inputed as [(s(x)-.5)/conv_w, (s(y)-.5)/conv_h, w, h]
    while priors are inputed as [x, y, w, h] where each coordinate
    is relative to size of the image (even sigmoid(x)). We do this
    in the network by dividing by the 'cell size', which is just
    the size of the convouts.
    
    Also note that prior_x and prior_y are center coordinates which
    is why we have to subtract .5 from sigmoid(pred_x and pred_y).
    
    Args:
        - loc:    The predicted bounding boxes of size [num_priors, 4]
        - priors: The priorbox coords with size [num_priors, 4]
    
    Returns: A tensor of decoded relative coordinates in point form 
             form with size [num_priors, 4]
    """

    if use_yolo_regressors:
        # Decoded boxes in center-size notation
        boxes = torch.cat((
            loc[:, :2] + priors[:, :2],
            priors[:, 2:] * torch.exp(loc[:, 2:])
        ), 1)

        boxes = point_form(boxes)
    else:
        variances = [0.1, 0.2]
        
        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
    
    return boxes



def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1)) + x_max


@torch.jit.script
def sanitize_coordinates(_x1, _x2, img_size:int, padding:int=0, cast:bool=True):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    if cast:
        _x1 = _x1.long()
        _x2 = _x2.long()
    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1-padding, min=0)
    x2 = torch.clamp(x2+padding, max=img_size)

    return x1, x2


@torch.jit.script
def crop(masks, boxes, padding:int=1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.size()
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)

    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)
    
    masks_left  = rows >= x1.view(1, 1, -1)
    masks_right = rows <  x2.view(1, 1, -1)
    masks_up    = cols >= y1.view(1, 1, -1)
    masks_down  = cols <  y2.view(1, 1, -1)
    
    crop_mask = masks_left * masks_right * masks_up * masks_down
    
    return masks * crop_mask.float()



def index2d(src, idx):
    """
    Indexes a tensor by a 2d index.

    In effect, this does
        out[i, j] = src[i, idx[i, j]]
    
    Both src and idx should have the same size.
    """

    offs = torch.arange(idx.size(0), device=idx.device)[:, None].expand_as(idx)
    idx  = idx + offs * idx.size(1)

    return src.view(-1)[idx.view(-1)].view(idx.size())
