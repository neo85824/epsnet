from data import COCODetection, COCOPanoptic, COCOPanoptic_inst_sem,  get_label_map, MEANS, COLORS

from epsnet import EPSNet
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation, semantic_logit, instance_logit, panoptic_logit
import pycocotools
import matplotlib.pyplot as plt
import PIL.Image as Image
import time



from data import cfg, set_cfg, set_dataset
import torch.nn.functional as F


import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image
from panopticapi.combine_semantic_and_instance_predictions import combine_predictions
from panopticapi.utils import IdGenerator, id2rgb

import cv2

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='EPSNet COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_instance', default=True, type=str2bool,
                        help='Whether or not to display instance')
    parser.add_argument('--display_stuff', default=True, type=str2bool,
                        help='Whether or not to display stuff segm')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--stuff_det_file', default='results/stuff_detections.json', type=str,
                        help='The output file for coco stuff results if --coco_results is set.')
    parser.add_argument('--panoptic_det_file', default='results/panoptic_detections.json', type=str,
                        help='The output file for coco panoptic results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--overlap_threshold', default=0.5, type=float,
                        help='overlap threshold for generating panoptic result')
    parser.add_argument('--stuff_area_limit', default=64*64, type=float,
                        help='stuff segm area under this limit will be ignored')
    parser.add_argument('--multiscale_test', default=False, type=str2bool,
                        help='Multi-scale testing')
    parser.add_argument('--flip_test', default=False, type=str2bool,
                        help='Horizontal flipping testing')
    parser.add_argument('--alpha', default=0.5, type=float,
    help='The opacity value of displaying the panoptic segmentation and image')

    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False, 
                       no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True
    
    if args.seed is not None:
        random.seed(args.seed)

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
coco_stuff_cats = {}
coco_stuff_cats_inv = {}
color_cache = defaultdict(lambda: {})
pan_categories = None

def postprocess_stuff(dets_out, h ,w, batch_idx=0, interpolation_mode='bilinear'):
    dets = dets_out[batch_idx]
    segment_data = dets['segm']

    proto_out = dets['proto']
    proto_h, proto_w, mask_dim = proto_out.size()


    with torch.no_grad():
        segment_data = F.interpolate(segment_data.unsqueeze(0) , (h, w),
                                                mode=interpolation_mode, align_corners=False).squeeze()
        segment_data = F.log_softmax(segment_data, dim=0) 
        # max_idx = torch.max(segment_data, dim=0)[1].unique()
        # print(max_idx)
        # for i in range(55):
        #     plt.imsave('visual_test/{}.png'.format(i), segment_data[i].detach().cpu())
        # plt.imsave('visual_test/output.png'.format(i), torch.argmax(segment_data, dim=0).detach().cpu())
        # max_value = torch.max(segment_data, dim=0)[0].repeat(segment_data.size(0), 1, 1)
        # segment_data = (segment_data-max_value).gt(-1e-8).float()
        

    
    return segment_data


def postprocess_stuff_lincomb(dets_out, h ,w, batch_idx=0, interpolation_mode='bilinear'):
    dets = dets_out[batch_idx]
    upsampled_mask = semantic_logit(dets, h, w)
    # max_value = torch.max(upsampled_mask, dim=0)[0].repeat(upsampled_mask.size(0), 1, 1)
    # upsampled_mask = (upsampled_mask-max_value).gt(-1e-8).float()

    return upsampled_mask



def prep_coco_cats():
    """ Prepare inverted table for category id lookup given a coco cats object. """
    #things catogories
    for coco_cat_id, transformed_cat_id_p1 in get_label_map().items():
        transformed_cat_id = transformed_cat_id_p1 - 1
        coco_cats[transformed_cat_id] = coco_cat_id
        coco_cats_inv[coco_cat_id] = transformed_cat_id
    
    for coco_cat_id, transformed_cat_id_p1 in get_label_map(is_stuff=True).items(): #id 0 = background 
        # transformed_cat_id = transformed_cat_id_p1 - 1
        coco_stuff_cats[transformed_cat_id_p1] = coco_cat_id
        coco_stuff_cats_inv[coco_cat_id] = transformed_cat_id
    

def get_coco_cat(transformed_cat_id, is_stuff=False):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    if is_stuff is not True:
        return coco_cats[transformed_cat_id]
    else:
        return coco_stuff_cats[transformed_cat_id]



def get_transformed_cat(coco_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    if is_stuff is not True:
        return coco_cats_inv[transformed_cat_id]
    else:
        return coco_stuff_cats_inv[transformed_cat_id]




def postprocess_ins_sem(dets, file_name, h, w):
    with timer.env('Postprocess'):
        if cfg.sem_lincomb is True:
            stuff_mask = postprocess_stuff_lincomb(dets, h, w)
        else:
            stuff_mask = postprocess_stuff(dets, h, w)

        if 'mask' not in dets[0].keys(): # handle cases of no remaining instance masks 
            masks = torch.zeros(0, h, w)
            return [[], [], None, None , stuff_mask[:-1, :, :]]
        else:
            classes, scores, boxes, masks = postprocess(dets, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)
            classes = list(classes.cpu().numpy().astype(int))
            scores = list(scores.cpu().numpy().astype(float))
            boxes = boxes.cuda()
            if masks.size(0) > 0:
                stuff_mask[0, :, :] = stuff_mask[-1, :, :] - torch.max(masks, dim=0)[0] # unknown prediction
                stuff_mask = stuff_mask[:-1, :, :]
                return [classes, scores, boxes, masks, stuff_mask]
            else:
                return [[], [], None, None , stuff_mask[:-1, :, :]]


def merge_segmentation(classes, scores, boxes, masks, stuff_mask, file_name, h, w, image_id, segm_folder, overlap_thr=0.5, stuff_area_limit=64*64):
    with timer.env('JSON Output'):
        masks = masks.view(-1, h, w).cpu().numpy().astype(np.uint8)
        stuff_mask =  torch.argmax(stuff_mask, dim=0).cpu().numpy().astype(np.uint8)

        id_generator = IdGenerator(pan_categories)
        pan_segm_id = np.zeros((h,w), dtype=np.uint32)
        used = None
        annotation = {}
        try:
            annotation['image_id'] = int(image_id)
        except Exception:
            annotation['image_id'] = image_id

        annotation['file_name'] = file_name.replace('.jpg', '.png') 
        
        segments_info = []
    with timer.env('thing process'):

        for i in range(masks.shape[0]):
            if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) <= 0:
                continue
            mask = masks[i,:,:]
            area = mask.sum()
            if area == 0:
                continue
            if used is None:
                intersect = 0
                used = mask.copy()
            else:
                intersect = (used & mask).sum()
            if intersect / area > overlap_thr:
                continue
            used = used | mask

            if intersect != 0:
                mask = (pan_segm_id == 0) & mask
            cat_id = get_coco_cat(int(classes[i]))
            segment_id = id_generator.get_id(cat_id)
            panoptic_ann = {}
            panoptic_ann['id'] = segment_id
            panoptic_ann['category_id'] = cat_id
            pan_segm_id[mask.astype(np.bool_)] = segment_id
            segments_info.append(panoptic_ann)

        pan_left = (pan_segm_id == 0) 
    with timer.env('stuff process'):

        # for c in range(1, cfg.stuff_num_classes-cfg.num_classes+1): # skip background and things

        stuff_idx = np.unique(stuff_mask) #all predicted segment with their panoptic idx

        for c in stuff_idx: # skip background and things 
            if c == 0 :
                continue
            mask = (stuff_mask==c).astype(np.bool_)
            mask_left = pan_left & mask
            if mask_left.sum()  < stuff_area_limit:
                continue
            cat_id = get_coco_cat(c, is_stuff=True)
            segment_id = id_generator.get_id(cat_id)
            panoptic_ann = {}
            panoptic_ann['id'] = segment_id
            panoptic_ann['category_id'] = cat_id
            pan_segm_id[mask_left] = segment_id
            segments_info.append(panoptic_ann)
        annotation['segments_info'] = segments_info

        Image.fromarray(id2rgb(pan_segm_id)).save(
            os.path.join(segm_folder, annotation['file_name'])
        )
        return annotation
        
    
def prep_panoptic_result(dets, img, file_name, h, w, image_id, segm_folder, overlap_thr=0.5, stuff_area_limit=64*64):
    with timer.env('Postprocess'):
        if cfg.sem_lincomb is True:
            stuff_mask = postprocess_stuff_lincomb(dets, h, w)
        else:
            stuff_mask = postprocess_stuff(dets, h, w)

        if 'mask' not in dets[0].keys(): # handle cases of no remaining instance masks 
            masks = torch.zeros(0, h, w)
        else:
            classes, scores, boxes, masks = postprocess(dets, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)
            classes = list(classes.cpu().numpy().astype(int))
            scores = list(scores.cpu().numpy().astype(float))
            boxes = boxes.cuda()
            if masks.size(0) > 0:
                stuff_mask[0, :, :] = stuff_mask[-1, :, :] - torch.max(masks, dim=0)[0] # unknown prediction
        stuff_mask = stuff_mask[:-1, :, :]

        # if classes.size(0) == 0:
        #     return
      

    with timer.env('JSON Output'):
        masks = masks.view(-1, h, w).cpu().numpy().astype(np.uint8)
        stuff_mask =  torch.argmax(stuff_mask, dim=0).cpu().numpy().astype(np.uint8)

        id_generator = IdGenerator(pan_categories)
        pan_segm_id = np.zeros((h,w), dtype=np.uint32)
        used = None
        annotation = {}
        try:
            annotation['image_id'] = int(image_id)
        except Exception:
            annotation['image_id'] = image_id

        annotation['file_name'] = file_name.replace('.jpg', '.png') 
        
        segments_info = []

    with timer.env('thing process'):
        for i in range(masks.shape[0]):
            if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) <= 0:
                continue
            mask = masks[i,:,:]
            area = mask.sum()
            if area == 0:
                continue
            if used is None:
                intersect = 0
                used = mask.copy()
            else:
                intersect = (used & mask).sum()
            if intersect / area > overlap_thr:
                continue
            used = used | mask

            if intersect != 0:
                mask = (pan_segm_id == 0) & mask
            cat_id = get_coco_cat(int(classes[i]))
            segment_id = id_generator.get_id(cat_id)
            panoptic_ann = {}
            panoptic_ann['id'] = segment_id
            panoptic_ann['category_id'] = cat_id
            pan_segm_id[mask.astype(np.bool_)] = segment_id
            segments_info.append(panoptic_ann)

        pan_left = (pan_segm_id == 0) 
    with timer.env('stuff process'):
        stuff_idx = np.unique(stuff_mask) #all predicted segment with their panoptic idx

        for c in stuff_idx: # skip background and things 
            if c == 0 :
                continue
            mask = (stuff_mask==c).astype(np.bool_)
            mask_left = pan_left & mask
            if mask_left.sum()  < stuff_area_limit:
                continue
            cat_id = get_coco_cat(c, is_stuff=True)
            segment_id = id_generator.get_id(cat_id)
            panoptic_ann = {}
            panoptic_ann['id'] = segment_id
            panoptic_ann['category_id'] = cat_id
            pan_segm_id[mask_left] = segment_id
            segments_info.append(panoptic_ann)
        annotation['segments_info'] = segments_info

        Image.fromarray(id2rgb(pan_segm_id)).save(
            os.path.join(segm_folder, annotation['file_name'])
        )
        return annotation


def prep_panoptic_display(dets, img, h, w, undo_transform=True, overlap_thr=0.5, stuff_area_limit=64*64, alpha=0.5):
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    with timer.env('Postprocess'):
        if cfg.sem_lincomb is True:
            stuff_mask = postprocess_stuff_lincomb(dets, h, w)
        else:
            stuff_mask = postprocess_stuff(dets, h, w)
        if 'mask' not in dets[0].keys(): # handle cases of no remaining instance masks 
            masks = torch.zeros(0, h, w)
        else:
            classes, scores, boxes, masks = postprocess(dets, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)
            classes = list(classes.cpu().numpy().astype(int))
            scores = list(scores.cpu().numpy().astype(float))
            boxes = boxes.cuda()
            if masks.size(0) > 0:
                stuff_mask[0, :, :] = stuff_mask[-1, :, :] - torch.max(masks, dim=0)[0] # unknown prediction
    
    stuff_mask = stuff_mask[:-1, :, :]
    masks = masks.view(-1, h, w).cpu().numpy().astype(np.uint8)
    stuff_mask =  torch.argmax(stuff_mask, dim=0).cpu().numpy().astype(np.uint8)

    id_generator = IdGenerator(pan_categories)
    pan_segm_id = np.zeros((h,w), dtype=np.uint32)
    used = None
    with timer.env('panoptic display thing'):
        for i in range(masks.shape[0]):
            if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) <= 0:
                continue
            mask = masks[i,:,:]
            area = mask.sum()
            if area == 0:
                continue
            if used is None:
                intersect = 0
                used = mask.copy()
            else:
                intersect = (used & mask).sum()
            if intersect / area > overlap_thr:
                continue
            used = used | mask
            if intersect != 0:
                mask = (pan_segm_id == 0) & mask
            cat_id = get_coco_cat(int(classes[i]))
            segment_id = id_generator.get_id(cat_id)
            pan_segm_id[mask.astype(np.bool_)] = segment_id

        pan_left = (pan_segm_id == 0) 
    with timer.env('panoptic display stuff'):
        stuff_idx = np.unique(stuff_mask) #all predicted segment with their panoptic idx

        for c in stuff_idx: # skip background and things 
            if c == 0 :
                continue
            mask = (stuff_mask==c).astype(np.bool_)
            mask_left = pan_left & mask
            if mask_left.sum()  < stuff_area_limit:
                continue
            cat_id = get_coco_cat(c, is_stuff=True)
            segment_id = id_generator.get_id(cat_id)
            pan_segm_id[mask_left] = segment_id
            # pan_left = pan_left + mask_left

    img = img.cpu().numpy().astype(np.uint8)
    img_pan_segm = img.copy()
    img_pan_segm[pan_segm_id != 0] = 0    
    img_pan_segm += cv2.cvtColor(id2rgb(pan_segm_id), cv2.COLOR_RGB2BGR)
    img_out = cv2.addWeighted(img, alpha, img_pan_segm, 1-alpha, 0)

    return img_out



def prep_unified_result(dets, img, file_name, h, w, image_id, segm_folder, overlap_thr=0.5, stuff_area_limit=64*64):

    with timer.env('panoptic logit'):
        pan_logit, classes = panoptic_logit(dets, w, h, score_threshold=args.score_threshold, overlap_thr=overlap_thr)

    with timer.env('JSON Output'):
        
        id_generator = IdGenerator(pan_categories)
        annotation = {}
        try:
            annotation['image_id'] = int(image_id)
        except Exception:
            annotation['image_id'] = image_id
        annotation['file_name'] = file_name.replace('.jpg', '.png')
        segments_info = []

        pan_result = torch.argmax(pan_logit, dim=0).cpu().numpy() # bg + n_stuff + n_thing
        n_thing = 0 if classes is None else classes.shape[0]
        n_pan = pan_logit.size(0)
        n_stuff = n_pan - n_thing
        pan_idx = np.unique(pan_result) # all predicted segment with their panoptic idx
        thing_cls_map = {idx:classes[i] for i, idx in enumerate(range(n_pan-n_thing, n_pan))}

        for idx in pan_idx:
            if idx < n_stuff and idx != 0: #stuff
                if (pan_result==idx).sum() < stuff_area_limit:
                    pan_result[pan_result==idx] = 0
                    continue
                cat_id = get_coco_cat(int(idx), is_stuff=True)
                segment_id = id_generator.get_id(cat_id)
                pan_result[pan_result==idx] = segment_id
                panoptic_ann = {}
                panoptic_ann['id'] = segment_id
                panoptic_ann['category_id'] = cat_id
                segments_info.append(panoptic_ann)
            elif idx >= n_stuff: #things
                cat_id = get_coco_cat(thing_cls_map[int(idx)])
                segment_id = id_generator.get_id(cat_id)
                pan_result[pan_result==idx] = segment_id
                panoptic_ann = {}
                panoptic_ann['id'] = segment_id
                panoptic_ann['category_id'] = cat_id
                segments_info.append(panoptic_ann)

        annotation['segments_info'] = segments_info

        Image.fromarray(id2rgb(pan_result)).save(
            os.path.join(segm_folder, annotation['file_name'])
        )
        return annotation
        

def prep_unified_display(dets, img, h, w, undo_transform=True, overlap_thr=0.5, alpha=0.5):
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    with timer.env('panoptic logit'):
        pan_logit, classes = panoptic_logit(dets, w, h, score_threshold=args.score_threshold, overlap_thr=overlap_thr)

    with timer.env('unified  postprocess'):
        id_generator = IdGenerator(pan_categories)
        pan_result = torch.argmax(pan_logit, dim=0).cpu().numpy() #bg + n_stuff + n_thing
        n_thing = 0 if classes is None else classes.shape[0]
        n_pan = pan_logit.size(0)
        n_stuff = n_pan - n_thing
        pan_idx = np.unique(pan_result) #all predicted segment with their panoptic idx
        thing_cls_map = {idx:classes[i] for i, idx in enumerate(range(n_pan-n_thing, n_pan))}
        for idx in pan_idx:
            if idx < n_stuff and idx != 0:
                cat_id = get_coco_cat(int(idx), is_stuff=True)
                segment_id = id_generator.get_id(cat_id)
                pan_result[pan_result==idx] = segment_id
            elif idx >= n_stuff:
                cat_id = get_coco_cat(thing_cls_map[int(idx)])
                segment_id = id_generator.get_id(cat_id)
                pan_result[pan_result==idx] = segment_id

        img = img.cpu().numpy().astype(np.uint8)
        img_pan_segm = img.copy()
        img_pan_segm[pan_result != 0] = 0    
        img_pan_segm += id2rgb(pan_result)
        img_out = cv2.addWeighted(img, alpha, img_pan_segm, 1-alpha, 0)

        return img_out

def badhash(x):
    """
    Just a quick and dirty hash function for doing a deterministic shuffle based on image_id.
    Source:
    https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    """
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x =  ((x >> 16) ^ x) & 0xFFFFFFFF
    return x

def evalimage(net:EPSNet, path:str, save_path:str=None):
    frame = torch.from_numpy(cv2.imread(path)).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))

    timer.reset()
    preds = net(batch)

    # img_numpy = prep_display(preds, frame, None, None, undo_transform=False)
    if cfg.use_panoptic_head:
        img_numpy = prep_unified_display(preds, frame, None, None, undo_transform=False, overlap_thr=args.overlap_threshold)
    else:
        img_numpy = prep_panoptic_display(preds, frame, None, None, undo_transform=False, overlap_thr=args.overlap_threshold, stuff_area_limit=args.stuff_area_limit, alpha=args.alpha)
    
    # timer.print_stats()

    if save_path is None:
        img_numpy = img_numpy[:, :, (2, 1, 0)]

    else:
        cv2.imwrite(save_path, img_numpy)

def evalimages(net:EPSNet, input_folder:str, output_folder:str):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    print()
    
    for p in Path(input_folder).glob('*'): 
        path = str(p)
        name = os.path.basename(path)
        name = '.'.join(name.split('.')[:-1]) + '.png'
        out_path = os.path.join(output_folder, name)

        evalimage(net, path, out_path)
        print(path + ' -> ' + out_path)
    print('Done.')

from multiprocessing.pool import ThreadPool
from queue import Queue

class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """
    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])

def evalvideo(net:EPSNet, path:str):
    # If the path is a digit, parse it as a webcam index
    is_webcam = path.isdigit()
    
    if is_webcam:
        vid = cv2.VideoCapture(int(path))
    else:
        vid = cv2.VideoCapture(path)
    
    if not vid.isOpened():
        print('Could not open video "%s"' % path)
        exit(-1)
    
    net = CustomDataParallel(net).cuda()
    transform = torch.nn.DataParallel(FastBaseTransform()).cuda()
    frame_times = MovingAverage(100)
    fps = 0
    # The 0.8 is to account for the overhead of time.sleep
    frame_time_target = 1 / vid.get(cv2.CAP_PROP_FPS)
    running = True

    def cleanup_and_exit():
        print()
        pool.terminate()
        vid.release()
        cv2.destroyAllWindows()
        exit()

    def get_next_frame(vid):
        return [vid.read()[1] for _ in range(args.video_multiframe)]

    def transform_frame(frames):
        with torch.no_grad():
            frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
            return frames, transform(torch.stack(frames, 0))

    def eval_network(inp):
        with torch.no_grad():
            frames, imgs = inp
            return frames, net(imgs)

    def prep_frame(inp):
        with torch.no_grad():
            frame, preds = inp
            return prep_panoptic_display(preds, frame, None, None, undo_transform=False, overlap_thr=args.overlap_threshold, stuff_area_limit=args.stuff_area_limit, alpha=args.alpha)
    frame_buffer = Queue()
    video_fps = 0

    # All this timing code to make sure that 
    def play_video():
        nonlocal frame_buffer, running, video_fps, is_webcam

        video_frame_times = MovingAverage(100)
        frame_time_stabilizer = frame_time_target
        last_time = None
        stabilizer_step = 0.0005

        while running:
            frame_time_start = time.time()

            if not frame_buffer.empty():
                next_time = time.time()
                if last_time is not None:
                    video_frame_times.add(next_time - last_time)
                    video_fps = 1 / video_frame_times.get_avg()
                cv2.imshow(path, frame_buffer.get())
                last_time = next_time

            if cv2.waitKey(1) == 27: # Press Escape to close
                running = False

            buffer_size = frame_buffer.qsize()
            if buffer_size < args.video_multiframe:
                frame_time_stabilizer += stabilizer_step
            elif buffer_size > args.video_multiframe:
                frame_time_stabilizer -= stabilizer_step
                if frame_time_stabilizer < 0:
                    frame_time_stabilizer = 0

            new_target = frame_time_stabilizer if is_webcam else max(frame_time_stabilizer, frame_time_target)

            next_frame_target = max(2 * new_target - video_frame_times.get_avg(), 0)
            target_time = frame_time_start + next_frame_target - 0.001 # Let's just subtract a millisecond to be safe
            # This gives more accurate timing than if sleeping the whole amount at once
            while time.time() < target_time:
                time.sleep(0.001)


    extract_frame = lambda x, i: (x[0][i] if 'mask' not in x[1][i].keys() else x[0][i].to(x[1][i]['box'].device), [x[1][i]])

    # Prime the network on the first frame because I do some thread unsafe things otherwise
    print('Initializing model... ', end='')
    eval_network(transform_frame(get_next_frame(vid)))
    print('Done.')

    # For each frame the sequence of functions it needs to go through to be processed (in reversed order)
    sequence = [prep_frame, eval_network, transform_frame]
    pool = ThreadPool(processes=len(sequence) + args.video_multiframe + 2)
    pool.apply_async(play_video)

    active_frames = []

    print()
    while vid.isOpened() and running:
        start_time = time.time()

        # Start loading the next frames from the disk
        next_frames = pool.apply_async(get_next_frame, args=(vid,))
        
        # For each frame in our active processing queue, dispatch a job
        # for that frame using the current function in the sequence
        for frame in active_frames:
            frame['value'] = pool.apply_async(sequence[frame['idx']], args=(frame['value'],))
        
        # For each frame whose job was the last in the sequence (i.e. for all final outputs)
        for frame in active_frames:
            if frame['idx'] == 0:
                frame_buffer.put(frame['value'].get())

        # Remove the finished frames from the processing queue
        active_frames = [x for x in active_frames if x['idx'] > 0]

        # Finish evaluating every frame in the processing queue and advanced their position in the sequence
        for frame in list(reversed(active_frames)):
            frame['value'] = frame['value'].get()
            frame['idx'] -= 1

            if frame['idx'] == 0:
                # Split this up into individual threads for prep_frame since it doesn't support batch size
                active_frames += [{'value': extract_frame(frame['value'], i), 'idx': 0} for i in range(1, args.video_multiframe)]
                frame['value'] = extract_frame(frame['value'], 0)

        
        # Finish loading in the next frames and add them to the processing queue
        active_frames.append({'value': next_frames.get(), 'idx': len(sequence)-1})
        
        # Compute FPS
        frame_times.add(time.time() - start_time)
        fps = args.video_multiframe / frame_times.get_avg()

        print('\rProcessing FPS: %.2f | Video Playback FPS: %.2f | Frames in Buffer: %d    ' % (fps, video_fps, frame_buffer.qsize()), end='')
    
    cleanup_and_exit()

def savevideo(net:EPSNet, in_path:str, out_path:str):

    vid = cv2.VideoCapture(in_path)

    target_fps   = round(vid.get(cv2.CAP_PROP_FPS))
    frame_width  = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames   = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height))

    transform = FastBaseTransform()
    frame_times = MovingAverage()
    progress_bar = ProgressBar(30, num_frames)

    try:
        for i in range(num_frames):
            timer.reset()
            with timer.env('Video'):
                frame = torch.from_numpy(vid.read()[1]).cuda().float()
                batch = transform(frame.unsqueeze(0))
                preds = net(batch)
                processed = prep_panoptic_display(preds, frame, None, None, undo_transform=False, overlap_thr=args.overlap_threshold, stuff_area_limit=args.stuff_area_limit, alpha=args.alpha)

                out.write(processed)
            
            if i > 1:
                frame_times.add(timer.total_time())
                fps = 1 / frame_times.get_avg()
                progress = (i+1) / num_frames * 100
                progress_bar.set_val(i+1)

                print('\rProcessing Frames  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                    % (repr(progress_bar), i+1, num_frames, progress, fps), end='')
    except KeyboardInterrupt:
        print('Stopping early.')
    
    vid.release()
    out.release()
    print()




def evaluate(net:EPSNet, dataset, train_mode=False):
    net.detect.use_fast_nms = args.fast_nms
    cfg.mask_proto_debug = args.mask_proto_debug
    global pan_categories
    with open(cfg.dataset.categories_json_file, 'r') as f:  #load panoptic catefories info (color , id....)
        cat_json = json.load(f)
    pan_categories = {el['id']: el for el in cat_json}

    if args.image is not None:
        if ':' in args.image:
            inp, out = args.image.split(':')
            evalimage(net, inp, out)
        else:
            evalimage(net, args.image)
        return
    elif args.images is not None:
        inp, out = args.images.split(':')
        evalimages(net, inp, out)
        return
    elif args.video is not None:
        if ':' in args.video:
            inp, out = args.video.split(':')
            savevideo(net, inp, out)
        else:
            evalvideo(net, args.video)
        return

    frame_times = MovingAverage()
    dataset_size = len(dataset) if args.max_images < 0 else min(args.max_images, len(dataset))
    progress_bar = ProgressBar(30, dataset_size)

    print()

    if not args.display :
        #prepare for panoptic segm annotaion (json)
        panoptic_json = [] #annotaion json
        pan_segm_folder = args.panoptic_det_file.split('.')[0] #segm files
        if not os.path.exists(pan_segm_folder):
            os.makedirs(pan_segm_folder)

    else:
        timer.disable('Load Data')

    dataset_indices = list(range(len(dataset)))
    

    if args.shuffle:
        random.shuffle(dataset_indices)
    elif not args.no_sort:
        # Do a deterministic shuffle based on the image ids
        #
        # I do this because on python 3.5 dictionary key order is *random*, while in 3.6 it's
        # the order of insertion. That means on python 3.6, the images come in the order they are in
        # in the annotations file. For some reason, the first images in the annotations file are
        # the hardest. To combat this, I use a hard-coded hash function based on the image ids
        # to shuffle the indices we use. That way, no matter what python version or how pycocotools
        # handles the data, we get the same result every time.
        hashed = [badhash(x) for x in dataset.ids]
        dataset_indices.sort(key=lambda x: hashed[x])

    dataset_indices = dataset_indices[:dataset_size]
    
    try:
        # Main eval loop
        for it, image_idx in enumerate(dataset_indices):
            timer.reset()
          
            img, gt, gt_masks, gt_stuff_masks, h, w, num_crowd = dataset.pull_item(image_idx)
            

            # Test flag, do not upvote
            if cfg.mask_proto_debug:
                with open('scripts/info.txt', 'w') as f:
                    f.write(str(dataset.ids[image_idx]))
                np.save('scripts/gt.npy', gt_masks)

            batch = Variable(img.unsqueeze(0))
            if args.cuda:
                batch = batch.cuda()

            with timer.env('Network Extra'):
                if not args.multiscale_test and not args.flip_test:
                    preds = net(batch)

            # Perform the meat of the operation here depending on our mode.
            if args.display:
                img_numpy = prep_panoptic_display(preds, frame, None, None, undo_transform=False, overlap_thr=args.overlap_threshold, stuff_area_limit=args.stuff_area_limit, alpha=args.alpha)

            else:
                if cfg.use_panoptic_head:
                    annotaion = prep_unified_result(preds, img, dataset.pull_image_name(image_idx), h, w, dataset.ids[image_idx], pan_segm_folder , overlap_thr=args.overlap_threshold, stuff_area_limit=args.stuff_area_limit)

                
                elif args.multiscale_test:
                    scales_x = [550, 605, 660, 715, 770, 825]

                    classes = []
                    scores = []
                    boxes = []
                    masks = []
                    stuff_masks = torch.zeros(cfg.stuff_num_classes-1, h, w)

                    img_np = img.numpy().transpose(1,2,0)
                    for s in scales_x:
                        img_resize = torch.from_numpy(cv2.resize(img_np, (s,s)).transpose(2,0,1) )

                        batch = Variable(img_resize.unsqueeze(0))
                        if args.cuda:
                            batch = batch.cuda()

                        preds = net(batch)

                        post_results = postprocess_ins_sem(preds, dataset.pull_image_name(image_idx), h, w)
                        classes += post_results[0]
                        scores += post_results[1]
                        if post_results[2] is not None:
                            boxes.append(post_results[2])
                            masks.append(post_results[3])
                        stuff_masks += post_results[4]
                    
                    if len(masks) == 0:
                        masks = torch.zeros(0, h, w)
                        boxes = torch.zeros(0, 4)
                    else:
                        masks = torch.cat( masks, dim=0)
                        boxes = torch.cat( boxes, dim=0)

                    stuff_masks = stuff_masks / len(scales_x)
                    annotaion = merge_segmentation(classes, scores, boxes, masks, stuff_masks, dataset.pull_image_name(image_idx), h, w, dataset.ids[image_idx], pan_segm_folder , overlap_thr=args.overlap_threshold, stuff_area_limit=args.stuff_area_limit)

                elif args.flip_test:
                    classes = []
                    scores = []
                    boxes = []
                    masks = []
                    stuff_masks = torch.zeros(cfg.stuff_num_classes-1, h, w)

                    flip = [0,  1]
                    for f in flip:
                        if f > 0:
                            img_f = img.flip(dims=[2])
                        else:
                            img_f = img

                        batch = Variable(img_f.unsqueeze(0))
                        if args.cuda:
                            batch = batch.cuda()

                        preds = net(batch)

                        post_results = postprocess_ins_sem(preds, dataset.pull_image_name(image_idx), h, w)
                        classes += post_results[0]
                        scores += post_results[1]
                        if post_results[2] is not None:
                            if f > 0:
                                boxes_f = post_results[2].clone()
                                boxes_f[:, 0] = w - post_results[2][:, 2]
                                boxes_f[:, 2] = w - post_results[2][:, 0]
                                masks_f = post_results[3].flip(dims=[2])
                                boxes.append(boxes_f)
                                masks.append(masks_f)
                                stuff_masks += post_results[4].flip(dims=[2])
                            else:
                                boxes.append(post_results[2])
                                masks.append(post_results[3])
                                stuff_masks += post_results[4]
                    
                    if len(masks) == 0:
                        masks = torch.zeros(0, h, w)
                        boxes = torch.zeros(0, 4)
                    else:
                        masks = torch.cat( masks, dim=0)
                        boxes = torch.cat( boxes, dim=0)

                    stuff_masks = stuff_masks / len(flip)
                    annotaion = merge_segmentation(classes, scores, boxes, masks, stuff_masks, dataset.pull_image_name(image_idx), h, w, dataset.ids[image_idx], pan_segm_folder , overlap_thr=args.overlap_threshold, stuff_area_limit=args.stuff_area_limit)

                else:
                    annotaion = prep_panoptic_result(preds, img, dataset.pull_image_name(image_idx), h, w, dataset.ids[image_idx], pan_segm_folder , overlap_thr=args.overlap_threshold, stuff_area_limit=args.stuff_area_limit)
          
                
                if annotaion is None or len(annotaion)==0:
                    print(annotaion)
                    print('error!!', dataset.pull_image_name(image_idx))
                    exit()
                panoptic_json.append(annotaion)
            
            # First couple of images take longer because we're constructing the graph.
            # Since that's technically initialization, don't include those in the FPS calculations.
            if it > 1:
                frame_times.add(timer.total_time())
            
            if args.display:
                if it > 1:
                    print('Avg FPS: %.4f' % (1 / frame_times.get_avg()))
                plt.imsave('results/{}.png'.format(str(dataset.ids[image_idx]) ), img_numpy)
                # plt.imshow(img_numpy)
                # plt.title(str(dataset.ids[image_idx]))
                # plt.show()
            elif not args.no_bar:
                if it > 1: fps = 1 / frame_times.get_avg()
                else: fps = 0
                progress = (it+1) / dataset_size * 100
                progress_bar.set_val(it+1)
                print('\rProcessing Images  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                    % (repr(progress_bar), it+1, dataset_size, progress, fps), end='')



        if not args.display:
            print()
            print('Dumping detections...')
            with open(cfg.dataset.valid_info, 'r') as f:
                coco_d = json.load(f)
            coco_d['annotations'] = panoptic_json
            coco_d['categories'] = list(pan_categories.values())
            with open(args.panoptic_det_file, 'w') as f:
                json.dump(coco_d, f) 



    except KeyboardInterrupt:
        print('Stopping...')


def calc_map(ap_data):
    print('Calculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    for _class in range(len(cfg.dataset.class_names)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    # Looking back at it, this code is really hard to read :/
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0 # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold*100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values())-1))
    
    print_maps(all_maps)
    return all_maps

def print_maps(all_maps):
    # Warning: hacky 
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n:  ('-------+' * n)

    print()
    print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
    print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        print(make_row([iou_type] + ['%.2f' % x for x in all_maps[iou_type].values()]))
    print(make_sep(len(all_maps['box']) + 1))
    print()


if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)

    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if args.detect:
        cfg.eval_mask_branch = False

    if args.dataset is not None:
        set_dataset(args.dataset)

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if args.cuda:
            cudnn.benchmark = True
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if args.resume and not args.display:
            with open(args.ap_data_file, 'rb') as f:
                ap_data = pickle.load(f)
            calc_map(ap_data)
            exit()

        if args.image is None and args.video is None and args.images is None:
            print(cfg.dataset.name)
            dataset = COCOPanoptic(image_path=cfg.dataset.valid_images,
                                    info_file=cfg.dataset.valid_info,
                                    pan_folder=None,
                                    transform=BaseTransform(MEANS))
        else:
            dataset = None        

        prep_coco_cats()

        print('Loading model...', end='')
        net = EPSNet()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')
        if args.cuda:
            net = net.cuda()
        
        evaluate(net, dataset)