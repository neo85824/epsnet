from .config import *
from .coco import COCODetection, COCOPanoptic, COCOPanoptic_inst_sem,  COCOAnnotationTransform, get_label_map
import torch
import cv2
import numpy as np

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and (lists of annotations, masks)

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list<tensor>, list<tensor>, list<int>) annotations for a given image are stacked
                on 0 dim. The output gt is a tuple of annotations and masks.
    """
    targets = []
    imgs = []
    masks_thing = []
    masks_stuff = []
    num_crowds = []

    for sample in batch:
        if sample[1][1] is None:  #if target is None , pass
            continue        
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1][0]))
        masks_thing.append(torch.FloatTensor(sample[1][1]))
        masks_stuff.append(torch.LongTensor(sample[1][2]))
        num_crowds.append(sample[1][3])
    if len(imgs) > 0: 
        return torch.stack(imgs, 0), (targets, masks_thing, masks_stuff, num_crowds)
    else:  #maybe all tensor in the batch are None
        return None
