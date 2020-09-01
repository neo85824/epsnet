import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
from .config import cfg
import json
import PIL.Image as Image
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
from panopticapi.utils import  rgb2id


def get_label_map(is_stuff=False):
    label_map = cfg.dataset.label_map if is_stuff is False else cfg.dataset.stuff_label_map
    class_names = cfg.dataset.class_names if is_stuff is False else cfg.dataset.stuff_class_names
    if label_map is None:
        return {x+1: x+1 for x in range(len(class_names))}
    else:
        return label_map 



class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self, is_stuff=False):
        self.label_map = get_label_map()

    def __call__(self, target, width, height, is_stuff=False):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        self.label_map = get_label_map(is_stuff=is_stuff)
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                label_idx = self.label_map[obj['category_id']] - 1 if is_stuff is not True else self.label_map[obj['category_id']]  #**Cause Stuff id=0 denotes background 
                final_box = list(np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("No bbox found for object ", obj)

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class COCODetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, image_path, info_file, transform=None,
                 target_transform=COCOAnnotationTransform(),
                 dataset_name='MS COCO', has_gt=True):
        # Do this here because we have too many things named COCO
        from pycocotools.coco import COCO
        
        self.root = image_path
        self.coco = COCO(info_file)
        
        self.ids = list(self.coco.imgToAnns.keys())
        if len(self.ids) == 0 or not has_gt:
            self.ids = list(self.coco.imgs.keys())
        
        self.transform = transform
        self.target_transform = target_transform
        
        self.name = dataset_name
        self.has_gt = has_gt

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, (target, masks, num_crowds)).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, masks, h, w, num_crowds = self.pull_item(index)
        return im, (gt, masks, num_crowds)

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width, crowd).
                   target is the object returned by ``coco.loadAnns``.
            Note that if no crowd annotations exist, crowd will be None
        """
        img_id = self.ids[index]

        if self.has_gt:
            target = self.coco.imgToAnns[img_id]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)

            # Target has {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
            target = self.coco.loadAnns(ann_ids)
        else:
            target = []

        # Separate out crowd annotations. These are annotations that signify a large crowd of
        # objects of said class, where there is no annotation for each individual object. Both
        # during testing and training, consider these crowds as neutral.
        crowd  = [x for x in target if     ('iscrowd' in x and x['iscrowd'])]
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        num_crowds = len(crowd)

        # This is so we ensure that all crowd annotations are at the end of the array
        target += crowd
        
        # The split here is to have compatibility with both COCO2014 and 2017 annotations.
        # In 2014, images have the pattern COCO_{train/val}2014_%012d.jpg, while in 2017 it's %012d.jpg.
        # Our script downloads the images as %012d.jpg so convert accordingly.
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        
        if file_name.startswith('COCO'):
            file_name = file_name.split('_')[-1]

        path = osp.join(self.root, file_name)
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        
        img = cv2.imread(path)
        height, width, _ = img.shape
        
        if len(target) > 0:
            # Pool all the masks for this image into one [num_objects,height,width] matrix
            masks = [self.coco.annToMask(obj).reshape(-1) for obj in target]
            masks = np.vstack(masks)
            masks = masks.reshape(-1, height, width)
        if self.target_transform is not None and len(target) > 0:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            if len(target) > 0:
                target = np.array(target)
                img, masks, boxes, labels = self.transform(img, masks, target[:, :4],
                    {'num_crowds': num_crowds, 'labels': target[:, 4]})

                # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations
                num_crowds = labels['num_crowds']
                labels     = labels['labels']
                
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))


            else:
                img, _, _, _ = self.transform(img, np.zeros((1, height, width), dtype=np.float), np.array([[0, 0, 1, 1]]),
                    {'num_crowds': 0, 'labels': np.array([0])})
                masks = None
                target = None

        return torch.from_numpy(img).permute(2, 0, 1), target, masks, height, width, num_crowds

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class COCOPanoptic(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, image_path, info_file , pan_folder , transform=None,
                 target_transform=COCOAnnotationTransform(),
                 dataset_name='MS COCO', has_gt=True):
        # Do this here because we have too many things named COCO
        with open(info_file, 'r') as f:  
            self.info_json = json.load(f)

        if has_gt:
            if pan_folder is None:
                pan_folder = info_file.replace('.json', '')
            self.gt_folder = pan_folder
            self.gt_annotations = {el['image_id']: el for el in self.info_json['annotations']} # {image_id : annotation}
        
        self.pan_categories = {el['id']: el for el in self.info_json['categories']}  # { cat_id : cat_info }
        self.ids = [el['id'] for el in self.info_json['images']]
        self.file_names = {el['id']:el['file_name'] for el in self.info_json['images']}
        self.root = image_path
        
        self.transform = transform
        self.target_transform = target_transform
        
        self.name = dataset_name
        self.has_gt = has_gt

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, (target, masks, num_crowds)).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, masks_thing, mask_stuff, h, w, num_crowds = self.pull_item(index)
        return im, (gt, masks_thing, mask_stuff, num_crowds)

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width, crowd).
                   target is the object returned by ``coco.loadAnns``.
            Note that if no crowd annotations exist, crowd will be None
        """
        img_id = self.ids[index]

        if self.has_gt:
            target_thing = []
            target_stuff = []
            annotation = self.gt_annotations[img_id]  # {segment_info , file_name, image_id}
            pan_gt = np.array(Image.open(os.path.join(self.gt_folder, self.file_names[img_id].replace('.jpg','.png'))), dtype=np.uint32)
            pan_gt = rgb2id(pan_gt)
            seg_infos = {el['id']:el  for el in annotation['segments_info']} #get all segments' info
            seg_ids = list(seg_infos.keys())  
            seg_masks = {seg_id:(pan_gt==seg_id).astype(np.float16) for seg_id in seg_ids}
            for seg_id in seg_ids:
                cat_id = seg_infos[seg_id]['category_id']
                target_dict = seg_infos[seg_id]  # { seg_id , cat_id, is_crowd, bbox, area, mask}
                if self.pan_categories[cat_id]['isthing'] == 1:
                    target_thing.append(target_dict)
                else:
                    target_stuff.append(target_dict)
        else:
            target_thing = []
            target_stuff = []
               

        # Separate out crowd annotations. These are annotations that signify a large crowd of
        # objects of said class, where there is no annotation for each individual object. Both
        # during testing and training, consider these crowds as neutral.
        crowd  = [x for x in target_thing if     ('iscrowd' in x and x['iscrowd'])]
        target_thing = [x for x in target_thing if not ('iscrowd' in x and x['iscrowd'])]
        num_crowds = len(crowd)

        # This is so we ensure that all crowd annotations are at the end of the array
        target_thing += crowd
        
        n_thing = len(target_thing)
        n_stuff = len(target_stuff)

        # The split here is to have compatibility with both COCO2014 and 2017 annotations.
        # In 2014, images have the pattern COCO_{train/val}2014_%012d.jpg, while in 2017 it's %012d.jpg.
        # Our script downloads the images as %012d.jpg so convert accordingly.
        file_name = self.file_names[img_id]
        
        if file_name.startswith('COCO'):
            file_name = file_name.split('_')[-1]

        path = osp.join(self.root, file_name)

        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        
        img = cv2.imread(path)
        height, width, _ = img.shape

        if n_thing > 0 and self.has_gt:
            masks = np.array([seg_masks[t['id']] for t in target_thing]  + [seg_masks[t['id']] for t in target_stuff] )
        else:
            masks = None

        #mix the target from thing and stuff
        if self.target_transform is not None:
            if len(target_thing) > 0:
                target_thing = self.target_transform(target_thing, width, height)
            if len(target_stuff) > 0:
                target_stuff = self.target_transform(target_stuff, width, height, is_stuff=True)

        target = target_thing + target_stuff
        if self.transform is not None:
            if len(target_thing) > 0:
                target = np.array(target)
                thing_checks = np.ones((target.shape[0], 1))  #record each mask which belongs to thing or mask , if thing check=1 else 0
                thing_checks[n_thing:, :] = 0 
                cls_labels = np.concatenate((thing_checks, target[:,4].reshape(-1,1)), axis=1)

                img, masks, boxes, labels = self.transform(img, masks.astype(np.uint8), target[:, :4],
                    {'num_crowds': num_crowds, 'labels': cls_labels })

                # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations
                num_crowds = labels['num_crowds']
                labels     = labels['labels']

                #only keep thing in target(label and bboxes)
                thing_idx =  np.where(labels[:,0]==1)[0]
                label_thing = labels[thing_idx, 1]
                stuff_idx =  np.where(labels[:,0]==0)[0]
                label_stuff = labels[stuff_idx, 1]
                boxes = boxes[thing_idx, :]
                target = np.hstack((boxes, np.expand_dims(label_thing, axis=1)))      
        
            else:
                img, _, _, _ = self.transform(img, np.zeros((1, height, width), dtype=np.float), np.array([[0, 0, 1, 1]]),
                    {'num_crowds': 0, 'labels': np.array([0])})
                masks = None
                target = None

        # build semantic segmentation format output

        if masks is not None and len(thing_idx)>0: #mask sure that after cropping there are still things mask
            masks_thing = masks[thing_idx, :, :]
            mask_stuff = masks[stuff_idx, :, :]
            stuff_logits = np.zeros((mask_stuff.shape[1], mask_stuff.shape[2]))
            things_idx = cfg.stuff_num_classes - 1
            for i in range(len(stuff_idx)):
                stuff_logits[mask_stuff[i,:,:] != 0] = label_stuff[i]
            stuff_logits[np.sum(masks_thing, axis=0) > 0] = things_idx

        else:
            masks_thing = None
            stuff_logits = None
    
        return torch.from_numpy(img).permute(2, 0, 1), target, masks_thing, stuff_logits, height, width, num_crowds

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.file_names[img_id]

        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_image_name(self, index):
        img_id = self.ids[index]
        return self.file_names[img_id]
        
    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        return self.gt_annotations[img_id]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class COCOPanoptic_inst_sem(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, image_path, info_file ,stuff_info_file, transform=None,
                 target_transform=COCOAnnotationTransform(),
                 dataset_name='MS COCO', has_gt=True):
        # Do this here because we have too many things named COCO
        from pycocotools.coco import COCO
        
        self.root = image_path
        self.coco = COCO(info_file)
        self.coco_stuff = COCO(stuff_info_file)
        
        self.ids = list(self.coco.imgToAnns.keys())
        if len(self.ids) == 0 or not has_gt:
            self.ids = list(self.coco.imgs.keys())
        
        self.transform = transform
        self.target_transform = target_transform
        
        self.name = dataset_name
        self.has_gt = has_gt

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, (target, masks, num_crowds)).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, masks_thing, mask_stuff, h, w, num_crowds = self.pull_item(index)
        return im, (gt, masks_thing, mask_stuff, num_crowds)

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width, crowd).
                   target is the object returned by ``coco.loadAnns``.
            Note that if no crowd annotations exist, crowd will be None
        """
        img_id = self.ids[index]

        if self.has_gt:
            target = self.coco.imgToAnns[img_id]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            ann_ids_stuff = self.coco_stuff.getAnnIds(imgIds=img_id)

            # Target has {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
            target = self.coco.loadAnns(ann_ids)
            target_stuff = self.coco_stuff.loadAnns(ann_ids_stuff)
        else:
            target = []

        # Separate out crowd annotations. These are annotations that signify a large crowd of
        # objects of said class, where there is no annotation for each individual object. Both
        # during testing and training, consider these crowds as neutral.
        crowd  = [x for x in target if     ('iscrowd' in x and x['iscrowd'])]
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        num_crowds = len(crowd)

        # This is so we ensure that all crowd annotations are at the end of the array
        target += crowd
        
        n_thing = len(target)
        n_stuff = len(target_stuff)

        # The split here is to have compatibility with both COCO2014 and 2017 annotations.
        # In 2014, images have the pattern COCO_{train/val}2014_%012d.jpg, while in 2017 it's %012d.jpg.
        # Our script downloads the images as %012d.jpg so convert accordingly.
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        
        if file_name.startswith('COCO'):
            file_name = file_name.split('_')[-1]

        path = osp.join(self.root, file_name)
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        
        img = cv2.imread(path)
        height, width, _ = img.shape
        
        if len(target) > 0:
            # Pool all the masks for this image into one [num_objects,height,width] matrix
            masks_thing = [self.coco.annToMask(obj).reshape(-1) for obj in target]
            masks_stuff = [self.coco_stuff.annToMask(obj).reshape(-1) for obj in target_stuff]
            masks = masks_thing + masks_stuff
            masks = np.vstack(masks)
            masks = masks.reshape(-1, height, width)

        #mix the target from thing and stuff
        if self.target_transform is not None:
            if len(target) > 0:
                target = self.target_transform(target, width, height)
            if len(target_stuff) > 0:
                target_stuff = self.target_transform(target_stuff, width, height, is_stuff=True)

        target = target + target_stuff

        if self.transform is not None:
            if len(target) > 0:
                target = np.array(target)
                thing_checks = np.ones((target.shape[0], 1))  #record each mask which belongs to thing or mask , if thing check=1 else 0
                thing_checks[n_thing:, :] = 0 
                cls_labels = np.concatenate((thing_checks, target[:,4].reshape(-1,1)), axis=1)

                img, masks, boxes, labels = self.transform(img, masks, target[:, :4],
                    {'num_crowds': num_crowds, 'labels': cls_labels })

                # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations
                num_crowds = labels['num_crowds']
                labels     = labels['labels']

                #only keep thing in target(label and bboxes)
                thing_idx =  np.where(labels[:,0]==1)[0]
                label_thing = labels[thing_idx, 1]
                stuff_idx =  np.where(labels[:,0]==0)[0]
                label_stuff = labels[stuff_idx, 1]
                boxes = boxes[thing_idx, :]
                target = np.hstack((boxes, np.expand_dims(label_thing, axis=1)))              

            else:
                img, _, _, _ = self.transform(img, np.zeros((1, height, width), dtype=np.float), np.array([[0, 0, 1, 1]]),
                    {'num_crowds': 0, 'labels': np.array([0])})
                masks = None
                target = None

        #build semantic segmentation format output
        masks_thing = masks[thing_idx, :, :]
        mask_stuff = masks[stuff_idx, :, :]
        stuff_logits = np.zeros((mask_stuff.shape[1], mask_stuff.shape[2]))
        for i in range(len(stuff_idx)):
            stuff_logits[mask_stuff[i,:,:] != 0] = label_stuff[i]

        return torch.from_numpy(img).permute(2, 0, 1), target, masks_thing, stuff_logits, height, width, num_crowds

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_image_name(self, index):
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return path
        
    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
