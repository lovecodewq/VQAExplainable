import logging
import cv2
from fast_rcnn.config import cfg  # Import Fast R-CNN config
from torchvision.models.detection.image_list import ImageList
from datasets.utils.blob import im_list_to_blob, prep_im_for_blob
import numpy as np
import torch

# Create a picklable collate function class
class VGCollator:
    def __init__(self, dataset, device='cpu'):
        self.dataset = dataset
        self.device = device
        
    def __call__(self, batch):
        """
        Convert VG format to the format required by BottomUpAttention
        """
        logger = logging.getLogger(__name__)
        logger.debug(f"Collate_fn processing batch of size: {len(batch)}")
        
        processed_ims = []
        im_scales = []
        gt_boxes_list = []
        image_sizes = []
        
        # Process each image in the batch
        for roidb_idx in batch:
            # Get image
            img_path = self.dataset.image_path_at(roidb_idx)
            im = cv2.imread(img_path)
            
            if im is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
            # Prepare image using original Faster R-CNN preprocessing
            target_size = cfg.TRAIN.SCALES[0]
            im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
            # Record resized size (the actual tensor we'll feed RPN)
            new_h, new_w = im.shape[:2]
            image_sizes.append((new_h, new_w))

            processed_ims.append(im)
            im_scales.append(im_scale)
            
            # Get ground truth data
            roidb = self.dataset.gt_roidb()[roidb_idx]
            boxes = roidb['boxes']  # [N, 4]
            classes = roidb['gt_classes']  # [N]
            
            # Extract attribute info
            attr_matrix = roidb['gt_attributes'].toarray()  # Convert sparse to dense
            # Get the highest confidence attribute for each object (or 0 for no attribute)
            attrs = np.zeros(len(classes))
            for i in range(len(classes)):
                nonzero_attrs = attr_matrix[i].nonzero()[0]
                if len(nonzero_attrs) > 0:
                    attrs[i] = nonzero_attrs[0]  # Take the first attribute
            
            # Build [x1, y1, x2, y2, obj_label, attr_label] format
            N = len(boxes)
            gt = torch.zeros((N, 6), dtype=torch.float32)
            for i in range(N):
                # Scale the bounding boxes according to the image resize
                x1, y1, x2, y2 = boxes[i]
                x1 = x1 * im_scale
                x2 = x2 * im_scale
                y1 = y1 * im_scale
                y2 = y2 * im_scale
                gt[i, 0:4] = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
                gt[i, 4] = float(classes[i])
                gt[i, 5] = float(attrs[i])
            
            gt_boxes_list.append(gt)
        
        # Convert image list to network input blob
        blob = im_list_to_blob(processed_ims)
        imgs_tensor = torch.from_numpy(blob).to(self.device)
        
        # Create ImageList for RPN and ROI pooling
        imgs = ImageList(imgs_tensor, image_sizes)
        
        # Create list of ground truth targets in format expected by RPN
        gt_targets = []
        for gt in gt_boxes_list:
            gt_targets.append({
                'boxes': gt[:, :4].to(self.device),  # (x1, y1, x2, y2)
                'labels': gt[:, 4].long().to(self.device),  # object class ids
                'attributes': gt[:, 5].long().to(self.device)  # attribute ids
            })
        
        # Create image info [height, width, scale]
        _, _, H, W = imgs_tensor.shape
        im_info = torch.tensor([[H, W, s] for s in im_scales], device=self.device)
        
        return imgs, im_info, gt_targets