# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import pickle as cPickle
import gzip
import PIL
import json
import logging
from datasets.vg_eval import vg_eval
from fast_rcnn.config import cfg

# Configure logging
logger = logging.getLogger(__name__)

class vg(imdb):
    def __init__(self, version, image_set, ):
        imdb.__init__(self, 'vg_' + version + '_' + image_set)
        self._version = version
        self._image_set = image_set
        self._data_path = os.path.join(cfg.DATA_DIR, 'visual_genome')
        self._img_path = os.path.join(cfg.DATA_DIR, 'visual_genome', 'images')
        # VG specific config options
        self.config = {'cleanup' : False}
        
        # Load classes
        self._classes = ['__background__']
        self._class_to_ind = {}
        self._class_to_ind[self._classes[0]] = 0
        with open(os.path.join(self._data_path, self._version, 'objects_vocab.txt'), encoding='utf-8') as f:
          count = 1
          for object in f.readlines():
            names = [n.lower().strip() for n in object.split(',')]
            self._classes.append(names[0])
            for n in names:
              self._class_to_ind[n] = count
            count += 1 
            
        # Load attributes
        self._attributes = ['__no_attribute__']
        self._attribute_to_ind = {}
        self._attribute_to_ind[self._attributes[0]] = 0
        with open(os.path.join(self._data_path, self._version, 'attributes_vocab.txt'), encoding='utf-8') as f:
          count = 1
          for att in f.readlines():
            names = [n.lower().strip() for n in att.split(',')]
            self._attributes.append(names[0])
            for n in names:
              self._attribute_to_ind[n] = count
            count += 1           
            
        # Load relations
        self._relations = ['__no_relation__']
        self._relation_to_ind = {}
        self._relation_to_ind[self._relations[0]] = 0
        with open(os.path.join(self._data_path, self._version, 'relations_vocab.txt'), encoding='utf-8') as f:
          count = 1
          for rel in f.readlines():
            names = [n.lower().strip() for n in rel.split(',')]
            self._relations.append(names[0])
            for n in names:
              self._relation_to_ind[n] = count
            count += 1      
            
        self._image_ext = '.jpg'
        self._image_index, self._id_to_dir = self._load_image_set_index()

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        folder = self._id_to_dir[index]
        image_path = os.path.join(self._img_path, folder,
                                  str(index) + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path
        
    def _image_split_path(self):
        if self._image_set == "minitrain":
          return os.path.join(self._data_path, "split", "train.txt")
        if self._image_set == "minival":
          return os.path.join(self._data_path, "split", "val.txt")
        else:
          return os.path.join(self._data_path, "split", self._image_set+'.txt')

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        training_split_file = self._image_split_path()
        assert os.path.exists(training_split_file), \
                'Path does not exist: {}'.format(training_split_file)
        with open(training_split_file, encoding='utf-8') as f:
          metadata = f.readlines()
          if self._image_set == "minitrain":
            metadata = metadata[:1000]
          elif self._image_set == "minival":
            metadata = metadata[:100]
          
        image_index = []
        id_to_dir = {}
        for line in metadata:
          im_file,ann_file = line.split()
          image_id = int(ann_file.split('/')[-1].split('.')[0])
          filename = self._annotation_path(image_id)
          if os.path.exists(filename):
              # Some images have no bboxes after object filtering, so there
              # is no xml annotation for these.
              tree = ET.parse(filename)
              for obj in tree.findall('object'):
                  obj_name = obj.find('name').text.lower().strip()
                  if obj_name in self._class_to_ind:
                      # We have to actually load and check these to make sure they have
                      # at least one object actually in vocab
                      image_index.append(image_id)
                      id_to_dir[image_id] = im_file.split('/')[0]
                      break
        return image_index, id_to_dir
      
    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            fid = gzip.open(cache_file,'rb') 
            roidb = cPickle.load(fid)
            fid.close()
            logger.info('%s gt roidb loaded from %s', self.name, cache_file)
            return roidb

        gt_roidb = [self._load_vg_annotation(index)
                    for index in self.image_index]

        fid = gzip.open(cache_file,'wb')       
        cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        fid.close()
        logger.info('wrote gt roidb to %s', cache_file)
        return gt_roidb
      
    def _get_size(self, index):
      return PIL.Image.open(self.image_path_from_index(index)).size
      
    def _annotation_path(self, index):
        return os.path.join(self._data_path, 'xml', str(index) + '.xml')
      
    def _load_vg_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        width, height = self._get_size(index)
        filename = self._annotation_path(index)
        tree = ET.parse(filename)
        objs = tree.findall('object')
        
        # First, count how many valid objects we have
        valid_objs = []
        for obj in objs:
            obj_name = obj.find('name').text.lower().strip()
            if obj_name in self._class_to_ind:
                valid_objs.append(obj)
        
        num_valid_objs = len(valid_objs)
        
        # Only allocate arrays for valid objects
        boxes = np.zeros((num_valid_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_valid_objs), dtype=np.int32)
        gt_attributes = np.zeros((num_valid_objs, 16), dtype=np.int32)
        overlaps = np.zeros((num_valid_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_valid_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        obj_dict = {}
        ix = 0
        
        # Only process valid objects
        for obj in valid_objs:
            obj_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            x1 = max(0,float(bbox.find('xmin').text))
            y1 = max(0,float(bbox.find('ymin').text))
            x2 = min(width-1,float(bbox.find('xmax').text))
            y2 = min(height-1,float(bbox.find('ymax').text))
            # If bboxes are not positive, just give whole image coords (there are a few examples)
            if x2 < x1 or y2 < y1:
                logger.warning('Failed bbox in %s, object %s', filename, obj_name)
                x1 = 0
                y1 = 0
                x2 = width-1
                y2 = width-1
            class_label = self._class_to_ind[obj_name]
            obj_dict[obj.find('object_id').text] = ix
            atts = obj.findall('attribute')
            n = 0
            for att in atts:
                att = att.text.lower().strip()
                if att in self._attribute_to_ind:
                    gt_attributes[ix, n] = self._attribute_to_ind[att]
                    n += 1
                if n >= 16:
                    break
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = class_label
            overlaps[ix, class_label] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
            ix += 1

        overlaps = scipy.sparse.csr_matrix(overlaps)
        gt_attributes = scipy.sparse.csr_matrix(gt_attributes)

        # Process relations - modified to handle only valid objects
        rels = tree.findall('relation')
        num_rels = len(rels) 
        gt_relations = set() # Avoid duplicates
        for rel in rels:
            pred = rel.find('predicate').text
            if pred: # One is empty
                pred = pred.lower().strip()
                if pred in self._relation_to_ind:
                    try:
                        triple = []
                        triple.append(obj_dict[rel.find('subject_id').text])
                        triple.append(self._relation_to_ind[pred])
                        triple.append(obj_dict[rel.find('object_id').text])
                        gt_relations.add(tuple(triple))
                    except:
                        pass # Object not in dictionary
        gt_relations = np.array(list(gt_relations), dtype=np.int32)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_attributes' : gt_attributes,
                'gt_relations' : gt_relations,
                'gt_overlaps' : overlaps,
                'width' : width,
                'height': height,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(self.classes, all_boxes, output_dir)
        mean_ap = self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_vg_results_file_template(output_dir).format(cls)
                os.remove(filename)
        return mean_ap

    def evaluate_attributes(self, all_boxes, output_dir):
        self._write_voc_results_file(self.attributes, all_boxes, output_dir)
        self._do_python_eval(output_dir, eval_attributes = True)
        if self.config['cleanup']:
            for cls in self._attributes:
                if cls == '__no_attribute__':
                    continue
                filename = self._get_vg_results_file_template(output_dir).format(cls)
                os.remove(filename)
                
    def _get_vg_results_file_template(self, output_dir):
        filename = 'detections_' + self._image_set + '_{:s}.txt'
        path = os.path.join(output_dir, filename)
        return path

    def _write_voc_results_file(self, classes, all_boxes, output_dir):
        for cls_ind, cls in enumerate(classes):
            if cls == '__background__':
                continue
            print('Writing "{}" vg results file'.format(cls))
            filename = self._get_vg_results_file_template(output_dir).format(cls)
            with open(filename, 'wt', encoding='utf-8') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets is None or dets.size == 0:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(str(index), dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
                                                
                                       
    def _do_python_eval(self, output_dir, pickle=True, eval_attributes = False):
        # We re-use parts of the pascal voc python code for visual genome
        aps = []
        nposs = []
        thresh = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        # Load ground truth    
        gt_roidb = self.gt_roidb()
        if eval_attributes:
            classes = self._attributes
        else:
            classes = self._classes
        for i, cls in enumerate(classes):
            if cls == '__background__' or cls == '__no_attribute__':
                continue
            filename = self._get_vg_results_file_template(output_dir).format(cls)
            rec, prec, ap, scores, npos = vg_eval(
                filename, gt_roidb, self.image_index, i, ovthresh=0.5,
                use_07_metric=use_07_metric, eval_attributes=eval_attributes)

            # Determine per class detection thresholds that maximise f score
            if npos > 1:
                f = np.nan_to_num((prec*rec)/(prec+rec))
                thresh += [scores[np.argmax(f)]]
            else: 
                thresh += [0]
            # Handle None AP values (no detections or no ground truth)
            ap = 0.0 if ap is None else ap
            aps += [ap]
            nposs += [float(npos)]
            print('AP for {} = {:.4f} (npos={:,})'.format(cls, ap, npos))
            if pickle:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap, 
                        'scores': scores, 'npos':npos}, f)
         
        # Set thresh to mean for classes with poor results 
        thresh = np.array(thresh)
        avg_thresh = np.mean(thresh[thresh!=0])
        thresh[thresh==0] = avg_thresh
        if eval_attributes:
            filename = 'attribute_thresholds_' + self._image_set + '.txt'
        else:
            filename = 'object_thresholds_' + self._image_set + '.txt'
        path = os.path.join(output_dir, filename)       
        with open(path, 'wt', encoding='utf-8') as f:
            for i, cls in enumerate(classes[1:]):
                f.write('{:s} {:.3f}\n'.format(cls, thresh[i]))           
                
        # Calculate mean AP
        aps = np.array(aps)
        nposs = np.array(nposs)
        weights = nposs / nposs.sum()
        mean_ap = np.mean(aps)
        weighted_mean_ap = np.average(aps, weights=weights)
        print('Mean AP = {:.4f}'.format(mean_ap))
        print('Weighted Mean AP = {:.4f}'.format(weighted_mean_ap))
        print('Mean Detection Threshold = {:.3f}'.format(avg_thresh))
        print('~~~~~~~~')
        print('Results:')
        for ap,npos in zip(aps,nposs):
            print('{:.3f}\t{:.3f}'.format(ap,npos))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** PASCAL VOC Python eval code.')
        print('--------------------------------------------------------------')           
        
        return mean_ap  # Return the mean AP value


if __name__ == '__main__':
    d = datasets.vg('val')
    res = d.roidb
    from IPython import embed; embed()
