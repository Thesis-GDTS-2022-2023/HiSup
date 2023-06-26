import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
del sys, os

import os
import os.path as osp
import json
import torch
import logging
import numpy as np
import scipy
import scipy.ndimage

from PIL import Image
from tqdm import tqdm
from skimage import io
from tools.evaluation import coco_eval, boundary_eval, polis_eval
from hisup.utils.comm import to_device
from hisup.utils.polygon import generate_polygon
from hisup.utils.visualizer import viz_inria
from hisup.dataset.build import build_test_dataset
from hisup.dataset.build_ import build_transform
from hisup.utils.polygon import juncs_in_bbox

from shapely.geometry import Polygon
from skimage.measure import label, regionprops

from pycocotools import mask as coco_mask



def poly_to_bbox(poly):
    """
    input: poly----2D array with points
    """
    lt_x = np.min(poly[:,0])
    lt_y = np.min(poly[:,1])
    w = np.max(poly[:,0]) - lt_x
    h = np.max(poly[:,1]) - lt_y
    return [float(lt_x), float(lt_y), float(w), float(h)]

def generate_coco_ann(polys, scores, img_id):
    sample_ann = []
    for i, polygon in enumerate(polys[0]):
        if polygon.shape[0] < 3:
            continue

        vec_poly = polygon.ravel().tolist()
        poly_bbox = poly_to_bbox(polygon)
        ann_per_building = {
                'image_id': img_id,
                'category_id': 0,
                'segmentation': [vec_poly],
                'bbox': poly_bbox,
                'score': float(scores[0][i]),
            }
        sample_ann.append(ann_per_building)

    return sample_ann

def generate_coco_mask(mask, img_id):
    sample_ann = []
    props = regionprops(label(mask[0] > 0.50))
    for prop in props:
        if ((prop.bbox[2] - prop.bbox[0]) > 0) & ((prop.bbox[3] - prop.bbox[1]) > 0):
            prop_mask = np.zeros_like(mask[0], dtype=np.uint8)
            prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1

            masked_instance = np.ma.masked_array(mask[0], mask=(prop_mask != 1))
            score = masked_instance.mean()
            encoded_region = coco_mask.encode(np.asfortranarray(prop_mask))
            ann_per_building = {
                'image_id': img_id,
                'category_id': 0,
                'segmentation': {
                    "size": encoded_region["size"],
                    "counts": encoded_region["counts"].decode()
                },
                'score': float(score),
            }
            sample_ann.append(ann_per_building)

    return sample_ann


class TestPipeline():
    def __init__(self, cfg, eval_type='coco_iou'):
        self.cfg = cfg
        self.device = cfg.MODEL.DEVICE
        self.output_dir = cfg.OUTPUT_DIR
        self.dataset_name = cfg.DATASETS.TEST[0]
        self.eval_type = eval_type
        
    

    def eval(self,gt_file,dt_file):
        logger = logging.getLogger("testing")
        logger.info('Evalutating on {}'.format(self.eval_type))
        if self.eval_type == 'coco_iou':
            coco_eval(gt_file, dt_file)
        elif self.eval_type == 'boundary_iou':
            boundary_eval(gt_file, dt_file)
        elif self.eval_type == 'polis':
            polis_eval(gt_file, dt_file)

    def test_batch(self,imgs,output,anns,res,mask_res):
        batch_size = imgs.size(0)
        batch_scores = output['scores']
        batch_polygons = output['polys_pred']
        batch_masks = output['mask_pred']

        for b in range(batch_size):
            img_id = anns[b]['id']

            scores = batch_scores[b]
            polys = batch_polygons[b]
            mask_pred = batch_masks[b]
            # print(mask_pred.shape)

            image_result = generate_coco_ann(polys, scores, img_id)
            if len(image_result) != 0:
                res.extend(image_result)

            image_masks = generate_coco_mask(mask_pred, img_id)
            if len(image_masks) != 0:
                mask_res.extend(image_masks)

    def test_log(self,res,logger,gt_file,type='non-mask',test_set='target'):
        dt_file = osp.join(self.output_dir,'{}_{}_mask.json'.format(test_set,self.dataset_name)) if type == 'mask' else\
                  osp.join(self.output_dir,'{}_{}.json'.format(test_set,self.dataset_name)) 
        
        logger.info('Writing the results of the {} dataset into {}'.format(self.dataset_name,
                    dt_file))
        with open(dt_file,'w') as _out:
            json.dump(res,_out)

        self.gt_file = gt_file
        self.dt_file = dt_file
        self.eval(gt_file,dt_file)
    
    def test(self, model):
        logger = logging.getLogger("testing")
        logger.info('Testing on {} dataset'.format(self.dataset_name))

        tar_results = []
        tar_mask_results = []
        aux_results = []
        aux_mask_results = []
        test_dataset, tar_gt_file, aux_ft_file = build_test_dataset(self.cfg)
        for i, (images, annotations) in enumerate(tqdm(test_dataset)):
            imgs_tar, imgs_aux = images
            anns_tar, anns_aux = [ann[0] for ann in annotations], [ann[1] for ann in annotations]
            
            with torch.no_grad():
                output, _ = model(imgs_tar.to(self.device), imgs_aux.to(self.device),to_device(anns_tar, self.device),to_device(anns_aux, self.device))
                output = to_device(output,'cpu')

    
            self.test_batch(imgs_tar,output,anns_tar,tar_results,tar_mask_results)
            self.test_batch(imgs_aux,output,anns_aux,aux_results,aux_mask_results)


        self.test_log(tar_results,logger,tar_gt_file)

        self.test_log(tar_mask_results,tar_gt_file,type='mask')

        self.test_log(aux_results,aux_ft_file,test_set='auxiliary')

        self.test_log(aux_mask_results,aux_ft_file,type='mask',test_set='auxiliary')

