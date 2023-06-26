import cv2
import numpy as np
import os.path as osp
import torchvision.datasets as dset
from torch.utils.data import Dataset

from PIL import Image
from pycocotools.coco import COCO
from shapely.geometry import Polygon
from torch.utils.data.dataloader import default_collate
import pandas as pd

from hisup.dataset.test_dataset_ import TestDatasetWithAnnotations as TestSetWithAnnotations


class TestDatasetWithAnnotations(Dataset):
    def __init__(self, root_t, ann_file_t,root_a, ann_file_a, csv_file, transform = None):
        # super(TestDatasetWithAnnotations, self).__init__(root_t, ann_file_t,root_a, ann_file_a, csv_file)
        self.tar_set = TestSetWithAnnotations(root_t, ann_file_t, transform)
        self.aux_set = TestSetWithAnnotations(root_a, ann_file_a, transform)

        self.pairs = pd.read_csv(csv_file)

        self.collate_fn = TestSetWithAnnotations.collate_fn

    
    def __getitem__(self, idx):
        img_tar, ann_tar = self.tar_set.__getitem__(self.pairs['target'][idx])
        img_aux, ann_aux = self.aux_set.__getitem__(self.pairs['auxiliary'][idx])

        return (img_tar, img_aux), (ann_tar, ann_aux)
    

    def __len__(self):
        return self.pairs.shape[0]
