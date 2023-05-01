import os
import os.path as osp

class DatasetCatalog(object):

    DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__),
                '..','..','data'))
    
    DATASETS = {
        'vietnam_osm_train': {
            'img_dir': '/home/ubuntu/Documents/data/osm_vietnam/imgs',
            'ann_file': '/home/ubuntu/Documents/data/osm_vietnam/labels/coco/train.json'
        },
        'vietnam_osm_val': {
            'img_dir': '/home/ubuntu/Documents/data/osm_vietnam/imgs',
            'ann_file': '/home/ubuntu/Documents/data/osm_vietnam/labels/coco/val.json'
        },
        'crowdai_train_small': {
            'img_dir': '/home/ubuntu/Documents/data/crowdai/train/images',
            'ann_file': '/home/ubuntu/Documents/data/crowdai/train/annotation-small.json'
        },
        'crowdai_test_small': {
            'img_dir': '/home/ubuntu/Documents/data/crowdai/val/images',
            'ann_file': '/home/ubuntu/Documents/data/crowdai/val/annotation-small.json'
        },
        'crowdai_train': {
            'img_dir': '/home/ubuntu/Documents/data/crowdai/train/images',
            'ann_file': '/home/ubuntu/Documents/data/crowdai/train/annotation.json'
        },
        'crowdai_test': {
            'img_dir': '/home/ubuntu/Documents/data/crowdai/val/images',
            'ann_file': '/home/ubuntu/Documents/data/crowdai/val/annotation.json'
        },
        'inria_train': {
            'img_dir': '/home/ubuntu/Documents/data/inria/train/images',
            'ann_file': '/home/ubuntu/Documents/data/inria/train/annotation.json',
        },
        'inria_test': {
            'img_dir': '/home/ubuntu/Documents/data/inria/test/images',
            'ann_file': '/home/ubuntu/Documents/data/inria/test/annotation.json',
        },
        'crowdai_demo': {
            'img_dir': '/home/ubuntu/Documents/data/crowdai/demo/images',
            'ann_file': '/home/ubuntu/Documents/data/crowdai/demo/annotation.json',
        }
    }

    @staticmethod
    def get(name):
        assert name in DatasetCatalog.DATASETS
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]

        args = dict(
            root = osp.join(data_dir,attrs['img_dir']),
            ann_file = osp.join(data_dir,attrs['ann_file'])
        )

        if 'train' in name:
            return dict(factory="TrainDataset",args=args)
        if 'test' in name and 'ann_file' in attrs:
            return dict(factory="TestDatasetWithAnnotations",
                        args=args)
        raise NotImplementedError()
