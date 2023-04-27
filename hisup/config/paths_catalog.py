import os
import os.path as osp

class DatasetCatalog(object):

    DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__),
                '..','..','data'))
    
    DATASETS = {
        'vietnam_osm_train': {
            'img_dir': '/home/thevi/Documents/data/osm_vietnam/imgs',
            'ann_file': '/home/thevi/Documents/data/osm_vietnam/labels/coco/train.json'
        },
        'vietnam_osm_val': {
            'img_dir': '/home/thevi/Documents/data/osm_vietnam/imgs',
            'ann_file': '/home/thevi/Documents/data/osm_vietnam/labels/coco/val.json'
        },
        'crowdai_train_small': {
            'img_dir': 'crowdai/train/images',
            'ann_file': 'crowdai/train/annotation-small.json'
        },
        'crowdai_test_small': {
            'img_dir': 'crowdai/val/images',
            'ann_file': 'crowdai/val/annotation-small.json'
        },
        'crowdai_train': {
            'img_dir': 'crowdai/train/images',
            'ann_file': 'crowdai/train/annotation.json'
        },
        'crowdai_test': {
            'img_dir': 'crowdai/val/images',
            'ann_file': 'crowdai/val/annotation.json'
        },
        'inria_train': {
            'img_dir': 'inria/train/images',
            'ann_file': 'inria/train/annotation.json',
        },
        'inria_test': {
            'img_dir': 'inria/test/images',
            'ann_file': 'inria/test/annotation.json',
        },
        'crowdai_demo': {
            'img_dir': 'crowdai/demo/images',
            'ann_file': 'crowdai/demo/annotation.json',
        }
    }

    DATASETS = {
        'osm_train': {
            'img_dir_t': '/home/thevi/Documents/data/osm_vietnam/target/imgs',
            'ann_file_t': '/home/thevi/Documents/data/osm_vietnam/target/labels/coco/train.json',
            'img_dir_a': '/home/thevi/Documents/data/osm_vietnam/auxiliary',
            'ann_file_a': '/home/thevi/Documents/data/osm_vietnam/auxiliary/coco-labels/train.json',
            'csv_file': '/home/thevi/Documents/data/osm_vietnam/cross_data/train.csv'
        },
        'osm_val': {
            'img_dir_t': '/home/thevi/Documents/data/osm_vietnam/target/imgs',
            'ann_file_t': '/home/thevi/Documents/data/osm_vietnam/target/labels/coco/val.json',
            'img_dir_a': '/home/thevi/Documents/data/osm_vietnam/auxiliary',
            'ann_file_a': '/home/thevi/Documents/data/osm_vietnam/auxiliary/coco-labels/val.json',
            'csv_file': '/home/thevi/Documents/data/osm_vietnam/cross_data/val.csv'
        },
        'osm_test': {
            'img_dir_t': '/home/thevi/Documents/data/osm_vietnam/target/imgs',
            'ann_file_t': '/home/thevi/Documents/data/osm_vietnam/target/labels/coco/test.json',
            'img_dir_a': '/home/thevi/Documents/data/osm_vietnam/auxiliary',
            'ann_file_a': '/home/thevi/Documents/data/osm_vietnam/auxiliary/coco-labels/test.json',
            'csv_file': '/home/thevi/Documents/data/osm_vietnam/cross_data/test.csv'
        }
    }

    @staticmethod
    def get(name):
        assert name in DatasetCatalog.DATASETS
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]

        args = dict(
            root_a = osp.join(data_dir,attrs['img_dir_t']),
            ann_file_a = osp.join(data_dir,attrs['ann_file_t']),
            root_b = osp.join(data_dir,attrs['img_dir_a']),
            ann_file_b = osp.join(data_dir,attrs['ann_file_a']),
            csv_file = osp.join(data_dir,attrs['csv_file'])
        )

        if 'train' in name:
            return dict(factory="TrainDataset",args=args)
        if 'test' in name:
            return dict(factory="TestDatasetWithAnnotations",
                        args=args)
        raise NotImplementedError()
