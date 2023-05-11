import os
import os.path as osp

class DatasetCatalog(object):

    DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__),
                '..','..','data'))

    DATASETS = {
        'osm_train': {
            'img_dir_t': '/home/ubuntu/Documents/data/osm_vietnam/target/imgs',
            'ann_file_t': '/home/ubuntu/Documents/data/osm_vietnam/target/labels/coco/train.json',
            'img_dir_a': '/home/ubuntu/Documents/data/osm_vietnam/auxiliary',
            'ann_file_a': '/home/ubuntu/Documents/data/osm_vietnam/auxiliary/coco-labels/train.json',
            'csv_file': '/home/ubuntu/Documents/data/osm_vietnam/cross_data/train.csv'
        },
        'osm_val': {
            'img_dir_t': '/home/ubuntu/Documents/data/osm_vietnam/target/imgs',
            'ann_file_t': '/home/ubuntu/Documents/data/osm_vietnam/target/labels/coco/val.json',
            'img_dir_a': '/home/ubuntu/Documents/data/osm_vietnam/auxiliary',
            'ann_file_a': '/home/ubuntu/Documents/data/osm_vietnam/auxiliary/coco-labels/val.json',
            'csv_file': '/home/ubuntu/Documents/data/osm_vietnam/cross_data/val.csv'
        },
        'osm_test': {
            'img_dir_t': '/home/ubuntu/Documents/data/osm_vietnam/target/imgs',
            'ann_file_t': '/home/ubuntu/Documents/data/osm_vietnam/target/labels/coco/test.json',
            'img_dir_a': '/home/ubuntu/Documents/data/osm_vietnam/auxiliary',
            'ann_file_a': '/home/ubuntu/Documents/data/osm_vietnam/auxiliary/coco-labels/test.json',
            'csv_file': '/home/ubuntu/Documents/data/osm_vietnam/cross_data/test.csv'
        },
        'crowdai-osm_train': {
            'img_dir_t': '/home/ubuntu/Documents/data/osm_vietnam/target/imgs/',
            'ann_file_t': '/home/ubuntu/Documents/data/osm_vietnam/target/labels/coco/train.json',
            'img_dir_a': '/home/ubuntu/Documents/data/crowdai/train/imgs/',
            'ann_file_a': '/home/ubuntu/Documents/data/crowdai/train/annotation-small.json',
            'csv_file': '/home/ubuntu/Documents/data/osm_vietnam/cross_data/osmvn-crowdai-ids/train.csv'
        },
        'crowdai-osm_val': {
            'img_dir_t': '/home/ubuntu/Documents/data/osm_vietnam/target/imgs/',
            'ann_file_t': '/home/ubuntu/Documents/data/osm_vietnam/target/labels/coco/val.json',
            'img_dir_a': '/home/ubuntu/Documents/data/crowdai/val/imgs/',
            'ann_file_a': '/home/ubuntu/Documents/data/crowdai/val/annotation-small.json',
            'csv_file': '/home/ubuntu/Documents/data/osm_vietnam/cross_data/osmvn-crowdai-ids/val.csv'
        },
        'crowdai-osm_test': {
            'img_dir_t': '/home/ubuntu/Documents/data/osm_vietnam/target/imgs/',
            'ann_file_t': '/home/ubuntu/Documents/data/osm_vietnam/target/labels/coco/test.json',
            'img_dir_a': '/home/ubuntu/Documents/data/osm_vietnam/target/imgs/',
            'ann_file_a': '/home/ubuntu/Documents/data/osm_vietnam/target/labels/coco/test.json',
            'csv_file': '/home/ubuntu/Documents/data/osm_vietnam/cross_data/osmvn-crowdai-ids/test.csv'
        },
        'crowdai-osmctdn_train': {
            'img_dir_t': '/home/ubuntu/Documents/data/osm_vietnam/target/imgs/',
            'ann_file_t': '/home/ubuntu/Documents/data/osm_vietnam/target/labels/coco/train.json',
            'img_dir_a': '/home/ubuntu/Documents/data/crowdai/train/imgs/',
            'ann_file_a': '/home/ubuntu/Documents/data/crowdai/train/annotation-small.json',
            'csv_file': '/home/ubuntu/Documents/data/osm_vietnam/cross_data/ctdn_osm-crowdai-ids/train.csv'
        },
        'crowdai-osmctdn_val': {
            'img_dir_t': '/home/ubuntu/Documents/data/osm_vietnam/target/imgs/',
            'ann_file_t': '/home/ubuntu/Documents/data/osm_vietnam/target/labels/coco/val.json',
            'img_dir_a': '/home/ubuntu/Documents/data/crowdai/val/imgs/',
            'ann_file_a': '/home/ubuntu/Documents/data/crowdai/val/annotation-small.json',
            'csv_file': '/home/ubuntu/Documents/data/osm_vietnam/cross_data/ctdn_osm-crowdai-ids/val.csv'
        },
        'crowdai-osmctdn_test': {
            'img_dir_t': '/home/ubuntu/Documents/data/osm_vietnam/target/imgs/',
            'ann_file_t': '/home/ubuntu/Documents/data/osm_vietnam/target/labels/coco/test.json',
            'img_dir_a': '/home/ubuntu/Documents/data/osm_vietnam/target/imgs/',
            'ann_file_a': '/home/ubuntu/Documents/data/osm_vietnam/target/labels/coco/test.json',
            'csv_file': '/home/ubuntu/Documents/data/osm_vietnam/cross_data/ctdn_osm-crowdai-ids/test.csv'
        }
    }

    @staticmethod
    def get(name):
        assert name in DatasetCatalog.DATASETS
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]

        args = dict(
            root_t = osp.join(data_dir,attrs['img_dir_t']),
            ann_file_t = osp.join(data_dir,attrs['ann_file_t']),
            root_a = osp.join(data_dir,attrs['img_dir_a']),
            ann_file_a = osp.join(data_dir,attrs['ann_file_a']),
            csv_file = osp.join(data_dir,attrs['csv_file'])
        )

        if 'train' in name:
            return dict(factory="TrainDataset",args=args)
        if 'test' in name:
            return dict(factory="TestDatasetWithAnnotations",
                        args=args)
        raise NotImplementedError()
