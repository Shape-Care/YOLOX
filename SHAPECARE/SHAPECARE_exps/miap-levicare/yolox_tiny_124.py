#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from torch.utils.data import ConcatDataset
from yolox.exp import Exp as MyExp
from yolox.data import COCODataset, TrainTransform, MosaicDetection
from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
import torch

class ConcatDatasetWithPull(ConcatDataset):
    def pull_item(self, idx):
        if idx < 0:
            idx = len(self) + idx

        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset.pull_item(idx)
            idx -= len(dataset)

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.375
        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.mosaic_prob = 0.0
        self.enable_mixup = False
        self.mixup_prob = 0.0
        self.hsv_prob = 0.0
        self.flip_prob = 0.0
        self.degrees = 0.0
        self.translate = 0.0
        self.shear = 0.0
        self.mosaic_scale = (1.0, 1.0)
        self.mixup_scale = (1.0, 1.0)

        self.random_size = (10, 20)

        # Główny dataset (i do walidacji)
        self.data_dir = "shapecare_datasets/akcesoria_60"
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.test_ann = "annotations/val.json"
        self.train_images = "train2017"
        self.val_images = "val2017"

        # Drugi dataset
        self.data_dir_2 = "shapecare_datasets/kamizelki_64"
        self.train_ann_2 = "train.json"
        self.train_images_2 = "train2017"

        self.num_classes = 1

        # --- Hiperparametry treningu dostosowane do fine-tuningu ---
        self.max_epoch = 100
        self.no_aug_epochs = 100    
        self.warmup_epochs = 3
        self.basic_lr_per_img = 0.00001 / 64.0 
        self.min_lr_ratio = 0.05
        self.weight_decay = 5e-4
        self.momentum = 0.9

        self.train_batch_size = 32 
        self.eval_batch_size = 16

        self.eval_interval = 5
        self.save_history_ckpt = True


    def get_dataset(self, cache=False, cache_type="ram"):

        def make_base_dataset(data_dir, ann, img_folder):
            return COCODataset(
                data_dir=data_dir,
                json_file=ann,
                img_size=self.input_size,
                name=img_folder,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob
                ),
                cache=cache,
                cache_type=cache_type,
            )

        # SUROWE datasety (bez Mosaic)
        ds1 = make_base_dataset(self.data_dir, self.train_ann, self.train_images)
        ds2 = make_base_dataset(self.data_dir_2, self.train_ann_2, self.train_images_2)

        concat_ds = ConcatDatasetWithPull([ds1, ds2])

        return concat_ds

        # return MosaicDetection(
        #     dataset=concat_ds,
        #     mosaic=False,
        #     img_size=self.input_size,
        #     preproc=TrainTransform(
        #         max_labels=50,
        #         flip_prob=self.flip_prob,
        #         hsv_prob=self.hsv_prob
        #     ),
        #     degrees=self.degrees,
        #     translate=self.translate,
        #     mosaic_scale=self.mosaic_scale,
        #     mixup_scale=self.mixup_scale,
        #     shear=self.shear,
        #     enable_mixup=self.enable_mixup,
        #     mosaic_prob=self.mosaic_prob,
        #     mixup_prob=self.mixup_prob,
        # )

