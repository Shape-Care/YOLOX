#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from torch.utils.data import ConcatDataset
from yolox.exp import Exp as MyExp
from yolox.data import COCODataset, TrainTransform, MosaicDetection
from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
import torch

class ConcatDatasetWithPull(ConcatDataset):

    @property
    def input_dim(self):
        return self.datasets[0].input_dim

    def _get_dataset_and_local_idx(self, idx):
        if idx < 0:
            idx = len(self) + idx
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset, idx
            idx -= len(dataset)
        raise IndexError(f"Index {idx} out of range")

    def pull_item(self, idx):
        dataset, local_idx = self._get_dataset_and_local_idx(idx)
        return dataset.pull_item(local_idx)

    def load_anno(self, idx):
        dataset, local_idx = self._get_dataset_and_local_idx(idx)
        return dataset.load_anno(local_idx)

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.375
        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Główny dataset (i do walidacji)
        self.data_dir = "shapecare_datasets/akcesoria_60"
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.test_ann = "val.json"
        self.train_images = "train2017"
        self.val_images = "val2017"

        # Drugi dataset
        self.data_dir_2 = "shapecare_datasets/kamizelki_64"
        self.train_ann_2 = "train.json"
        self.train_images_2 = "train2017"

        self.num_classes = 1

        # --- Hiperparametry treningu dostosowane do fine-tuningu ---
        self.weight_decay = 5e-4
        self.momentum = 0.9

        self.train_batch_size = 32 
        self.eval_batch_size = 16

        self.save_history_ckpt = True

        # ── FAZA 2: pełny fine-tuning ───────────────────────────────
        self.max_epoch = 150
        self.eval_interval = 5

        self.basic_lr_per_img = 0.0001 / 64
        self.min_lr_ratio = 0.05
        self.warmup_epochs = 3

        # Augmentacje mocniejsze
        self.mosaic_prob = 0.75
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = True
        self.mixup_prob = 0.1
        self.no_aug_epochs = 15  
        # self.hsv_prob = 0.0
        # self.flip_prob = 0.0
        # self.degrees = 0.0
        # self.translate = 0.0
        # self.shear = 0.0
        # self.mixup_scale = (1.0, 1.0)

    def get_model(self):
        model = super().get_model()
        # Odmrożone — wszystkie parametry trenują
        for param in model.parameters():
            param.requires_grad = True
        return model

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
        #     img_size=self.input_size,
        #     preproc=TrainTransform(max_labels=120, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
        #     degrees=self.degrees,
        #     translate=self.translate,
        #     mosaic_scale=self.mosaic_scale,
        #     mixup_scale=self.mixup_scale,
        #     shear=self.shear,
        #     enable_mixup=self.enable_mixup,
        #     mosaic_prob=self.mosaic_prob,
        #     mixup_prob=self.mixup_prob,
        # )