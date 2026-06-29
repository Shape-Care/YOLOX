#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Dataset path
        self.data_dir = "shapecare_datasets/kamizelki_64"
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.test_ann = "val.json"
        self.train_images = "images/train"
        self.val_images = "images/val"

        self.num_classes = 1

        # ── FAZA 2: pełny fine-tuning ───────────────────────────────
        self.max_epoch = 150
        self.eval_interval = 5

        self.basic_lr_per_img = 0.00001 / 64   # bardzo mały LR — nie zniszczyć cech z MIAP
        self.min_lr_ratio = 0.05
        self.warmup_epochs = 1

        # Augmentacje mocniejsze — model już wie co szukać
        self.mosaic_prob = 0.75
        self.enable_mixup = True
        self.mixup_prob = 0.1
        self.no_aug_epochs = 15


    def get_model(self):
        model = super().get_model()
        # Odmrożone — wszystkie parametry trenują
        for param in model.parameters():
            param.requires_grad = True
        return model
        
