#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Dataset path
        self.data_dir = "shapecare_datasets/levi_20"
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.test_ann = "val.json"
        self.train_images = "images/train"
        self.val_images = "images/val"

        # Training setup
        self.max_epoch = 300
        self.eval_interval = 10
        self.num_classes = 1

        # Augmentation settings
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.mosaic_prob = 0.25
        self.enable_mixup = True
        self.mixup_prob = 0.05

        self.test_conf = 0.001
        self.input_size = (640, 640)
        self.test_size = (640, 640)
