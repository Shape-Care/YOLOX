#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.375
        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.enable_mixup = False # True 
        self.mosaic_prob = 0.5 #0.25
        #self.mixup_prob = 0.05
        
        # Define yourself dataset path
        self.data_dir = "shapecare_datasets/akcesoria_60"
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.test_ann = "val.json"
        self.train_images = "images/train"
        self.val_images = "images/val"
        
        self.num_classes = 1
        self.eval_interval = 5
