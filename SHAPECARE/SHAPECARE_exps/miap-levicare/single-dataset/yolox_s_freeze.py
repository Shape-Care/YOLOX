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

        # ── FAZA 1: zamrożony backbone ──────────────────────────────
        self.max_epoch = 50
        self.eval_interval = 5

        self.basic_lr_per_img = 0.00005 / 64   # mały LR — head uczy się od zera
        self.min_lr_ratio = 0.05
        self.warmup_epochs = 3

        # Augmentacje umiarkowane (backbone zamrożony, nie ma sensu mocno augmentować)
        self.mosaic_prob = 0.5
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = False
        self.mixup_prob = 0.0
        self.random_size = (10, 20)
        self.no_aug_epochs = 5             # ostatnie 5 epok bez mosaic/mixup

    def get_model(self):
        model = super().get_model()

        # Zamroź backbone
        for param in model.backbone.parameters():
            param.requires_grad = False

        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        total  = sum(p.numel() for p in model.parameters())
        print(f"[FAZA 1] Zamrożono {frozen:,} / {total:,} parametrów")

        return model
        
