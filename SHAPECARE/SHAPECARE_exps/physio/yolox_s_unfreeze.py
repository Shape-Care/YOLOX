#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# SHAPECARE — Physio (general base) — YOLOX-s — FAZA 2: pełny fine-tuning
#
# Kontynuacja z best_ckpt.pth z FAZY 1 (patrz --resume):
#   python tools/train.py \
#       -f SHAPECARE/SHAPECARE_exps/physio/yolox_s_unfreeze.py \
#       -c YOLOX_outputs/yolox_s_freeze/best_ckpt.pth \
#       --resume --fp16 -d 1 -b 16 -o -l mlflow

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # YOLOX-s
        self.depth = 0.33
        self.width = 0.50
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # ── Dataset ─────────────────────────────────────────────────
        self.data_dir = "SHAPECARE/data/Physio"
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.test_ann = "val.json"
        self.train_images = "train2017"
        self.val_images = "val2017"
        self.num_classes = 1

        # ── Ogólne ──────────────────────────────────────────────────
        self.data_num_workers = 4
        self.save_history_ckpt = True
        self.print_interval = 50
        self.eval_interval = 5

        # ── FAZA 2: pełny fine-tuning (backbone odmrożony) ──────────
        self.max_epoch = 100
        self.basic_lr_per_img = 0.00001 / 64   # bardzo mały LR — nie zniszczyć cech MIAP/Levi
        self.min_lr_ratio = 0.05
        self.warmup_epochs = 1

        # Augmentacje mocniejsze — model już wie czego szukać
        self.mosaic_prob = 0.75
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = True
        self.mixup_prob = 0.1
        self.mixup_scale = (0.5, 1.5)
        self.no_aug_epochs = 15
        self.hsv_prob = 0.5
        self.flip_prob = 0.5
        self.degrees = 0.0
        self.translate = 0.1
        self.shear = 0.0

    def get_model(self):
        model = super().get_model()
        for p in model.parameters():
            p.requires_grad = True
        return model
