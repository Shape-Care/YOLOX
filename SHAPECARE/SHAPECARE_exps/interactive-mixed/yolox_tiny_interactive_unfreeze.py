#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# SHAPECARE — InteractiveSport mixed — YOLOX-tiny — FAZA 2: pełny fine-tuning
#
# Resume from the matching FAZA-1 run (keep the -expn consistent per base):
#   python tools/train.py -f SHAPECARE/SHAPECARE_exps/interactive-mixed/yolox_tiny_interactive_unfreeze.py \
#       -c YOLOX_outputs/is_tiny_from_miap_freeze/best_ckpt.pth \
#       -expn is_tiny_from_miap_unfreeze --resume --fp16 -d 1 -b 16 -o -l mlflow

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # YOLOX-tiny
        self.depth = 0.33
        self.width = 0.375
        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # ── Dataset ─────────────────────────────────────────────────
        self.data_dir = "SHAPECARE/data/InteractiveSport_mixed"
        self.train_ann = "train.json"
        self.train_images = "train"
        self.val_ann = "val.json"
        self.test_ann = "val.json"
        self.val_images = "val"
        self.num_classes = 1

        # ── Ogólne ──────────────────────────────────────────────────
        self.data_num_workers = 4
        self.save_history_ckpt = True
        self.print_interval = 50
        self.eval_interval = 5

        # ── FAZA 2: pełny fine-tuning (backbone odmrożony) ──────────
        self.max_epoch = 120
        self.basic_lr_per_img = 0.00005 / 64
        self.min_lr_ratio = 0.05
        self.warmup_epochs = 3
        self.random_size = (10, 20)

        # Augmentacje mocniejsze — mix domen (różne sale/ludzie)
        self.mosaic_prob = 0.75
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = True
        self.mixup_prob = 0.15
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

    def get_dataset(self, cache=False, cache_type="ram"):
        from yolox.data import COCODataset, TrainTransform
        return COCODataset(
            data_dir=self.data_dir, json_file=self.train_ann, name=self.train_images,
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            cache=cache, cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        legacy = kwargs.get("legacy", False)
        return COCODataset(
            data_dir=self.data_dir, json_file=self.val_ann, name=self.val_images,
            img_size=self.test_size, preproc=ValTransform(legacy=legacy),
        )
