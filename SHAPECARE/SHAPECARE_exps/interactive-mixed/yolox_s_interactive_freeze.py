#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# SHAPECARE — InteractiveSport mixed (brzesko + squash + podleze) — YOLOX-s — FAZA 1
#
# Target-domain fine-tuning. ONE merged train set → shuffling puts brzesko,
# squash and podleze frames in every batch (different rooms/lighting/people).
#
# Start from EITHER base (choose with -c), give each run its own output dir:
#   # from the general MIAP/Levicare person detector:
#   python tools/train.py -f SHAPECARE/SHAPECARE_exps/interactive-mixed/yolox_s_interactive_freeze.py \
#       -c /shared/shapecare-models/yolox_s_miap_levi_1.pth \
#       -expn is_s_from_miap_freeze --fp16 -d 1 -b 16 -o -l mlflow
#   # from the physio general base:
#   python tools/train.py -f SHAPECARE/SHAPECARE_exps/interactive-mixed/yolox_s_interactive_freeze.py \
#       -c YOLOX_outputs/yolox_s_unfreeze/best_ckpt.pth \
#       -expn is_s_from_physio_freeze --fp16 -d 1 -b 16 -o -l mlflow

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

        # ── Dataset (single merged COCO; file_name = "<source>/<frame>") ──
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

        # ── FAZA 1: zamrożony backbone ──────────────────────────────
        self.max_epoch = 40
        self.basic_lr_per_img = 0.0005 / 64
        self.min_lr_ratio = 0.05
        self.warmup_epochs = 3

        # Augmentacje umiarkowane (backbone zamrożony)
        self.mosaic_prob = 0.5
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = False
        self.mixup_prob = 0.0
        self.no_aug_epochs = 5
        self.hsv_prob = 0.5          # różne sale/oświetlenie: brzesko vs squash vs podleze
        self.flip_prob = 0.5
        self.degrees = 0.0
        self.translate = 0.1
        self.shear = 0.0

    def get_model(self):
        model = super().get_model()
        for p in model.backbone.parameters():
            p.requires_grad = False
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[FAZA 1] Zamrozono {frozen:,} / {total:,} parametrow")
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
