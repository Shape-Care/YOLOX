#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# SHAPECARE — Physio (general base) — YOLOX-s — FAZA 1: zamrożony backbone
#
# Baza (podaj przez -c):  /shared/shapecare-models/yolox_s_miap_levi_1.pth
# Dataset: SHAPECARE/data/Physio  (klasa Person, poprawione boxy CVAT; 4079 train / 441 val)
#
# To jest OGÓLNY model bazowy. best_ckpt.pth z FAZY 2 (unfreeze) jest punktem
# startowym do fine-tuningu na InteractiveSport.
#
#   python tools/train.py \
#       -f SHAPECARE/SHAPECARE_exps/physio/yolox_s_freeze.py \
#       -c /shared/shapecare-models/yolox_s_miap_levi_1.pth \
#       --fp16 -d 1 -b 16 -o -l mlflow

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
        self.print_interval = 50          # ~4079/bs=16 ≈ 255 iter/epokę
        self.eval_interval = 5

        # ── FAZA 1: zamrożony backbone (head dostraja się do physio) ─
        # Więcej danych niż w levicare/interactive (~4x), więc mniej epok.
        self.max_epoch = 25
        self.basic_lr_per_img = 0.00005 / 64   # mały LR — dostrajamy tylko head/neck
        self.min_lr_ratio = 0.05
        self.warmup_epochs = 3

        # Augmentacje umiarkowane (backbone zamrożony)
        self.mosaic_prob = 0.5
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = False
        self.mixup_prob = 0.0
        self.no_aug_epochs = 5
        self.hsv_prob = 0.5
        self.flip_prob = 0.5
        self.degrees = 0.0                 # osoby zawsze pionowo — bez rotacji
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
