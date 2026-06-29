#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from torch.utils.data import ConcatDataset
from yolox.exp import Exp as MyExp
from yolox.data import COCODataset, TrainTransform
from multi_val import build_eval_dataset


class ConcatDatasetWithPull(ConcatDataset):

    @property
    def input_dim(self):
        return self.datasets[0].input_dim

    def _get_dataset_and_local_idx(self, idx):
        if idx < 0:
            idx = len(self) + idx
        for ds in self.datasets:
            if idx < len(ds):
                return ds, idx
            idx -= len(ds)
        raise IndexError(f"Index {idx} out of range")

    def pull_item(self, idx):
        ds, local = self._get_dataset_and_local_idx(idx)
        return ds.pull_item(local)

    def load_anno(self, idx):
        ds, local = self._get_dataset_and_local_idx(idx)
        return ds.load_anno(local)


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # YOLOX-s
        self.depth = 0.33
        self.width = 0.50
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # ── Treningowe źródła ───────────────────────────────────────
        # Główne (pierwsze) źródło = również źródło walidacji (val_ann/val_images)
        self.data_dir = "shapecare_datasets/brzesko_mixed_470"
        self.train_ann = "train.json"
        self.train_images = "train2017"

        self.data_dir_2 = "shapecare_datasets/squash_mixed_dataset_590"
        self.train_ann_2 = "train.json"
        self.train_images_2 = "train2017"

        # Trzecie źródło treningowe – opcjonalne, ustaw None aby wyłączyć
        self.data_dir_3 = "shapecare_datasets/brzesko_val_50"
        self.train_ann_3 = "val.json"
        self.train_images_3 = "val2017"

        self.ann_dir_4 = "shapecare_datasets/squash_val_50_2"
        self.ann_ann_4 = "val.json"
        self.ann_images_4 = "val2017"

        # ── Walidacyjne źródła ──────────────────────────────────────
        # 1) z głównego data_dir (above)
        self.val_ann = "val.json"
        self.test_ann = "val.json"
        self.val_images = "val2017"

        # 2) drugie źródło walidacji – opcjonalne, ustaw None aby wyłączyć
        self.val_dir_2 = "shapecare_datasets/squash_val_50_1"
        self.val_ann_2 = "val.json"
        self.val_images_2 = "val2017"

        # 3) trzecie źródło walidacji – opcjonalne, ustaw None aby wyłączyć
        self.val_dir_3 = None
        self.val_ann_3 = "val.json"
        self.val_images_3 = "val2017"

        # 4) czwarte źródło walidacji – opcjonalne, ustaw None aby wyłączyć
        self.val_dir_4 = None
        self.val_ann_4 = "val.json"
        self.val_images_4 = "val2017"

        self.num_classes = 1

        # --- Hiperparametry (fine-tuning na ~1000 ramkach) ---
        self.data_num_workers = 4
        self.weight_decay = 5e-4
        self.momentum = 0.9

        self.train_batch_size = 8
        self.eval_batch_size = 8

        self.save_history_ckpt = True
        self.print_interval = 20

        # ── FAZA 2: pełny fine-tuning ───────────────────────────────
        self.max_epoch = 150
        self.eval_interval = 5

        self.basic_lr_per_img = 0.00005 / 64
        self.min_lr_ratio = 0.05
        self.warmup_epochs = 3

        self.mosaic_prob = 0.75
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = True
        self.mixup_prob = 0.15
        self.no_aug_epochs = 15
        self.hsv_prob = 0.5
        self.flip_prob = 0.5
        self.degrees = 0.0
        self.translate = 0.1
        self.shear = 0.0
        self.mixup_scale = (0.5, 1.5)

    def get_model(self):
        model = super().get_model()
        for p in model.parameters():
            p.requires_grad = True
        return model

    def _make_train_coco(self, data_dir, ann, img_folder, cache, cache_type):
        return COCODataset(
            data_dir=data_dir,
            json_file=ann,
            img_size=self.input_size,
            name=img_folder,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_dataset(self, cache=False, cache_type="ram"):
        datasets = [
            self._make_train_coco(self.data_dir, self.train_ann, self.train_images, cache, cache_type),
            self._make_train_coco(self.data_dir_2, self.train_ann_2, self.train_images_2, cache, cache_type),
        ]
        if self.data_dir_3:
            datasets.append(
                self._make_train_coco(self.data_dir_3, self.train_ann_3, self.train_images_3, cache, cache_type)
            )

        total = sum(len(d) for d in datasets)
        print(f"[TrainDataset] {total} ramek z {len(datasets)} zrodel: "
              + ", ".join(str(len(d)) for d in datasets))
        return ConcatDatasetWithPull(datasets)

    def get_eval_dataset(self, **kwargs):
        legacy = kwargs.get("legacy", False)
        testdev = kwargs.get("testdev", False)
        ann_attr = "test_ann" if testdev else "val_ann"

        sources = [(self.data_dir, getattr(self, ann_attr), self.val_images)]
        for i in range(2, 10):
            d = getattr(self, f"val_dir_{i}", None)
            if d:
                sources.append((d, getattr(self, f"val_ann_{i}"), getattr(self, f"val_images_{i}")))

        return build_eval_dataset(sources, self.test_size, legacy=legacy)
