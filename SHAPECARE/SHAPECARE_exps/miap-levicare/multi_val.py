"""Multi-source COCO validation helper.

Wraps several COCODataset instances into one that yolox.evaluators.COCOEvaluator
can use unchanged: it exposes a merged `.coco` (pycocotools index) and `.class_ids`,
and yields per-image (img, target, info, id) like a normal COCODataset.

Image IDs are offset per source so the merged COCO has no ID collisions.
Categories must be consistent across sources (same category_id meaning).
"""

import os
from torch.utils.data import ConcatDataset
from pycocotools.coco import COCO
from yolox.data import COCODataset, ValTransform


class MergedCOCOValDataset(ConcatDataset):

    ID_OFFSET_STEP = 10 ** 7

    def __init__(self, sub_datasets):
        for i, ds in enumerate(sub_datasets):
            offset = i * self.ID_OFFSET_STEP
            if offset:
                for img in ds.coco.dataset["images"]:
                    img["id"] += offset
                for ann in ds.coco.dataset["annotations"]:
                    ann["id"] += offset
                    ann["image_id"] += offset
                ds.coco.createIndex()
                ds.ids = ds.coco.getImgIds()
                ds.annotations = ds._load_coco_annotations()
                ds.path_filename = [os.path.join(ds.name, a[3]) for a in ds.annotations]

        super().__init__(sub_datasets)

        merged = COCO()
        merged.dataset = {
            "images":      sum((d.coco.dataset["images"]      for d in sub_datasets), []),
            "annotations": sum((d.coco.dataset["annotations"] for d in sub_datasets), []),
            "categories":  sub_datasets[0].coco.dataset["categories"],
        }
        merged.createIndex()

        self.coco = merged
        self.class_ids = sub_datasets[0].class_ids
        self.cats = sub_datasets[0].cats
        self.ids = sorted(merged.getImgIds())
        self.num_imgs = sum(d.num_imgs for d in sub_datasets)

    @property
    def input_dim(self):
        return self.datasets[0].input_dim


def build_eval_dataset(sources, img_size, legacy=False):
    """sources: list of (data_dir, json_file, image_folder_name) tuples.
    Returns a single COCODataset for one source, or a MergedCOCOValDataset for many."""
    sub = [
        COCODataset(
            data_dir=dd,
            json_file=ann,
            name=imgs,
            img_size=img_size,
            preproc=ValTransform(legacy=legacy),
        )
        for dd, ann, imgs in sources
    ]
    total = sum(len(d) for d in sub)
    print(f"[ValDataset] {total} ramek z {len(sub)} zrodel: "
          + ", ".join(str(len(d)) for d in sub))
    return sub[0] if len(sub) == 1 else MergedCOCOValDataset(sub)
