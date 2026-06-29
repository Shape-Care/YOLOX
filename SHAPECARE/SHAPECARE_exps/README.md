# SHAPECARE — YOLOX fine-tuning

Experiment configs and training/export commands for SHAPECARE YOLOX models.
Run all commands from the YOLOX repo root (`/shared/YOLOX`).

Configs live in `SHAPECARE_exps/<experiment>/` (e.g. `miap-levicare/`),
base checkpoints in `/shared/shapecare-models/`, and training outputs are
written to `YOLOX_outputs/<exp-name>/`.

## 1. Set the dataset paths in the exp config

Each experiment `.py` file defines its own dataset paths in the `Exp.__init__`.
Edit them to point at your data (COCO format) before training:

```python
self.data_dir   = "SHAPECARE/data/your-dataset"  # folder with images
self.train_ann  = "train.json"                        # COCO annotations (train)
self.val_ann    = "val.json"                           # COCO annotations (val)
self.test_ann   = "val.json"
```

If the config merges several datasets, set the additional ones too
(`self.data_dir_2`, `self.train_ann_2`, ...). Paths are relative to the
YOLOX repo root.

## 2. Configure MLflow

Set the MLflow connection in `.env` (tracking URI, experiment name, run name):

```
MLFLOW_TRACKING_URI="http://office-fractal:5050"
MLFLOW_EXPERIMENT_NAME="yolox-finetune"
YOLOX_MLFLOW_RUN_NAME="your-run-name"
```

MLflow logging is enabled by passing `-l mlflow` to `tools/train.py`.

## 3. Stage 1 — train with frozen backbone

```bash
python tools/train.py \
    -f SHAPECARE_exps/miap-levicare/yolox_s_124_freeze.py \
    -c /shared/shapecare-models/yolox_s_miap_1.pth \
    --fp16 \
    -d 1 -b 16 -o -l mlflow
```

## 4. Stage 2 — unfreeze and resume

Continue from the best checkpoint produced by stage 1
(`YOLOX_outputs/<stage1-exp>/best_ckpt.pth`):

```bash
python tools/train.py \
    -f SHAPECARE_exps/miap-levicare/yolox_s_124_unfreeze.py \
    -c YOLOX_outputs/yolox_s_124_freeze/best_ckpt.pth \
    --resume --fp16 \
    -d 1 -b 16 -o -l mlflow
```

## 5. Export to ONNX

```bash
python tools/export_onnx.py \
    -f SHAPECARE_exps/miap-levicare/yolox_tiny_124_unfreeze.py \
    -c /shared/shapecare-models/yolox_tiny_miap_levi_2_124.pth \
    --output-name yolox.onnx \
    --opset 18 \
    --decode_in_inference
```

## Flag reference

| Flag | Meaning |
|------|---------|
| `-f` | path to the experiment config (`.py`) |
| `-c` | checkpoint to start from (fine-tune) / resume |
| `--resume` | resume training (optimizer + epoch) from `-c` |
| `--fp16` | mixed-precision training |
| `-d 1` | number of GPUs/devices |
| `-b 16` | total batch size |
| `-o` | occupy GPU memory up front |
| `-l mlflow` | log to MLflow |
| `--opset 18` | ONNX opset version (export) |
| `--decode_in_inference` | include detection decoding inside the ONNX graph |
