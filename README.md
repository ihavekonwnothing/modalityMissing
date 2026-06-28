# Modality Missing SAR-Optical Water Segmentation

Code for SAR-optical water segmentation under missing optical modality.

The main model in this repository is `mask_aware_cross_attention_fusion_unet`, a dual-encoder U-Net with:

- SAR input: Sentinel-1 `VV,VH`
- optical input: Sentinel-2 `Blue,Green,Red,NIR`
- optical availability mask: `1=available`, `0=missing`
- mask-guided high-resolution fusion
- mask-aware cross-attention at deeper stages
- auxiliary SAR segmentation head for full optical missing fallback

Data, checkpoints, cached patches, generated figures, and experiment outputs are intentionally not committed.

## Setup

```bash
conda create -n modality_missing python=3.10 -y
conda activate modality_missing
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build that matches your system if the default `pip install torch` is not appropriate.

## Data

Set the S1S2-Water root:

```bash
export S1S2_WATER_ROOT=/path/to/S1S2-Water
```

Optional patch cache:

```bash
python tools/build_s1s2_water_patch_cache.py \
  --root "$S1S2_WATER_ROOT" \
  --output data/s1s2_water_patch_cache_512 \
  --patch-size 512
```

Sen1Floods11 zero-shot transfer expects the prepared 6-band layout:

```text
transfer_dataset/Sen1Floods11_6band/
  metadata.csv
  images/
  labels/
  valid_masks/
```

## Train Final Model

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/s1s2_water/mask_aware_cross_attention_fusion.yaml
```

Two GPUs with DDP:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29503 train.py \
  --config configs/s1s2_water/mask_aware_cross_attention_fusion.yaml
```

Default output:

```text
outputs/s1s2_water/proposed/mask_aware_cross_attention_fusion_unet_ddp
```

## Evaluate

Clean S1S2-Water test:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --config configs/s1s2_water/mask_aware_cross_attention_fusion.yaml \
  --checkpoint outputs/s1s2_water/proposed/mask_aware_cross_attention_fusion_unet_ddp/checkpoints/best.ckpt \
  --split test
```

Controlled optical degradation suite:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_degradation.py \
  --config configs/s1s2_water/mask_aware_cross_attention_fusion.yaml \
  --checkpoint outputs/s1s2_water/proposed/mask_aware_cross_attention_fusion_unet_ddp/checkpoints/best.ckpt \
  --split test \
  --suite \
  --output_csv outputs/s1s2_water/degradation_suite/mask_aware_cross_attention_best.csv
```

Sen1Floods11 zero-shot transfer:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_zero_shot_sen1floods11.py \
  --config configs/s1s2_water/mask_aware_cross_attention_fusion.yaml \
  --model mask_aware_cross_attention_fusion_unet \
  --checkpoint outputs/s1s2_water/proposed/mask_aware_cross_attention_fusion_unet_ddp/checkpoints/best.ckpt \
  --data-root transfer_dataset/Sen1Floods11_6band \
  --stats-path data/s1s2_water_patch_cache_512/stats.json \
  --output-dir outputs/zero_shot_sen1floods11_sar_x100/mask_aware_cross_attention_fusion_unet \
  --sar-preprocess target_x100_source_stats
```

## Fallback Ablation

This script compares global scalar fallback and pixelwise fallback without retraining:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_fallback_ablation.py \
  --config configs/s1s2_water/mask_aware_cross_attention_fusion.yaml \
  --checkpoint outputs/s1s2_water/proposed/mask_aware_cross_attention_fusion_unet_ddp/checkpoints/best.ckpt \
  --s1s2-cache-dir data/s1s2_water_patch_cache_512 \
  --transfer-root transfer_dataset/Sen1Floods11_6band \
  --stats-path data/s1s2_water_patch_cache_512/stats.json \
  --output-dir outputs/pixelwise_ablation
```

## Tests

```bash
python -m unittest tests.test_mask_aware_cross_attention_fusion_unet tests.test_fallback_rules
```

## Main Files

- `models/mask_aware_cross_attention_fusion_unet.py`: final model.
- `models/mask_guided_late_fusion_unet.py`: mask-guided late-fusion baseline.
- `models/SMAGnet.py` and `models/smagnet_adapter.py`: SMAGNet comparison model adapter.
- `train.py`: S1S2-Water training with AMP, DDP, checkpoint resume, logging, and controlled optical missing simulation.
- `evaluate.py`: clean S1S2-Water evaluation.
- `evaluate_degradation.py`: S1S2-Water controlled optical degradation evaluation.
- `tools/evaluate_zero_shot_sen1floods11.py`: transfer evaluation.
- `tools/evaluate_fallback_ablation.py`: global vs pixelwise fallback ablation.
