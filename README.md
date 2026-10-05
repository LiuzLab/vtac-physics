# Physics-Informed Deep Learning for False VT Alarm Reduction in the ICU

Code for the paper: **[arXiv:2609.08992](https://arxiv.org/abs/2609.08992)**

## Abstract

False ventricular tachycardia (VT) alarms are a leading contributor to alarm fatigue in intensive care units. We propose a deep learning framework combining a 1D SE-ResNet with ICU-realistic data augmentations and a physics-informed auxiliary reconstruction task based on the three-element Windkessel hemodynamic model, implemented as a differentiable forward simulation. By requiring the network’s latent representation to produce physiologically plausible arterial pressure waveforms, artifact-driven ECG patterns are penalized while true VT remains coherent across modalities. Evaluated on the VTaC benchmark under a strict real-time protocol (10 s pre-alarm window), our method achieves a Challenge Score of 85.08 ± 1.65, a 5-point improvement over prior state-of-the-art. Ablation studies confirm that the physics-informed objective is the primary performance driver, providing gains in accuracy, ~2× label efficiency, and more localized and clinically meaningful ECG segments.

## Setup

Python 3.8+; a CUDA GPU is recommended.

```bash
pip install -r requirements.txt
```

Download the VTaC dataset from PhysioNet (https://physionet.org/content/vtac/1.0/). The data folder must contain `waveforms/`, `benchmark_data_split.csv` and `event_labels.csv`. Below, `DATA=/path/to/vtac/1.0`.

Build the data cache once (last 10 s before each alarm, 14 channel slots + availability mask); all later commands reuse it:

```bash
python main.py --mode cache --data_root $DATA --cache_dir cache
```

## 1. Hyperparameter search

Optuna (TPE), maximizing the validation Challenge Score. Several workers can share one study; `--n_trials` is the total across workers (a worker may finish one extra trial after the limit is reached):

```bash
for w in 0 1 2 3 4; do
  python main.py --mode optuna --data_root $DATA --cache_dir cache \
      --n_trials 50 --worker_id $w --storage optuna_journal.log --output_dir optuna_w$w \
      --gpu_mem_frac 0.18 &
  sleep 10
done; wait
```

The best configuration is written to `best_hparams.json` (next to `optuna_journal.log`). `--gpu_mem_frac` caps each process's share of GPU memory when several run on one GPU.

## 2. Training

Train and test one model per seed with `best_hparams.json` (default seeds: 265, 1789, 3655, 6398, 8515):

```bash
python main.py --mode eval --data_root $DATA --cache_dir cache --output_dir outputs
```

Use `--seeds 1 2 3` for other seeds and `--hparams_file` for another configuration. Outputs per seed: `outputs/best_model_seed_<seed>.pt` and `outputs/result_seed_<seed>.json`; all runs are appended to `outputs/run_log.jsonl`.

## 3. Inference

Run trained checkpoints on the test split. Each checkpoint stores its hyperparameters and decision threshold (selected on the validation set).

```bash
python main.py --mode predict --data_root $DATA --cache_dir cache \
    --checkpoint outputs/best_model_seed_*.pt --output_dir predictions
```

Writes `predictions/predictions_test_<checkpoint>.csv` (record, event, label, probability, prediction) and `predictions/predict_test_summary.json` (metrics per checkpoint). Use `--split val` or `--split train` for other splits.
