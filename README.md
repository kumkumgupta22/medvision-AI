# MediVision AI — Project 604

Academic prototype for **brain MRI image/slice classification** into glioma, meningioma, no tumor, and pituitary classes. This software is not a clinical diagnostic system and must not guide patient care.

## Current measured baseline

The latest CPU run used seed 42 and early-stopped after 7 epochs. Its best checkpoint was epoch 4 with **83.33% validation accuracy**. On the held-out `Testing` split it achieved **75.94% accuracy** and **75.25% macro F1**. A 97% score is a target, not a promise, and has not been achieved. Do not tune against the test split or report unmeasured results as achieved.

## Dataset

The expected layout is:

```text
data/brain_tumor/
├── Training/{glioma,meningioma,notumor,pituitary}/
└── Testing/{glioma,meningioma,notumor,pituitary}/
```

The local dataset currently contains 7,200 readable images: 1,400 per class in `Training` and 400 per class in `Testing`. The dataset is excluded from Git by `.gitignore`. For a fresh copy, download a four-class MRI dataset from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), accept its dataset terms on Kaggle, and place the extracted `Training` and `Testing` folders in `data/brain_tumor/`. Kaggle downloads may require an authenticated Kaggle CLI setup; never commit Kaggle credentials.

## Setup (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you already have the project `venv`, you can use `venv\Scripts\python.exe` in place of `python` below.

## Verify and train

```powershell
python src\dataset_analysis.py
python src\train_cnn.py --epochs 8 --batch-size 32 --seed 42
```

The data pipeline converts images to grayscale, resizes to 128×128, normalizes them, and uses a deterministic, stratified 85/15 train/validation split from `Training`. `Testing` stays held out until final evaluation. Training augmentation uses small rotations and horizontal flips. CPU training may take a while; lower `--batch-size` to 16 or 8 if needed.

Training writes a best checkpoint to `models/` and metrics, history, classification report, and confusion matrix to `outputs/`. Those generated files are ignored by Git. The training script accepts `--epochs`, `--batch-size`, `--learning-rate`, `--patience`, `--threads`, `--seed`, `--data`, `--output`, and `--model-dir`.

## Transfer-learning experiment

An EfficientNet-B0 option fine-tunes ImageNet-pretrained visual features and writes its own checkpoint and reports, leaving the baseline CNN intact. The first run downloads the official TorchVision pretrained weights; `--from-scratch` disables that download but is not expected to perform as well.

```powershell
python src\train_efficientnet.py --epochs 15 --batch-size 16 --seed 42
```

The script uses the same deterministic stratified validation split and held-out test set. It writes `models/efficientnet_b0_best.pt` and results under `outputs/efficientnet_b0/`. On CPU this can take a long time; use `--batch-size 8` if memory is limited. The current web app uses the baseline CNN checkpoint; compare the actual held-out metrics before changing the serving model. No accuracy target is guaranteed.
## Web app

After training creates `models/cnn_baseline_best.pt`:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Visit `http://127.0.0.1:8000`. The app accepts one image at a time and returns a class prediction, scores, and a Grad-CAM visualization. Uploaded images are processed in memory and are not saved.

## GitHub

The repository should contain source, documentation, and configuration only. Dataset files, archives, virtual environments, secrets, model weights, and generated results are ignored. To connect and push, use the repository URL from your GitHub page:

```powershell
git init
git add .
git status
git commit -m "Initial MediVision AI project"
git branch -M main
git remote add origin <your-GitHub-repository-URL>
git push -u origin main
```

Review `git status` before committing to confirm no data, weights, or secrets are staged.


