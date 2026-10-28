# MediVision AI — Project 604

Academic prototype for **brain MRI image/slice classification** into glioma, meningioma, no tumor, and pituitary classes. This software is not a clinical diagnostic system and must not guide patient care.

## Measured results

The current runs produced these actual results on the provided `Testing` split:

- **CNN baseline:** best validation accuracy 83.33%; held-out test accuracy 75.94%; macro F1 75.25%.
- **EfficientNet-B0:** the best saved checkpoint was epoch 5 with 99.05% validation accuracy; on the held-out test set it reached **95.38% accuracy** and **95.27% macro F1**. Glioma recall was 82.75% (331/400 correct).

The 97% accuracy target has not been achieved. These are academic image/slice classification results, not clinical performance claims. Both models have now been evaluated on `Testing`; use a new independent external set for any future model selection or confirmation, and do not tune against `Testing`.
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

The script writes `models/efficientnet_b0_best.pt` and results under `outputs/efficientnet_b0/`. If training is interrupted after a checkpoint is saved, evaluate that checkpoint without retraining using `python src\evaluate_efficientnet.py`. The evaluator writes the test metrics, per-class report, and confusion matrix under `outputs/efficientnet_b0/`. CPU training can take a long time; use `--batch-size 8` if memory is limited. The web app still uses the CNN checkpoint.
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




