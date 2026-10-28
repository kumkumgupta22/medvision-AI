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





## External dataset overlap audit

A one-time audit used [BDNeuro-MRI Version 9](https://data.mendeley.com/datasets/zwr4ntf94j/9), a four-class dataset published under CC BY 4.0. Its published test split has 889 images. The external set was compared against all 7,200 local `Training` and `Testing` images: 458 were exact SHA-256 matches and 149 additional images were near matches at a 64-bit perceptual-hash Hamming distance of 5 or less. These 607 images were excluded, leaving 282 images for an overlap-filtered evaluation.

On those remaining images, the saved EfficientNet-B0 checkpoint scored **76.95% accuracy** and **72.19% macro F1**. The remaining subset is class-imbalanced (136 glioma, 54 meningioma, 37 no tumor, 55 pituitary). Because of the extensive overlap and small filtered sample, this dataset does **not** qualify as a clean independent validation set, and these results should not be used to claim independent generalization or to tune the model. The evaluation is an overlap audit, not a clinical performance result. The dataset authors describe duplicate removal and stratified image-level splits; patient-level separation is not established in its included README.

The dataset and generated reports remain local and are excluded from Git. To reproduce the overlap-filtered evaluation after downloading and extracting BDNeuro-MRI V9 into `data/external_validation/BDNeuro-MRI-V9/`:

```powershell
python src\evaluate_external.py --data data\external_validation\BDNeuro-MRI-V9 --reference-data data\brain_tumor --checkpoint models\efficientnet_b0_best.pt --output outputs\bdneuro_external_validation
```
