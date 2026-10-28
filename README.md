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

The script writes `models/efficientnet_b0_best.pt` and results under `outputs/efficientnet_b0/`. If training is interrupted after a checkpoint is saved, evaluate that checkpoint without retraining using `python src\evaluate_efficientnet.py`. The evaluator writes the test metrics, per-class report, and confusion matrix under `outputs/efficientnet_b0/`. CPU training can take a long time; use `--batch-size 8` if memory is limited. The web app lets you choose any model with a local checkpoint.
## Web app

The web interface lets you choose the CNN baseline, EfficientNet-B0, or ViT-Tiny before uploading a single MRI image. It returns the predicted dataset class, model score, class scores, and an explanation image. CNN and EfficientNet use Grad-CAM over convolutional feature maps; ViT uses Grad-CAM over patch embeddings. Model scores are not calibrated clinical probabilities.

Each model requires its checkpoint in the local `models/` folder:

- `models/cnn_baseline_best.pt`
- `models/efficientnet_b0_best.pt`
- `models/vit_tiny_best.pt`

Train the model you want to use with the commands above. The app loads a checkpoint when that model is first selected. Checkpoints are ignored by Git and are not included in a fresh clone.

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Visit `http://127.0.0.1:8000`. Uploaded images are processed in memory and are not saved.
## Cloud deployment (Render)

The `render.yaml` blueprint configures a Python 3.12 web service with `/health` as its health check. Render free web services sleep after inactivity, lose local files when they restart, and have 512 MB RAM. PyTorch inference may exceed that memory limit; do not choose a paid plan without reviewing its current price.

The dataset and model checkpoints remain out of GitHub. To enable cloud predictions, place the three trained checkpoint files in a public Hugging Face model repository, then set the Render environment variable `MEDIVISION_MODEL_REPO` to that repository ID. The app downloads only the selected checkpoint on first use and verifies its SHA-256 before loading it. The expected files are `cnn_baseline_best.pt`, `efficientnet_b0_best.pt`, and `vit_tiny_best.pt`.

A public deployment sends uploaded MRI images to Render for processing. The current API handles image bytes in memory and does not save uploads, but this academic prototype is not for clinical use or patient-care decisions. Avoid uploading identifiable patient data.

To deploy, connect this GitHub repository to Render, choose **New + → Blueprint**, and let Render read `render.yaml`. Add `MEDIVISION_MODEL_REPO` in the service environment settings after the public model repository is ready. Confirm the deployment plan and any charges before creating or upgrading a service.

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
## Vision Transformer baseline

The first ViT comparison uses a compact **ViT-Tiny trained from scratch** (96×96 grayscale input, 16×16 patches, 64-dimensional embeddings, 3 transformer blocks, 4 attention heads). It uses the same seed-42, stratified 85/15 train/validation split from `Training`; `Testing` is only scored after selecting the best validation-loss checkpoint.

- Best validation accuracy: **76.43%** (epoch 15)
- Held-out test accuracy: **69.88%**
- Held-out macro F1: **69.10%**

On the same provided `Testing` split, the recorded CNN baseline is 75.94% accuracy / 75.25% macro F1, and EfficientNet-B0 is 95.38% / 95.27%. The ViT did not outperform these models and none of these results reaches the 97% target. The test split has been used to compare multiple models, so use validation data for future model choices and seek a genuinely independent, patient-level dataset for final generalization claims.

To reproduce the ViT run from the project folder:

```powershell
python src\train_vit.py --epochs 15 --batch-size 64 --patience 4 --threads 4 --seed 42
```

The best checkpoint is saved as `models/vit_tiny_best.pt`; histories, metrics, classification report, confusion matrix, and training curves are written under `outputs/vit_tiny/`. These generated files are excluded from Git.
