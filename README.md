# PET/CT 3D Vascular Segmentation Benchmark

This repository implements a **comparative framework for 3D PET/CT
vascular segmentation** using MONAI.

The goal is to evaluate multiple deep learning architectures under a
**shared preprocessing, augmentation, and training pipeline** to ensure
a fair comparison.

Currently supported architectures include:

-   SwinUNETR
-   UNETR
-   UNet
-   Attention UNet
-   SegResNet

Additional architectures can easily be added through the `models.py`
factory.

------------------------------------------------------------------------

# Shared preprocessing pipeline

The preprocessing and augmentation pipeline is aligned with the
DiffU-Net transforms previously used in this project.

All models use the **same preprocessing and augmentation pipeline** to
ensure fair comparison.

Shared preprocessing includes:

-   identical dataset split logic
-   identical patch size: `96 × 96 × 96`
-   `CropForegroundd`
-   `SpatialPadd`
-   `RandCropByPosNegLabeld`
-   flip augmentations with identical probabilities
-   modality-aware intensity normalization
-   intensity scaling and shifting augmentations

------------------------------------------------------------------------

# Label encoding

Labels are stored as **integer masks with values 0--4**.

One-hot conversion is **not applied during preprocessing**.

Instead, the loss function (`DiceCELoss`) handles one-hot encoding
internally.

This keeps preprocessing identical across models.

------------------------------------------------------------------------

# Channel order

Images are loaded as:

    image[0] = CT  (*_0001.nii.gz)
    image[1] = PET (*_0000.nii.gz)

Intensity normalization ranges:

    CT  : [-1000, 1200]
    PET : [0, 5]

------------------------------------------------------------------------

# Dataset layout

    /data/pchatzi/nnUNet_data/nnUNet_raw/Dataset001_Vascular/

    imagesTr/
        CASE001_0000.nii.gz   # PET
        CASE001_0001.nii.gz   # CT

    labelsTr/
        CASE001.nii.gz

------------------------------------------------------------------------

# Dataset split policy

Default split:

    train : 70%
    val   : 10%
    test  : 20%
    seed  : 123

A split manifest is saved for reproducibility:

    logs/<run>/split_manifest.json

This guarantees that **all models use exactly the same dataset split**.

------------------------------------------------------------------------

# Repository structure

    petct_segmentation_benchmark/

    configs/
        base_petct_vascular.yaml
        swin_unetr.yaml
        unet.yaml
        attention_unet.yaml
        segresnet.yaml
        unetr.yaml
        vnet.yaml

    scripts/
        run_train.sh

    src/
        data.py
        engine.py
        losses_and_metrics.py
        models.py
        train.py
        transforms.py
        utils.py

    preprocessing_inspection/
        inspect_preprocessing.py

------------------------------------------------------------------------

# Running training

Training uses a **shared base configuration + model-specific
configuration**.

Example: run SwinUNETR

    python -m src.train \
        --base_config configs/base_petct_vascular.yaml \
        --model_config configs/swin_unetr.yaml

or using the helper script:

    bash scripts/run_train.sh configs/swin_unetr.yaml

------------------------------------------------------------------------

# Running with nohup

For long training runs:

    nohup bash scripts/run_train.sh configs/swin_unetr.yaml > swin_unetr.out 2>&1 &

------------------------------------------------------------------------

# Output structure

Each run creates a timestamped directory:

    logs/YYYYMMDD_HHMMSS_runname/

Example:

    logs/20260311_112908_swin_unetr_petct_vascular/

Run directory contents:

    models/        saved model checkpoints
    artifacts/     config files + training script snapshot
    run_info.txt   run metadata
    wandb/         W&B logs

The saved artifacts ensure that each experiment can be reproduced later.

------------------------------------------------------------------------

# Metrics

The following metrics are tracked during training:

    dice_mean_score
    dice_abdominal
    dice_aortic_arch
    dice_ascending
    dice_descending
    best_dice_mean_score

These correspond to label indices:

    0 background
    1 abdominal aorta
    2 aortic arch
    3 ascending aorta
    4 descending aorta

------------------------------------------------------------------------

# Hardware assumptions

The experiments assume:

    single GPU training
    patch size: 96³
    batch size: 1

Configuration parameters can be adjusted for different hardware setups.

------------------------------------------------------------------------

# Reproducing experiments

This repository is designed so experiments can be reproduced exactly
using saved configuration files.

Each experiment uses:

-   a **shared base configuration** (`base_petct_vascular.yaml`)
-   a **model-specific configuration** (`swin_unetr.yaml`, `unet.yaml`,
    etc.)

The base configuration defines:

-   dataset paths
-   preprocessing pipeline
-   augmentation policy
-   optimizer settings
-   training schedule

Model configs define:

-   architecture parameters
-   run naming
-   W&B configuration

This ensures that **all models are trained under identical conditions**.

------------------------------------------------------------------------

# Adding a new model

To add a new architecture:

1.  Implement the model in `src/models.py`.
2.  Create a new config file inside `configs/`.

Example:

    configs/unet.yaml
    configs/segresnet.yaml
    configs/unetr.yaml

Each model config only needs to specify:

    run_name
    model parameters
    wandb settings

Training can then be launched with:

    bash scripts/run_train.sh configs/unet.yaml

------------------------------------------------------------------------

# Fair comparison policy

All models share:

-   identical preprocessing
-   identical augmentation
-   identical dataset split
-   identical optimizer
-   identical training schedule
-   identical evaluation metrics

Only the **network architecture** changes between experiments.

This ensures a fair and reproducible comparison across models.
