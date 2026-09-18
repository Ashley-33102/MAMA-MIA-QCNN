# 3D Breast Tumor Segmentation with a Hybrid CNN-Quantum Architecture

A hybrid **3D CNN + quantum neural network (QNN)** for breast tumor segmentation from **DCE-MRI**, evaluated on a 100-case subset of the [MAMA-MIA](https://doi.org/10.1038/s41597-025-04707-4) dataset.

The model uses a lightweight 3D encoder-decoder with skip connections for spatial feature extraction and reconstruction, while a **4-qubit variational quantum circuit** is placed at the bottleneck as a global, sigmoid-gated channel modulation mechanism.

A matched **classical-only ablation** replaces the quantum layer with a classical MLP, allowing the contribution of the quantum component to be examined under the same overall architecture.

> **Scope:** This is a one-day feasibility study. The results should not be interpreted as evidence that the quantum layer improves overall segmentation accuracy.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch\&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-3D%20Medical%20Imaging-005CED?logo=monai\&logoColor=white)
![PennyLane](https://img.shields.io/badge/PennyLane-Quantum%20ML-6C3FC5?logo=pennylane\&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy\&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-Metrics-F7931E?logo=scikit-learn\&logoColor=white)
![NiBabel](https://img.shields.io/badge/NiBabel-NIfTI%20Processing-4B8BBE?logo=python\&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Datasets-FFD21E?logo=huggingface\&logoColor=black)

---

## Results at a Glance

The table below compares the full hybrid model with the matched classical-only ablation.

| Metric               |          With QNN | Without QNN (Ablation) |
| -------------------- | ----------------: | ---------------------: |
| Dice (mean / median) | **0.232 / 0.092** |          0.218 / 0.074 |
| IoU (mean)           |         **0.165** |                  0.163 |
| Precision (mean)     |             0.175 |              **0.177** |
| Recall (mean)        |         **0.725** |                  0.562 |
| HD95 (mean, mm)      |         **101.6** |                  104.3 |

The clearest difference between the two configurations was **recall**: the QNN configuration achieved higher mean recall while maintaining similar precision. The overall Dice and IoU values remained low, so the result is best viewed as a feasibility observation rather than evidence of a general segmentation improvement.

Full per-case and per-cohort results are available in `results/metrics/`, while the complete methodology, analysis, and limitations are provided in `docs/report.docx`.

---

## Dataset

This project uses a **100-case subset of MAMA-MIA**, with 25 cases sampled from each of four cohorts:

* ISPY1
* ISPY2
* NACT
* Duke

The project uses **one post-contrast DCE-MRI phase per case**.

The dataset itself is **not included in this repository**.

The project uses [MAMA-MIA-Lite](https://huggingface.co/datasets/YongchengYAO/MAMA-MIA-Lite), a Hugging Face redistribution derived from the original [MAMA-MIA](https://github.com/LidiaGarrucho/MAMA-MIA) dataset.

For the original dataset publication:

> L. Garrucho et al., "A large-scale multicenter breast cancer DCE-MRI benchmark dataset with expert segmentations," *Scientific Data*, 2025.

---

## Data Access

Download the dataset using Hugging Face:

```bash
pip install huggingface_hub

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='YongchengYAO/MAMA-MIA-Lite', repo_type='dataset', local_dir='./MAMA_MIA_raw')"
```

After extraction and case selection, arrange the selected data as:

```text
MAMA_MIA_100/
├── images/
│   ├── case_001.nii.gz
│   ├── case_002.nii.gz
│   └── ...
└── masks/
    ├── case_001.nii.gz
    ├── case_002.nii.gz
    └── ...
```

Each image must have a corresponding segmentation mask with the same filename.

---

## Pipeline

The complete pipeline consists of preprocessing, dataset splitting, model validation, training, and evaluation.

```text
Raw DCE-MRI
     │
     ▼
Preprocessing
     │
     ├── Isotropic resampling
     ├── Foreground-based cropping
     ├── Fixed-size resizing
     └── Normalization
     │
     ▼
Preprocessed 3D volumes
     │
     ▼
Fixed-seed train / validation / test split
     │
     ▼
Preflight checks
     │
     ├── Encoder-decoder
     ├── Quantum module
     └── Combined model
     │
     ▼
Hybrid 3D CNN + QNN
     │
     ▼
Tumor probability mask
     │
     ▼
Test-set evaluation
     │
     ├── Dice
     ├── IoU
     ├── Precision
     ├── Recall
     └── HD95
```

---

## Architecture

The model combines a lightweight 3D convolutional encoder-decoder with a quantum bottleneck.

```text
                    3D DCE-MRI
                 128 × 128 × 128
                         │
                         ▼
              ┌─────────────────────┐
              │   3D CNN Encoder    │
              │      4 stages        │
              └─────────────────────┘
                         │
              Skip Connections
                 ┌───────┴───────┐
                 │               │
                 ▼               │
          AdaptiveAvgPool3d      │
                 │               │
                 ▼               │
              Linear             │
             128 → 4             │
                 │               │
                 ▼               │
        ┌──────────────────┐     │
        │ 4-Qubit Variational│    │
        │   Quantum Circuit │     │
        └──────────────────┘     │
                 │               │
                 ▼               │
              Linear             │
              4 → 128             │
                 │               │
              Sigmoid             │
                 │               │
                 ▼               │
        Channel Modulation        │
                 │               │
                 └───────┬───────┘
                         ▼
              ┌─────────────────────┐
              │    3D Decoder       │
              │ Skip connections    │
              │ fused at each stage │
              └─────────────────────┘
                         │
                         ▼
                Tumor Probability
                      Mask
```

### Quantum Bottleneck

The quantum circuit operates on the **globally pooled bottleneck representation**, rather than individual voxels.

The bottleneck follows:

```text
128 feature channels
       │
       ▼
Global Average Pooling
       │
       ▼
   Linear 128 → 4
       │
       ▼
4-qubit Variational Circuit
       │
       ▼
   4 quantum outputs
       │
       ▼
   Linear 4 → 128
       │
       ▼
    Sigmoid
       │
       ▼
Channel-wise gate
       │
       ▼
Bottleneck modulation
```

The quantum circuit therefore performs **global channel modulation once per volume**.

It does not operate independently on every voxel. A voxel-wise quantum implementation at this resolution would require thousands of circuit evaluations per forward pass and was rejected during the design stage as computationally impractical.

---

## Classical Ablation

To examine the contribution of the quantum component, a classical-only configuration uses the **same encoder-decoder architecture** while replacing the quantum module with a classical MLP.

```text
                 Hybrid Model
                      │
              ┌───────┴───────┐
              │               │
        Quantum Path      Classical Path
              │               │
        4-qubit QNN          MLP
              │               │
              └───────┬───────┘
                      │
                Same Decoder
                      │
                      ▼
              Segmentation Mask
```

This matched design keeps the surrounding architecture unchanged, making the comparison focused on the bottleneck component.

---

## Preprocessing

Each MRI volume passes through the following preprocessing steps:

1. **Isotropic resampling** to 2.0 mm spacing.
2. **Image-derived foreground cropping**.
3. **Resize** to a fixed `128 × 128 × 128` volume.
4. **Intensity normalization**.
5. **Disk caching** for efficient training.

The 2.0 mm spacing was selected as a practical compromise for a one-day feasibility experiment using full `128³` volumes.

---

## Training

The model is trained using:

* **Loss:** Dice + Cross-Entropy
* **Optimizer:** AdamW
* **Early stopping**
* **Best-checkpoint selection**
* **Fixed dataset split**
* **Single random seed per configuration**

The reported experiment was performed on an **NVIDIA RTX A4500**.

---

## Reproduction

### 1. Create the environment

```bash
conda create -n mama-mia python=3.10
conda activate mama-mia

pip install -r requirements.txt
```

> The reported experiment used PyTorch 2.14 with a CUDA 13.0 build. A CUDA/PyTorch installation compatible with your own NVIDIA driver may be required.

For the appropriate PyTorch installation, see the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 2. Prepare the dataset

Place the selected cases under:

```text
./MAMA_MIA_100/
```

with:

```text
MAMA_MIA_100/
├── images/
└── masks/
```

### 3. Analyze the preprocessing requirements

Run the analysis step before preprocessing the complete dataset:

```bash
python src/preprocess_mama_mia.py \
    --data-dir ./MAMA_MIA_100 \
    --analyze-only
```

### 4. Preprocess

```bash
python src/preprocess_mama_mia.py \
    --data-dir ./MAMA_MIA_100 \
    --spacing-mm 2.0 \
    --target-size 128
```

### 5. Create the dataset split

```bash
python src/split_dataset.py \
    --report ./MAMA_MIA_100/preprocessing_output/preprocessing_report_128.csv
```

### 6. Run preflight checks

Before full training, verify each component independently:

```bash
python preflight_checks/test_stage1_encoder_decoder.py
python preflight_checks/test_stage2_qnn.py
python preflight_checks/test_stage3_combined.py
```

### 7. Train the hybrid model

```bash
python src/train.py \
    --cache-dir ./MAMA_MIA_100/preprocessing_output/MAMA_MIA_100_preprocessed_128 \
    --split-json ./MAMA_MIA_100/preprocessing_output/split_seed.json \
    --out-dir ./training_output
```

### 8. Evaluate

```bash
python src/evaluate.py \
    --checkpoint ./training_output/best_model.pt \
    --cache-dir ./MAMA_MIA_100/preprocessing_output/MAMA_MIA_100_preprocessed_128 \
    --split-json ./MAMA_MIA_100/preprocessing_output/split_seed.json \
    --history-json ./training_output/training_history.json \
    --out-dir ./results \
    --spacing-mm 2.0
```

### 9. Reproduce the classical ablation

Train the same architecture without the QNN:

```bash
python src/train.py \
    --cache-dir ./MAMA_MIA_100/preprocessing_output/MAMA_MIA_100_preprocessed_128 \
    --split-json ./MAMA_MIA_100/preprocessing_output/split_seed.json \
    --out-dir ./training_output_no_qnn \
    --no-qnn
```

Then evaluate the resulting checkpoint using a separate output directory.

---

## Repository Structure

```text
MAMA-MIA-QCNN/
│
├── src/
│   ├── preprocess_mama_mia.py
│   ├── split_dataset.py
│   ├── dataset.py
│   ├── model.py
│   ├── metrics.py
│   ├── train.py
│   ├── evaluate.py
│   └── export_test_results_json.py
│
├── preflight_checks/
│   ├── test_stage1_encoder_decoder.py
│   ├── test_stage2_qnn.py
│   └── test_stage3_combined.py
│
├── docs/
│   ├── proposal.md
│   └── report.docx
│
├── results/
│   ├── metrics/
│   └── figures/
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

### Key Components

| File / Directory             | Purpose                           |
| ---------------------------- | --------------------------------- |
| `src/preprocess_mama_mia.py` | MRI preprocessing and caching     |
| `src/split_dataset.py`       | Fixed-seed dataset splitting      |
| `src/dataset.py`             | PyTorch dataset implementation    |
| `src/model.py`               | 3D CNN + QNN architecture         |
| `src/metrics.py`             | Segmentation metrics              |
| `src/train.py`               | Model training                    |
| `src/evaluate.py`            | Test evaluation and visualization |
| `preflight_checks/`          | Component-level validation        |
| `docs/proposal.md`           | Final project proposal            |
| `docs/report.docx`           | Full methodology and analysis     |
| `results/metrics/`           | Per-case and per-cohort results   |
| `results/figures/`           | Training and qualitative figures  |

---

## What's Included

### Included

* Complete source code
* QNN and classical ablation implementation
* Preprocessing pipeline
* Dataset splitting code
* Training and evaluation scripts
* Preflight sanity checks
* Per-case metrics
* Per-cohort metrics
* Training and evaluation figures
* Project proposal
* Full project report
* `requirements.txt`
* MIT License

### Not Included

The following are intentionally excluded:

**Raw MAMA-MIA data**

The medical imaging dataset is not redistributed with this repository. Users should obtain the data through the appropriate dataset source and comply with its applicable terms.

**Preprocessed tensor cache**

The generated `.pt` cache files are large and fully reproducible from the preprocessing script.

**Model checkpoints**

Training checkpoints such as `best_model.pt` are excluded from version control. They can be regenerated using the training pipeline.

**Full qualitative output**

Only a small curated set of qualitative examples is included. Complete qualitative outputs can be regenerated using the evaluation script.

**Local environment files**

No machine-specific paths, virtual environments, or local configuration files are required by the source code.

---

## Known Limitations

This project was intentionally scoped as a **one-day feasibility study** rather than a full-scale medical segmentation benchmark.

Important limitations include:

* Training uses **whole volumes** rather than foreground-oversampled patches.
* The preprocessing spacing of **2.0 mm** is coarser than the dataset's recommended 1.0 mm spacing.
* Full `128³` volumes impose a practical resolution constraint.
* Only **one random seed** was used for each configuration.
* The test set is small, with **4 cases per cohort**.
* The experiment uses only a **single DCE-MRI phase**.
* The relatively low Dice and IoU values indicate that the segmentation task remains challenging under this constrained setup.
* The quantum layer was evaluated as a feasibility component rather than as a demonstrated replacement for established segmentation architectures.
* No claim of statistically significant superiority of the QNN configuration is made.

These limitations and their potential effects are discussed in detail in `docs/report.docx`.

---

## Interpretation

The experiment does **not** show a broad improvement in segmentation performance from adding the quantum layer.

Instead, the main observable difference was:

```text
                    With QNN     Without QNN
Recall                0.725          0.562
Precision             0.175          0.177
Dice                  0.232          0.218
IoU                   0.165          0.163
HD95                  101.6          104.3
```

The QNN configuration produced more inclusive tumor predictions, reflected primarily in its higher recall. However, precision remained similar and overall Dice/IoU remained modest.

This makes the experiment useful primarily as a **feasibility and architecture study** rather than as evidence for improved clinical segmentation performance.

---

## Citation

If you use the underlying MAMA-MIA dataset, please cite:

```text
L. Garrucho et al., "A large-scale multicenter breast cancer DCE-MRI
benchmark dataset with expert segmentations," Scientific Data, 2025.
```

Dataset:

[MAMA-MIA](https://github.com/LidiaGarrucho/MAMA-MIA)

MAMA-MIA-Lite:

[MAMA-MIA-Lite](https://huggingface.co/datasets/YongchengYAO/MAMA-MIA-Lite)

---

## Built With

* [PyTorch](https://pytorch.org/)
* [MONAI](https://monai.io/)
* [PennyLane](https://pennylane.ai/)
* [NumPy](https://numpy.org/)
* [scikit-learn](https://scikit-learn.org/)
* [NiBabel](https://nipy.org/nibabel/)
* [Hugging Face](https://huggingface.co/)

---

## License

The source code in this repository is released under the **MIT License**.

See [`LICENSE`](LICENSE) for the complete license text.

The MIT License applies **only to the source code in this repository**. It does not apply to the MAMA-MIA dataset or other third-party materials, which remain subject to their respective licenses and terms.
