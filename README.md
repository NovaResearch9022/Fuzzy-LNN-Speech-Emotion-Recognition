# Fuzzy-LNN-Speech-Emotion-Recognition
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22909572.svg)](https://doi.org/10.5281/zenodo.22909572)
This repository provides the Python implementation of a **Fuzzy Liquid Neural Network (Fuzzy-LNN)** framework for Speech Emotion Recognition (SER).

The public implementation focuses on experiments conducted using the **CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)** dataset.

## Overview

The proposed framework combines conventional acoustic features, dimensionality reduction, fuzzy clustering, and a Liquid Neural Network.

The processing pipeline is:

```text
Speech Signal
     ↓
MFCC + Δ + Δ²
     ↓
120-D Acoustic Features
     ↓
StandardScaler
     ↓
PCA
     ↓
Fuzzy C-Means (FCM)
     ↓
FCM-derived Distance Representation
     ↓
Liquid Neural Network (LNN/LTC)
     ↓
Emotion Classification
```

## Acoustic Features

Audio recordings are resampled to **16 kHz**.

For each utterance, the following acoustic features are extracted:

- 40 MFCC coefficients
- 40 first-order delta coefficients
- 40 second-order delta coefficients

This produces a **120-dimensional acoustic representation**.

## Dataset

The implementation uses the **CREMA-D** dataset.

The dataset itself is **not distributed with this repository**. Users should obtain CREMA-D separately and provide the path to the `AudioWAV` directory when running the code.

Expected structure:

```text
CREMA-D/
└── AudioWAV/
    ├── 1001_DFA_ANG_XX.wav
    ├── 1001_DFA_DIS_XX.wav
    ├── ...
```

The six emotion classes considered are:

```text
ANG  - Anger
DIS  - Disgust
FEA  - Fear
HAP  - Happiness
NEU  - Neutral
SAD  - Sadness
```

## Experimental Protocol

The implementation uses a **5-fold evaluation protocol**.

Training and test preprocessing are kept separate to prevent information leakage. In each fold, preprocessing transformations such as feature standardization, PCA, and Fuzzy C-Means fitting are performed using the training data and subsequently applied to the corresponding test data.

A fixed random seed of **42** is used for reproducibility.

## Installation

Clone the repository:

```bash
git clone https://github.com/NovaResearch9022/Fuzzy-LNN-Speech-Emotion-Recognition.git
cd Fuzzy-LNN-Speech-Emotion-Recognition
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Experiment

Run:

```bash
python fuzzy_lnn_cremad.py --dataset_path "/path/to/CREMA-D/AudioWAV"
```

For example:

```bash
python fuzzy_lnn_cremad.py --dataset_path "./CREMA-D/AudioWAV"
```

The script performs feature extraction, preprocessing, fuzzy clustering, LNN training, and evaluation.

## Repository Structure

```text
Fuzzy-LNN-Speech-Emotion-Recognition/
│
├── fuzzy_lnn_cremad.py
├── README.md
├── requirements.txt
└── LICENSE
```

### `fuzzy_lnn_cremad.py`

Main implementation of the CREMA-D Fuzzy-LNN experiment.

### `requirements.txt`

Python dependencies required to reproduce the experiment.

### `LICENSE`

License governing use of the source code.

## Reproducibility

The implementation uses a fixed random seed where applicable.

All preprocessing operations that require fitting are fitted using the corresponding training data only. Test data are transformed using the parameters learned from the training partition.

No pre-computed experimental results are embedded in the main implementation. Results are generated when the experiment is executed.

## Citation

If you use this code in your research, please cite the associated publication
and the archived software release:

**Software DOI:** 10.5281/zenodo.22909572

## License

This project is released under the MIT License.

## Contact

For questions concerning the implementation, please use the GitHub repository's Issues section.
