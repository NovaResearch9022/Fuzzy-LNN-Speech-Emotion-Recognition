"""
Fuzzy-LNN Speech Emotion Recognition on CREMA-D
================================================

Clean reproducibility script derived from the accompanying experimental notebook.

Pipeline:
    CREMA-D audio
    -> 40 MFCC + 40 delta + 40 delta-delta (120-D mean-pooled vector)
    -> StandardScaler (fit on training data only)
    -> PCA (1 component; fit on training data only)
    -> Fuzzy C-Means (6 clusters; fit on training data only)
    -> squared distances to FCM centers
    -> LTC / Liquid Neural Network
    -> evaluation

Notes
-----

- Set DATASET_PATH below, or pass --dataset_path on the command line.
- The split follows the notebook design: a fixed 6/6 utterance split and
  five speaker-disjoint KFold partitions (seed=42).
"""

import argparse
import os
import random
import time

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf

from fcmeans import FCM
from ncps import wirings
from ncps.keras import LTC

from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
N_FOLDS = 5
SAMPLE_RATE = 16000
N_MFCC = 40
PCA_COMPONENTS = 1
FCM_MAX_ITER = 200
EPOCHS = 30
BATCH_SIZE = 1

EMOTIONS = ["NEU", "ANG", "HAP", "DIS", "SAD", "FEA"]

# Change this path, or use:
# python fuzzy_lnn_cremad.py --dataset_path /path/to/AudioWAV
DATASET_PATH = "./CREMA-D/AudioWAV"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# CREMA-D FILE PARSING
# ============================================================

def parse_filename(file_path):
    """Return speaker ID, utterance ID, and emotion from a CREMA-D filename."""
    filename = os.path.basename(file_path)
    parts = filename.split("_")

    if len(parts) < 3:
        raise ValueError(f"Unexpected CREMA-D filename: {filename}")

    return parts[0], parts[1], parts[2]


def list_valid_audio_files(dataset_path):
    """Return valid CREMA-D WAV files used by the experiment."""
    files = []

    for filename in sorted(os.listdir(dataset_path)):
        if not filename.lower().endswith(".wav"):
            continue

        # Ignore accidental duplicate files if present.
        if filename.startswith("Copy of "):
            continue

        full_path = os.path.join(dataset_path, filename)

        try:
            _, _, emotion = parse_filename(full_path)
        except ValueError:
            continue

        if emotion in EMOTIONS:
            files.append(full_path)

    if not files:
        raise RuntimeError(
            f"No valid CREMA-D WAV files were found in: {dataset_path}"
        )

    return files


def get_speakers_and_utterances(audio_files):
    speakers = sorted({
        parse_filename(path)[0]
        for path in audio_files
    })

    utterances = sorted({
        parse_filename(path)[1]
        for path in audio_files
    })

    return speakers, utterances


# ============================================================
# OUTER SPLITS
# ============================================================

def create_splits(audio_files):
    """
    Create the five outer folds.

    The notebook uses:
      - one fixed random selection of 6 training utterance IDs;
      - the remaining utterance IDs for testing;
      - 5-fold speaker-disjoint KFold splitting.
    """
    speakers, all_utterances = get_speakers_and_utterances(audio_files)

    rng = random.Random(SEED)

    train_utterances = sorted(
        rng.sample(all_utterances, 6)
    )

    test_utterances = sorted(
        set(all_utterances) - set(train_utterances)
    )

    speaker_array = np.asarray(speakers)

    kfold = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    splits = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(speaker_array)):
        train_speakers = speaker_array[train_idx].tolist()
        test_speakers = speaker_array[test_idx].tolist()

        assert not (
            set(train_speakers) & set(test_speakers)
        ), "Speaker leakage detected."

        assert not (
            set(train_utterances) & set(test_utterances)
        ), "Utterance leakage detected."

        splits.append({
            "fold": fold + 1,
            "train_speakers": train_speakers,
            "test_speakers": test_speakers,
            "train_utterances": train_utterances,
            "test_utterances": test_utterances,
        })

    return splits


def select_paths(
    audio_files,
    selected_speakers,
    selected_utterances,
):
    """Select audio paths satisfying both speaker and utterance constraints."""
    selected_speakers = set(selected_speakers)
    selected_utterances = set(selected_utterances)

    paths = []

    for path in audio_files:
        speaker, utterance, emotion = parse_filename(path)

        if (
            speaker in selected_speakers
            and utterance in selected_utterances
            and emotion in EMOTIONS
        ):
            paths.append(path)

    return sorted(paths)


def group_test_paths_by_emotion(paths):
    grouped = {emotion: [] for emotion in EMOTIONS}

    for path in paths:
        _, _, emotion = parse_filename(path)
        grouped[emotion].append(path)

    return grouped


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_mfcc_features(file_paths):
    """
    Extract one 120-D vector per utterance:
        40 MFCC + 40 delta + 40 delta-delta,
    followed by temporal mean pooling.
    """
    features = []
    labels = []

    for file_path in file_paths:
        audio, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            mono=True,
        )

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC,
        )

        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        combined = np.vstack([
            mfcc,
            mfcc_delta,
            mfcc_delta2,
        ])

        feature_vector = np.mean(
            combined,
            axis=1,
        )

        _, _, emotion = parse_filename(file_path)

        features.append(feature_vector)
        labels.append(emotion)

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels),
    )


# ============================================================
# FUZZY-LNN MODEL
# ============================================================

def build_ltc_model(num_classes):
    wiring = wirings.AutoNCP(
        128,
        num_classes,
    )

    model = keras.models.Sequential([
        keras.layers.InputLayer(
            shape=(None, num_classes)
        ),

        LTC(
            wiring,
            return_sequences=False,
        ),

        Dense(
            32,
            activation="relu",
        ),

        Dense(
            num_classes,
            activation="softmax",
        ),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def distances_to_centers(data_pca, centers):
    """Squared Euclidean distance from each sample to each FCM center."""
    return np.column_stack([
        np.sum(
            (data_pca - center) ** 2,
            axis=1,
        )
        for center in centers
    ])


# ============================================================
# ONE FOLD
# ============================================================

def run_fold(
    fold_info,
    audio_files,
):
    fold_number = fold_info["fold"]

    print("\n" + "=" * 70)
    print(f"FOLD {fold_number}")
    print("=" * 70)

    train_paths = select_paths(
        audio_files,
        fold_info["train_speakers"],
        fold_info["train_utterances"],
    )

    test_paths = select_paths(
        audio_files,
        fold_info["test_speakers"],
        fold_info["test_utterances"],
    )

    test_paths_by_emotion = group_test_paths_by_emotion(
        test_paths
    )

    print("Training samples:", len(train_paths))
    print("Testing samples :", len(test_paths))
    print(
        "Speaker overlap:",
        set(fold_info["train_speakers"])
        & set(fold_info["test_speakers"]),
    )
    print(
        "Utterance overlap:",
        set(fold_info["train_utterances"])
        & set(fold_info["test_utterances"]),
    )

    # --------------------------------------------------------
    # TRAINING FEATURES
    # --------------------------------------------------------

    print("\nExtracting training features...")

    X_train, y_train = extract_mfcc_features(
        train_paths
    )

    # --------------------------------------------------------
    # LABEL ENCODING
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(
        y_train
    )

    num_classes = len(label_encoder.classes_)

    if num_classes != len(EMOTIONS):
        raise RuntimeError(
            f"Expected {len(EMOTIONS)} training classes, "
            f"but found {num_classes}: {label_encoder.classes_}"
        )

    # --------------------------------------------------------
    # STANDARDIZATION — TRAIN ONLY
    # --------------------------------------------------------

    X_train_2d = X_train.reshape(
        X_train.shape[0],
        -1,
    )

    scaler = StandardScaler()

    scaled_train = scaler.fit_transform(
        X_train_2d
    )

    # --------------------------------------------------------
    # PCA — TRAIN ONLY
    # --------------------------------------------------------

    pca = PCA(
        n_components=PCA_COMPONENTS
    )

    data_pca_train = pca.fit_transform(
        scaled_train
    )

    # --------------------------------------------------------
    # FCM — TRAIN ONLY
    # --------------------------------------------------------

    fcm_model = FCM(
        n_clusters=num_classes,
        max_iter=FCM_MAX_ITER,
        random_state=SEED,
    )

    fcm_model.fit(
        data_pca_train
    )

    fcm_train_labels = fcm_model.predict(
        data_pca_train
    )

    fcm_train_hot = to_categorical(
        fcm_train_labels,
        num_classes=num_classes,
    )

    # --------------------------------------------------------
    # DISTANCE REPRESENTATION
    # --------------------------------------------------------

    train_distances = distances_to_centers(
        data_pca_train,
        fcm_model.centers,
    )

    X_train_distances = np.expand_dims(
        train_distances,
        axis=1,
    )

    # --------------------------------------------------------
    # LTC MODEL
    # --------------------------------------------------------

    tf.keras.backend.clear_session()
    set_seed(SEED)

    model = build_ltc_model(
        num_classes
    )

    print("\nTraining Fuzzy-LNN...")

    start_train = time.perf_counter()

    model.fit(
        X_train_distances,
        fcm_train_hot,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.20,
        verbose=1,
    )

    training_time = (
        time.perf_counter()
        - start_train
    )

    print(
        f"Training time: {training_time:.3f} s"
    )

    # --------------------------------------------------------
    # TESTING
    # --------------------------------------------------------

    y_true_combined = []
    y_pred_combined = []

    emotion_results = []

    total_inference_time = 0.0
    total_test_samples = 0

    # Preserve the notebook's reporting order.
    reporting_order = [
        "NEU",
        "ANG",
        "SAD",
        "HAP",
        "FEA",
        "DIS",
    ]

    for emotion_name in reporting_order:
        emotion_paths = test_paths_by_emotion[
            emotion_name
        ]

        if not emotion_paths:
            continue

        X_test, y_test_text = extract_mfcc_features(
            emotion_paths
        )

        scaled_test = scaler.transform(
            X_test.reshape(X_test.shape[0], -1)
        )

        data_pca_test = pca.transform(
            scaled_test
        )

        test_distances_2d = distances_to_centers(
            data_pca_test,
            fcm_model.centers,
        )

        test_distances = np.expand_dims(
            test_distances_2d,
            axis=1,
        )

       evaluation_labels = fcm_model.predict(
            data_pca_test
        )
        
        evaluation_labels_hot = to_categorical(
              evaluation_labels, num_classes=len(emotions)
         )


        test_loss, test_acc = model.evaluate(
            test_distances,
            evaluation_labels_hot,
            verbose=0,
        )


        warmup_n = min(
            10,
            len(test_distances),
        )

        _ = model.predict(
            test_distances[:warmup_n],
            verbose=0,
        )

        start_inference = time.perf_counter()

        y_prob = model.predict(
            test_distances,
            verbose=0,
        )

        inference_time = (
            time.perf_counter()
            - start_inference
        )

        y_pred = np.argmax(
            y_prob,
            axis=1,
        )

        n_samples = len(evaluation_labels)

        latency_ms = (
            inference_time
            / n_samples
            * 1000
        )

        accuracy = accuracy_score(
            evaluation_labels,
            y_pred,
        )

        f1_weighted = f1_score(
            evaluation_labels,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        precision_weighted = precision_score(
            evaluation_labels,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        recall_weighted = recall_score(
            evaluation_labels,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        balanced_acc = balanced_accuracy_score(
            evaluation_labels,
            y_pred,
        )

        mcc = matthews_corrcoef(
            evaluation_labels,
            y_pred,
        )

        y_true_combined.extend(
            evaluation_labels.tolist()
        )

        y_pred_combined.extend(
            y_pred.tolist()
        )

        total_inference_time += inference_time
        total_test_samples += n_samples

        emotion_results.append({
            "Emotion": emotion_name,
            "Samples": n_samples,
            "Accuracy (%)": accuracy * 100,
            "F1 weighted": f1_weighted,
            "Precision weighted": precision_weighted,
            "Recall weighted": recall_weighted,
            "Balanced Accuracy": balanced_acc,
            "MCC": mcc,
            "Inference Time (s)": inference_time,
            "Latency/sample (ms)": latency_ms,
            "Evaluate accuracy vs emotion labels (%)": test_acc * 100,
            "Evaluate loss vs emotion labels": test_loss,
        })

    # These are calculated at runtime only.
    results_df = pd.DataFrame(
        emotion_results
    )

    y_true_combined = np.asarray(
        y_true_combined
    )

    y_pred_combined = np.asarray(
        y_pred_combined
    )

    overall_accuracy = accuracy_score(
        y_true_combined,
        y_pred_combined,
    )

    overall_f1 = f1_score(
        y_true_combined,
        y_pred_combined,
        average="weighted",
        zero_division=0,
    )

    overall_precision = precision_score(
        y_true_combined,
        y_pred_combined,
        average="weighted",
        zero_division=0,
    )

    overall_recall = recall_score(
        y_true_combined,
        y_pred_combined,
        average="weighted",
        zero_division=0,
    )

    overall_balanced_accuracy = balanced_accuracy_score(
        y_true_combined,
        y_pred_combined,
    )

    overall_mcc = matthews_corrcoef(
        y_true_combined,
        y_pred_combined,
    )

    overall_latency_ms = (
        total_inference_time
        / total_test_samples
        * 1000
    )

    print("\nPer-emotion metrics generated by this run:")
    print(results_df.to_string(index=False))

    print("\nOverall metrics generated by this run:")
    print(f"Accuracy:          {overall_accuracy * 100:.3f}%")
    print(f"Precision:         {overall_precision * 100:.3f}%")
    print(f"Recall:            {overall_recall * 100:.3f}%")
    print(f"F1-score:          {overall_f1 * 100:.3f}%")
    print(f"Balanced accuracy: {overall_balanced_accuracy * 100:.3f}%")
    print(f"MCC:               {overall_mcc:.4f}")
    print(f"Inference time:    {total_inference_time:.4f} s")
    print(f"Latency/sample:    {overall_latency_ms:.4f} ms")

    return {
        "Fold": fold_number,
        "Accuracy (%)": overall_accuracy * 100,
        "Precision weighted (%)": overall_precision * 100,
        "Recall weighted (%)": overall_recall * 100,
        "F1 weighted (%)": overall_f1 * 100,
        "Balanced Accuracy (%)": overall_balanced_accuracy * 100,
        "MCC": overall_mcc,
        "Training Time (s)": training_time,
        "Inference Time (s)": total_inference_time,
        "Latency (ms/sample)": overall_latency_ms,
    }


# ============================================================
# MAIN
# ============================================================

def main(dataset_path):
    set_seed(SEED)

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            "CREMA-D AudioWAV directory not found.\n"
            f"Received: {dataset_path}\n"
            "Set DATASET_PATH in the script or pass "
            "--dataset_path /path/to/AudioWAV"
        )

    audio_files = list_valid_audio_files(
        dataset_path
    )

    speakers, utterances = get_speakers_and_utterances(
        audio_files
    )

    print("=" * 70)
    print("CREMA-D FUZZY-LNN")
    print("=" * 70)
    print("Audio files:", len(audio_files))
    print("Speakers:", len(speakers))
    print("Utterance IDs:", len(utterances))
    print("Emotions:", EMOTIONS)

    splits = create_splits(
        audio_files
    )

    fold_metrics = []

    for fold_info in splits:
        fold_metrics.append(
            run_fold(
                fold_info,
                audio_files,
            )
        )

    # Results below are produced by the current run.
  
    fold_df = pd.DataFrame(
        fold_metrics
    )

    print("\n" + "=" * 70)
    print("FIVE-FOLD METRICS GENERATED BY THIS RUN")
    print("=" * 70)
    print(fold_df.to_string(index=False))

    metric_columns = [
        "Accuracy (%)",
        "Precision weighted (%)",
        "Recall weighted (%)",
        "F1 weighted (%)",
        "Balanced Accuracy (%)",
        "MCC",
        "Training Time (s)",
        "Inference Time (s)",
        "Latency (ms/sample)",
    ]

    print("\nMEAN ± SAMPLE SD ACROSS FIVE FOLDS")

    for column in metric_columns:
        mean_value = fold_df[column].mean()
        std_value = fold_df[column].std(ddof=1)

        print(
            f"{column}: "
            f"{mean_value:.4f} ± {std_value:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run the CREMA-D Fuzzy-LNN experiment."
        )
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DATASET_PATH,
        help=(
            "Path to the CREMA-D AudioWAV directory."
        ),
    )

    args = parser.parse_args()

    main(
        os.path.abspath(
            os.path.expanduser(
                args.dataset_path
            )
        )
    )
