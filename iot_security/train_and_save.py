"""
Train & Save Models
====================
This script trains the CNN classifier and Autoencoder anomaly detector
on the FULL provided dataset, then saves:
  - CNN model weights        -> saved_models/cnn_model.pth
  - Autoencoder weights      -> saved_models/autoencoder.pth
  - StandardScaler           -> saved_models/scaler.pkl
  - LabelEncoder             -> saved_models/label_encoder.pkl
  - Autoencoder threshold    -> saved_models/ae_threshold.pkl
  - Training metadata        -> saved_models/training_meta.json

After running this script once, the Flask app will automatically load
these pre-trained models so that anyone can use the system without
needing to re-train.

Usage:
    cd iot_security
    python train_and_save.py
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import torch

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from data_loader import get_train_test, FEATURE_COLS, ATTACK_GROUPS
from models.cnn_model import CNN1D, build_loaders, train_cnn, evaluate_cnn
from models.autoencoder import Autoencoder, train_autoencoder, compute_threshold, detect_anomalies

DEVICE = "cpu"
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # â”€â”€â”€ Configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    MAX_ROWS_PER_CLASS = 5000   # Use up to 5000 rows per attack class
    CNN_EPOCHS         = 10     # Number of training epochs for CNN
    AE_EPOCHS          = 20     # Number of training epochs for Autoencoder
    TEST_SIZE          = 0.2    # 80% train, 20% test
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    print("=" * 70)
    print("  SMART IDS â€” MODEL TRAINING & SAVING")
    print("=" * 70)

    # â”€â”€ Step 1: Load Dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[1/6] Loading dataset...")
    t0 = time.time()
    X_tr, X_te, y_tr, y_te, le, scaler = get_train_test(
        max_rows_per_class=MAX_ROWS_PER_CLASS,
        test_size=TEST_SIZE
    )
    print(f"  âœ“ Dataset loaded in {time.time()-t0:.1f}s")
    print(f"    Training samples : {len(X_tr)}")
    print(f"    Testing samples  : {len(X_te)}")
    print(f"    Features         : {X_tr.shape[1]}")
    print(f"    Classes          : {len(le.classes_)}")
    print(f"    Class names      : {list(le.classes_)}")

    # â”€â”€ Step 2: Train CNN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n[2/6] Training 1D-CNN classifier ({CNN_EPOCHS} epochs)...")
    num_features = len(FEATURE_COLS)
    num_classes = len(le.classes_)

    cnn_model = CNN1D(num_features=num_features, num_classes=num_classes)
    tr_loader, te_loader = build_loaders(X_tr, y_tr, X_te, y_te)

    t0 = time.time()
    history = train_cnn(cnn_model, tr_loader, DEVICE, epochs=CNN_EPOCHS)
    cnn_train_time = time.time() - t0
    print(f"  âœ“ CNN trained in {cnn_train_time:.1f}s")

    # â”€â”€ Step 3: Evaluate CNN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[3/6] Evaluating CNN on test set...")
    acc, report, cm = evaluate_cnn(cnn_model, te_loader, DEVICE, list(le.classes_))
    print(f"  âœ“ CNN Test Accuracy: {acc*100:.2f}%")
    print(f"\n  Classification Report:")
    # Print per-class metrics
    print(f"  {'Class':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print(f"  {'-'*70}")
    for cls in le.classes_:
        if cls in report:
            r = report[cls]
            print(f"  {cls:<30} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {r['support']:>10.0f}")
    print(f"  {'-'*70}")
    print(f"  {'Overall Accuracy':<30} {acc:>10.4f}")

    # â”€â”€ Step 4: Train Autoencoder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n[4/6] Training Autoencoder for anomaly detection ({AE_EPOCHS} epochs)...")
    benign_label = list(le.classes_).index("Benign_Final")
    benign_idx = np.where(y_tr == benign_label)[0]
    X_benign = X_tr[benign_idx]
    print(f"  Using {len(X_benign)} benign samples for AE training")

    ae_model = Autoencoder(input_dim=num_features)
    t0 = time.time()
    ae_history = train_autoencoder(ae_model, X_benign, DEVICE, epochs=AE_EPOCHS)
    ae_train_time = time.time() - t0
    print(f"  âœ“ Autoencoder trained in {ae_train_time:.1f}s")

    # Compute threshold
    threshold, benign_errors = compute_threshold(ae_model, X_benign, DEVICE)
    print(f"  âœ“ Anomaly threshold (95th percentile): {threshold:.6f}")

    # Test anomaly detection on attack samples
    attack_idx = np.where(y_te != benign_label)[0][:1000]
    X_attack_test = X_te[attack_idx]
    flagged, _ = detect_anomalies(ae_model, X_attack_test, threshold, DEVICE)
    detection_rate = flagged.mean()
    print(f"  âœ“ Attack detection rate: {detection_rate*100:.1f}%")

    # â”€â”€ Step 5: Save Everything â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[5/6] Saving trained models and preprocessing artifacts...")

    # CNN weights
    cnn_path = os.path.join(SAVE_DIR, "cnn_model.pth")
    torch.save(cnn_model.state_dict(), cnn_path)
    print(f"  âœ“ CNN weights       -> {cnn_path}")

    # Autoencoder weights
    ae_path = os.path.join(SAVE_DIR, "autoencoder.pth")
    torch.save(ae_model.state_dict(), ae_path)
    print(f"  âœ“ Autoencoder       -> {ae_path}")

    # Scaler
    scaler_path = os.path.join(SAVE_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  âœ“ Scaler            -> {scaler_path}")

    # Label Encoder
    le_path = os.path.join(SAVE_DIR, "label_encoder.pkl")
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    print(f"  âœ“ Label Encoder     -> {le_path}")

    # Threshold
    thresh_path = os.path.join(SAVE_DIR, "ae_threshold.pkl")
    with open(thresh_path, "wb") as f:
        pickle.dump(threshold, f)
    print(f"  âœ“ AE Threshold      -> {thresh_path}")

    # Training metadata
    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_rows_per_class": MAX_ROWS_PER_CLASS,
        "cnn_epochs": CNN_EPOCHS,
        "ae_epochs": AE_EPOCHS,
        "test_size": TEST_SIZE,
        "train_samples": int(len(X_tr)),
        "test_samples": int(len(X_te)),
        "num_features": num_features,
        "num_classes": num_classes,
        "class_names": list(le.classes_),
        "cnn_test_accuracy": float(acc),
        "cnn_train_time_sec": round(cnn_train_time, 1),
        "ae_train_time_sec": round(ae_train_time, 1),
        "ae_threshold": float(threshold),
        "ae_attack_detection_rate": float(detection_rate),
        "cnn_training_loss": [float(l) for l in history["loss"]],
        "cnn_training_acc": [float(a) for a in history["acc"]],
        "ae_training_loss": [float(l) for l in ae_history],
        "classification_report": report,
    }
    meta_path = os.path.join(SAVE_DIR, "training_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  âœ“ Training metadata -> {meta_path}")

    # â”€â”€ Step 6: Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE â€” SUMMARY")
    print("=" * 70)
    print(f"  CNN Test Accuracy      : {acc*100:.2f}%")
    print(f"  AE Detection Rate      : {detection_rate*100:.1f}%")
    print(f"  AE Anomaly Threshold   : {threshold:.6f}")
    print(f"  Total Training Samples : {len(X_tr)}")
    print(f"  Total Testing Samples  : {len(X_te)}")
    print(f"  Number of Classes      : {num_classes}")
    print(f"  Models saved to        : {SAVE_DIR}")
    print("=" * 70)
    print("\n  The Flask app will now auto-load these pre-trained models on startup.")
    print("  Users can upload their own CSV data for evaluation without re-training.\n")


if __name__ == "__main__":
    main()

