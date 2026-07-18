"""
main.py — Quick CLI trainer (optional, dashboard is the main interface)
Run: python main.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'iot_security'))

import torch
from data_loader import get_train_test, FEATURE_COLS
from models.cnn_model import CNN1D, build_loaders, train_cnn, evaluate_cnn
from models.autoencoder import Autoencoder, train_autoencoder, compute_threshold

DEVICE = "cpu"

def main():
    print("="*55)
    print("  IoT Intelligent Security System — Quick Train")
    print("="*55)

    print("\n[1/4] Loading dataset (1000 rows/class)…")
    X_tr, X_te, y_tr, y_te, le, sc = get_train_test(max_rows_per_class=1000)
    n_cls = len(le.classes_)
    print(f"      Classes: {n_cls}  Train: {len(X_tr)}  Test: {len(X_te)}")

    print("\n[2/4] Training CNN…")
    model = CNN1D(num_features=len(FEATURE_COLS), num_classes=n_cls)
    tr_loader, te_loader = build_loaders(X_tr, y_tr, X_te, y_te)
    train_cnn(model, tr_loader, DEVICE, epochs=5)
    acc, _, _ = evaluate_cnn(model, te_loader, DEVICE, list(le.classes_))
    print(f"      CNN Test Accuracy: {acc*100:.2f}%")

    print("\n[3/4] Training Autoencoder (benign only)…")
    import numpy as np
    benign_i = list(le.classes_).index("Benign_Final")
    X_ben = X_tr[y_tr == benign_i]
    ae = Autoencoder(input_dim=len(FEATURE_COLS))
    train_autoencoder(ae, X_ben, DEVICE, epochs=5)
    thresh, _ = compute_threshold(ae, X_ben, DEVICE)
    print(f"      AE Threshold: {thresh:.6f}")

    print("\n[4/4] Done! Launch dashboard with:")
    print("      python iot_security/app.py")
    print("="*55)

if __name__ == "__main__":
    main()
