# 🛡️ Smart IDS — Intrusion Detection System for IoT Networks

> A machine learning-based Intrusion Detection System that detects and blocks **34 types of cyberattacks** on IoT networks in real time using a **1D-CNN classifier** and **Autoencoder anomaly detector**, served through a full web dashboard.

---

## 📌 Problem Statement

IoT devices (smart cameras, thermostats, medical monitors, industrial controllers) are everywhere but have **no built-in security**. They run 24/7, are rarely updated, and are easy targets for hackers.

Traditional firewalls use **fixed rules** — they only catch known attacks. New attack types (zero-day attacks) bypass them completely.

**We need a smarter solution that learns from data.**

---

## ✅ Our Solution

Two machine learning models working together:

| Model | Type | Purpose |
|---|---|---|
| **1D-CNN Classifier** | Supervised | Classifies 34 known attack types |
| **Autoencoder** | Unsupervised | Detects unknown / zero-day attacks |

The CNN identifies **what** the attack is. The Autoencoder catches **anything that looks abnormal** — even attacks never seen before.

---

## 🎯 Key Features

- 📤 **Upload your own CSV** — analyzed through the full 9-stage pipeline instantly
- 🧠 **Pre-trained models** — no setup needed, works out of the box
- 🔍 **34 attack classes** detected — DDoS, DoS, Malware, Recon, Web Attacks, Spoofing, Brute Force
- ⚠️ **Zero-day detection** via Autoencoder anomaly scoring
- 🌐 **Federated Learning** — privacy-preserving multi-node training simulation
- 🛡️ **Adversarial robustness testing** — FGSM and PGD attack simulation
- 📊 **Real-time dashboard** — alerts, logs, threat intelligence, IP reputation
- 📁 **Export** logs and alerts to CSV/JSON

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| CNN Test Accuracy | **69.35%** (34-class classification) |
| Autoencoder Detection Rate | **77.4%** |
| Training Samples | 128,462 |
| Test Samples | 32,116 |
| Attack Classes | 34 |

> Random guessing baseline = 2.94%. Our model is **23× better than random**.

**Top performing classes:**

| Class | F1-Score |
|---|---|
| DDoS-ICMP_Flood | 100% |
| DDoS-PSHACK_Flood | 100% |
| DDoS-RSTFINFLOOD | 100% |
| DDoS-UDP_Fragmentation | 99.9% |
| Mirai-udpplain | 99.6% |
| DDoS-ICMP_Fragmentation | 98.9% |

---

## 🏗️ System Architecture

```
User Uploads CSV
      ↓
Stage 1 → IoT Network Simulation
Stage 2 → Traffic Capture
Stage 3 → Preprocessing & Normalization (StandardScaler)
Stage 4 → Detection Engine (CNN + Autoencoder)
Stage 5 → Federated Learning (FedAvg)
Stage 6 → Traffic Filtering (ALLOW / BLOCK / RATE_LIMIT / TERMINATE)
Stage 7 → Log Analysis & Threat Intelligence
Stage 8 → Alerts & Notifications
Stage 9 → Admin Dashboard
```

---

## 📁 Project Structure

```
├── main.py                          # Quick CLI trainer
├── SMART_IDS_COMPLETE_GUIDE.txt     # Full project documentation
│
└── iot_security/
    ├── app.py                       # Flask server + all API routes
    ├── data_loader.py               # Dataset loading & preprocessing
    ├── pipeline.py                  # 9-stage pipeline orchestrator
    ├── traffic_simulator.py         # IoT environment simulation
    ├── traffic_filter.py            # ALLOW/BLOCK decision engine
    ├── alert_system.py              # Alert management
    ├── log_manager.py               # Event logging
    ├── threat_intelligence.py       # IP reputation + IOC engine
    ├── qwen_explainer.py            # Attack narrative generator
    ├── train_and_save.py            # Offline training script
    │
    ├── models/
    │   ├── cnn_model.py             # 1D-CNN architecture
    │   ├── autoencoder.py           # Autoencoder architecture
    │   ├── federated_learning.py    # FedAvg simulation
    │   └── incremental_learning.py  # EWC continual learning
    │
    ├── attacks/
    │   ├── adversarial_attack.py    # FGSM + PGD attacks
    │   └── poison_attack.py         # Label flip + Isolation Forest
    │
    ├── saved_models/
    │   ├── cnn_model.pth            # Trained CNN weights
    │   ├── autoencoder.pth          # Trained Autoencoder weights
    │   ├── scaler.pkl               # Fitted StandardScaler
    │   ├── label_encoder.pkl        # LabelEncoder (34 classes)
    │   ├── ae_threshold.pkl         # Anomaly threshold = 0.0671
    │   └── training_meta.json       # Accuracy metrics & history
    │
    ├── templates/
    │   └── dashboard.html           # Single page web app
    │
    └── static/
        ├── css/style.css            # Glassmorphism dark theme
        └── js/dashboard.js          # Frontend logic & charts
```

---

## 🚀 How to Run

**1. Install dependencies**
```bash
pip install flask torch scikit-learn pandas numpy
```

**2. Launch the dashboard**
```bash
cd iot_security
python app.py
```

**3. Open in browser**
```
http://127.0.0.1:5000
```

Pre-trained models load automatically on startup — no training needed.

---

## 🔁 How to Retrain Models (Optional)

```bash
cd iot_security
python train_and_save.py
```

This reads all dataset CSVs, trains CNN + Autoencoder, and saves all model files to `saved_models/`. Takes ~2-3 minutes on CPU.

---

## 📤 Using the Upload Feature

1. Open the dashboard at `http://127.0.0.1:5000`
2. The **Upload panel is the first thing you see** at the top
3. Drag & drop any CSV with IoT network flow features
4. Click **"Analyze Traffic"**
5. Results appear instantly:
   - Total flows / Benign / Attacks / Anomalies
   - Attack group breakdown with progress bars
   - Per-row predictions
   - Full pipeline stage status animation

---

## 🗃️ Dataset

**CIC-IoT-2023** — Canadian Institute for Cybersecurity

| Category | Attack Types |
|---|---|
| DDoS | ACK Frag, HTTP Flood, ICMP Flood, SYN Flood, UDP Flood, SlowLoris + 6 more |
| DoS | HTTP, SYN, TCP, UDP Flood |
| Malware | Backdoor, Mirai (3 variants) |
| Recon | Port Scan, OS Scan, Ping Sweep, Host Discovery, Vuln Scan |
| Web Attacks | SQL Injection, XSS, Command Injection, Browser Hijacking, Upload Attack |
| Spoofing | DNS Spoofing, ARP Spoofing (MITM) |
| Brute Force | Dictionary Brute Force |
| Benign | Normal IoT traffic |

**39 features per flow:** Header length, protocol flags, packet statistics, timing, payload sizes, and more.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Web Backend | Flask (Python) |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Charts | Chart.js |
| Icons | FontAwesome 6 |

---

## 🔬 Research Features

- **Adversarial Testing** — FGSM & PGD evasion attacks on the CNN
- **Poisoning Defense** — Label flip attack + Isolation Forest detection
- **Federated Learning** — FedAvg across 2-8 simulated edge nodes
- **Threat Intelligence** — IP reputation scoring, attack chain correlation
- **Zero-Day Detection** — Autoencoder flags unknown attacks never seen in training

---

## 📄 Documentation

See [`SMART_IDS_COMPLETE_GUIDE.txt`](./SMART_IDS_COMPLETE_GUIDE.txt) for the full project documentation including problem statement, model architecture, pipeline explanation, and interview preparation guide.

---

## 👩‍💻 Author

Final Year Project — IoT Network Security using Machine Learning
