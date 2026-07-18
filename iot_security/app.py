"""
Smart IDS — Flask API Backend
=============================
Replaces the Streamlit dashboard with a robust Flask JSON API.
Orchestrates the 9 flowchart stages using IDSPipeline, and provides
endpoints for model training, attack testing, federated learning,
logs, alerts, threat intelligence, and report exports.
"""

import os
import sys
import time
import json
import pickle
import threading
import traceback
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, request, send_file, render_template
from datetime import datetime, timedelta

# Path setup
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Module imports
from data_loader import get_train_test, FEATURE_COLS, ATTACK_GROUPS
from models.cnn_model import CNN1D, build_loaders, train_cnn, evaluate_cnn
from models.autoencoder import Autoencoder, train_autoencoder, compute_threshold, detect_anomalies
from models.federated_learning import run_federated_simulation
from models.incremental_learning import incremental_train, EWC
from attacks.poison_attack import label_flip_attack, detect_poison_isolation_forest
from attacks.adversarial_attack import fgsm_attack, pgd_attack
from qwen_explainer import QwenExplainer
from traffic_simulator import TrafficCapture, IoTEnvironment, create_iot_network
from traffic_filter import TrafficFilter, FilterAction
from log_manager import LogManager, LogLevel, EventType
from alert_system import AlertManager, AlertSeverity, AlertPreferences
from threat_intelligence import ThreatIntelligence
from pipeline import IDSPipeline

DEVICE = "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# ═══════════════════════════════════════════════════════════════════════
#  System Global State
# ═══════════════════════════════════════════════════════════════════════

system_state = {
    # Dataset
    "loaded": False,
    "max_rows": 1000,
    "X_tr": None,
    "X_te": None,
    "y_tr": None,
    "y_te": None,
    "le": None,
    "scaler": None,
    "te_loader": None,

    # Models
    "cnn_model": None,
    "cnn_trained": False,
    "cnn_acc": 0.0,
    "cnn_report": None,
    "cnn_cm": None,

    "ae_model": None,
    "ae_trained": False,
    "ae_threshold": 0.5,
    "ae_detection_rate": 0.0,

    # Singletons
    "log_manager": LogManager(),
    "alert_manager": AlertManager(),
    "traffic_filter": TrafficFilter(),
    "threat_intel": ThreatIntelligence(),
    "explainer": QwenExplainer(),
    
    # Execution History
    "pipeline_history": []
}

# Link log and alert manager to threat intel
system_state["traffic_filter"].anomaly_threshold = system_state["ae_threshold"]

# Training progress states
training_state = {
    "cnn": {
        "status": "idle",  # "idle", "running", "completed", "error"
        "epoch": 0,
        "total_epochs": 0,
        "loss": [],
        "accuracy": [],
        "error_msg": ""
    },
    "ae": {
        "status": "idle",  # "idle", "running", "completed", "error"
        "epoch": 0,
        "total_epochs": 0,
        "loss": [],
        "error_msg": ""
    }
}

# Instantiate Pipeline
pipeline_orchestrator = IDSPipeline(
    log_manager=system_state["log_manager"],
    alert_manager=system_state["alert_manager"],
    traffic_filter=system_state["traffic_filter"],
    threat_intelligence=system_state["threat_intel"],
)

# ═══════════════════════════════════════════════════════════════════════
#  Auto-load Pre-trained Models (if available)
# ═══════════════════════════════════════════════════════════════════════

SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

def load_pretrained_models():
    """Load pre-trained models from saved_models/ directory if they exist."""
    global system_state
    meta_path = os.path.join(SAVED_MODELS_DIR, "training_meta.json")
    if not os.path.exists(meta_path):
        print("[INFO] No pre-trained models found. Train via UI or run train_and_save.py")
        return False

    print("[INFO] Loading pre-trained models...")
    try:
        # Load metadata
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Load scaler
        with open(os.path.join(SAVED_MODELS_DIR, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)

        # Load label encoder
        with open(os.path.join(SAVED_MODELS_DIR, "label_encoder.pkl"), "rb") as f:
            le = pickle.load(f)

        # Load CNN model
        num_features = meta["num_features"]
        num_classes = meta["num_classes"]
        cnn_model = CNN1D(num_features=num_features, num_classes=num_classes)
        cnn_model.load_state_dict(torch.load(
            os.path.join(SAVED_MODELS_DIR, "cnn_model.pth"),
            map_location=DEVICE, weights_only=True
        ))
        cnn_model.eval()

        # Load Autoencoder
        ae_model = Autoencoder(input_dim=num_features)
        ae_model.load_state_dict(torch.load(
            os.path.join(SAVED_MODELS_DIR, "autoencoder.pth"),
            map_location=DEVICE, weights_only=True
        ))
        ae_model.eval()

        # Load AE threshold
        with open(os.path.join(SAVED_MODELS_DIR, "ae_threshold.pkl"), "rb") as f:
            ae_threshold = pickle.load(f)

        # Update system state
        system_state["scaler"] = scaler
        system_state["le"] = le
        system_state["cnn_model"] = cnn_model
        system_state["cnn_trained"] = True
        system_state["cnn_acc"] = meta.get("cnn_test_accuracy", 0.0)
        system_state["cnn_report"] = meta.get("classification_report", None)
        system_state["ae_model"] = ae_model
        system_state["ae_trained"] = True
        system_state["ae_threshold"] = float(ae_threshold)
        system_state["ae_detection_rate"] = meta.get("ae_attack_detection_rate", 0.0)
        system_state["loaded"] = True

        # Update traffic filter with models
        system_state["traffic_filter"].cnn_model = cnn_model
        system_state["traffic_filter"].label_encoder = le
        system_state["traffic_filter"].scaler = scaler
        system_state["traffic_filter"].autoencoder = ae_model
        system_state["traffic_filter"].anomaly_threshold = float(ae_threshold)

        # Update pipeline orchestrator
        pipeline_orchestrator.scaler = scaler
        pipeline_orchestrator.label_encoder = le

        # Store metadata for /api/model_info
        system_state["training_meta"] = meta

        print(f"  [OK] CNN loaded (accuracy: {meta.get('cnn_test_accuracy', 0)*100:.2f}%)")
        print(f"  [OK] Autoencoder loaded (threshold: {ae_threshold:.6f})")
        print(f"  [OK] Scaler & LabelEncoder loaded ({num_classes} classes)")
        print("[INFO] Pre-trained models loaded successfully!")

        system_state["log_manager"].log(
            LogLevel.INFO, EventType.SYSTEM_START, "Model Loader",
            f"Pre-trained models loaded - CNN acc: {meta.get('cnn_test_accuracy', 0)*100:.2f}%, "
            f"{num_classes} classes, AE threshold: {ae_threshold:.6f}"
        )
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load pre-trained models: {e}")
        import traceback
        traceback.print_exc()
        return False

# Try to auto-load on startup
load_pretrained_models()

# ═══════════════════════════════════════════════════════════════════════
#  Helper Background Tasks
# ═══════════════════════════════════════════════════════════════════════

def bg_train_cnn(epochs, max_rows):
    global training_state, system_state
    try:
        training_state["cnn"]["status"] = "running"
        training_state["cnn"]["epoch"] = 0
        training_state["cnn"]["total_epochs"] = epochs
        training_state["cnn"]["loss"] = []
        training_state["cnn"]["accuracy"] = []
        training_state["cnn"]["error_msg"] = ""

        # Make sure dataset is loaded
        if not system_state["loaded"]:
            X_tr, X_te, y_tr, y_te, le, sc = get_train_test(max_rows_per_class=max_rows)
            system_state.update(dict(
                X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
                le=le, scaler=sc, loaded=True
            ))
            pipeline_orchestrator.scaler = sc
            pipeline_orchestrator.label_encoder = le

        X_tr, X_te = system_state["X_tr"], system_state["X_te"]
        y_tr, y_te = system_state["y_tr"], system_state["y_te"]
        le = system_state["le"]

        model = CNN1D(num_features=len(FEATURE_COLS), num_classes=len(le.classes_))
        tr_loader, te_loader = build_loaders(X_tr, y_tr, X_te, y_te)

        def progress_cb(ep, loss, acc):
            training_state["cnn"]["epoch"] = ep
            training_state["cnn"]["loss"].append(float(loss))
            training_state["cnn"]["accuracy"].append(float(acc))

        train_cnn(model, tr_loader, DEVICE, epochs=epochs, progress_callback=progress_cb)

        acc, report, cm = evaluate_cnn(model, te_loader, DEVICE, list(le.classes_))
        
        # Save to state
        system_state["cnn_model"] = model
        system_state["cnn_trained"] = True
        system_state["cnn_acc"] = float(acc)
        system_state["cnn_report"] = report
        # Confusion matrix needs to be JSON serializable
        system_state["cnn_cm"] = cm.tolist() if isinstance(cm, np.ndarray) else cm
        system_state["te_loader"] = te_loader

        # Update filter
        system_state["traffic_filter"].cnn_model = model
        system_state["traffic_filter"].label_encoder = le
        system_state["traffic_filter"].scaler = system_state["scaler"]

        system_state["log_manager"].log_model_event(
            "CNN", f"CNN trained — Test Accuracy: {acc*100:.2f}%",
            {"accuracy": acc, "epochs": epochs}
        )

        training_state["cnn"]["status"] = "completed"
    except Exception as e:
        traceback.print_exc()
        training_state["cnn"]["status"] = "error"
        training_state["cnn"]["error_msg"] = str(e)


def bg_train_ae(epochs, max_rows):
    global training_state, system_state
    try:
        training_state["ae"]["status"] = "running"
        training_state["ae"]["epoch"] = 0
        training_state["ae"]["total_epochs"] = epochs
        training_state["ae"]["loss"] = []
        training_state["ae"]["error_msg"] = ""

        # Make sure dataset is loaded
        if not system_state["loaded"]:
            X_tr, X_te, y_tr, y_te, le, sc = get_train_test(max_rows_per_class=max_rows)
            system_state.update(dict(
                X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
                le=le, scaler=sc, loaded=True
            ))
            pipeline_orchestrator.scaler = sc
            pipeline_orchestrator.label_encoder = le

        X_tr, X_te = system_state["X_tr"], system_state["X_te"]
        y_tr, y_te = system_state["y_tr"], system_state["y_te"]
        le = system_state["le"]

        benign_idx = np.where(y_tr == list(le.classes_).index("Benign_Final"))[0]
        X_benign = X_tr[benign_idx]

        ae = Autoencoder(input_dim=len(FEATURE_COLS))

        def progress_cb(ep, loss):
            training_state["ae"]["epoch"] = ep
            training_state["ae"]["loss"].append(float(loss))

        train_autoencoder(ae, X_benign, DEVICE, epochs=epochs, progress_callback=progress_cb)

        thresh, benign_err = compute_threshold(ae, X_benign, DEVICE)

        # Test anomaly detection
        attack_idx = np.where(y_te != list(le.classes_).index("Benign_Final"))[0][:500]
        X_att = X_te[attack_idx]
        flagged, att_err = detect_anomalies(ae, X_att, thresh, DEVICE)

        # Save to state
        system_state["ae_model"] = ae
        system_state["ae_trained"] = True
        system_state["ae_threshold"] = float(thresh)
        system_state["ae_detection_rate"] = float(flagged.mean())

        # Update filter
        system_state["traffic_filter"].autoencoder = ae
        system_state["traffic_filter"].anomaly_threshold = float(thresh)

        system_state["log_manager"].log_model_event(
            "Autoencoder", f"AE trained — Threshold: {thresh:.6f}, Detection Rate: {flagged.mean()*100:.1f}%",
            {"threshold": float(thresh), "detection_rate": float(flagged.mean())}
        )

        training_state["ae"]["status"] = "completed"
    except Exception as e:
        traceback.print_exc()
        training_state["ae"]["status"] = "error"
        training_state["ae"]["error_msg"] = str(e)


# ═══════════════════════════════════════════════════════════════════════
#  Flask Routes
# ═══════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/status")
def get_status():
    stats = {
        "dataset_loaded": system_state["loaded"],
        "cnn_trained": system_state["cnn_trained"],
        "cnn_accuracy": system_state["cnn_acc"],
        "ae_trained": system_state["ae_trained"],
        "ae_threshold": system_state["ae_threshold"],
        "ae_detection_rate": system_state["ae_detection_rate"],
        "total_alerts": len(system_state["alert_manager"].alerts),
        "active_alerts": sum(1 for a in system_state["alert_manager"].alerts if a.status == "ACTIVE"),
        "total_logs": len(system_state["log_manager"].logs),
        "blocked_ips_count": len(system_state["threat_intel"].get_blocked_ips()),
        "devices": [
            {"id": "SH-CAM-001", "type": "Security Camera", "ip": "192.168.1.45", "status": "online"},
            {"id": "SH-THRM-001", "type": "Smart Thermostat", "ip": "192.168.1.12", "status": "online"},
            {"id": "SH-LOCK-001", "type": "Smart Lock", "ip": "192.168.1.88", "status": "online"},
            {"id": "HC-MON-001", "type": "Patient Monitor", "ip": "192.168.1.201", "status": "online"},
            {"id": "IND-PLC-001", "type": "PLC Controller", "ip": "192.168.1.130", "status": "online"}
        ]
    }
    return jsonify(stats)


@app.route("/api/load_dataset", methods=["POST"])
def load_dataset_endpoint():
    try:
        data = request.json or {}
        max_rows = int(data.get("max_rows", 1000))
        system_state["max_rows"] = max_rows
        
        X_tr, X_te, y_tr, y_te, le, sc = get_train_test(max_rows_per_class=max_rows)
        system_state.update(dict(
            X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
            le=le, scaler=sc, loaded=True
        ))
        
        # Link pipeline scaler and label encoder
        pipeline_orchestrator.scaler = sc
        pipeline_orchestrator.label_encoder = le
        system_state["traffic_filter"].scaler = sc
        system_state["traffic_filter"].label_encoder = le

        system_state["log_manager"].log(
            LogLevel.INFO, EventType.SYSTEM_START, "Data Loader",
            f"Dataset loaded: {len(X_tr)} train, {len(X_te)} test, {len(le.classes_)} classes"
        )
        return jsonify({
            "status": "success",
            "message": f"Dataset loaded: {len(le.classes_)} classes | {len(X_tr)} train | {len(X_te)} test samples"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/train_cnn", methods=["POST"])
def train_cnn_endpoint():
    if training_state["cnn"]["status"] == "running":
        return jsonify({"status": "error", "message": "CNN training is already running"}), 400

    data = request.json or {}
    epochs = int(data.get("epochs", 5))
    max_rows = int(data.get("max_rows", system_state["max_rows"]))

    t = threading.Thread(target=bg_train_cnn, args=(epochs, max_rows))
    t.start()
    return jsonify({"status": "success", "message": "CNN training started in background"})


@app.route("/api/train_ae", methods=["POST"])
def train_ae_endpoint():
    if training_state["ae"]["status"] == "running":
        return jsonify({"status": "error", "message": "Autoencoder training is already running"}), 400

    data = request.json or {}
    epochs = int(data.get("epochs", 10))
    max_rows = int(data.get("max_rows", system_state["max_rows"]))

    t = threading.Thread(target=bg_train_ae, args=(epochs, max_rows))
    t.start()
    return jsonify({"status": "success", "message": "Autoencoder training started in background"})


@app.route("/api/training_status")
def get_training_status():
    return jsonify(training_state)


@app.route("/api/run_pipeline", methods=["POST"])
def run_pipeline_endpoint():
    if not system_state["loaded"]:
        return jsonify({"status": "error", "message": "Dataset must be loaded first"}), 400

    data = request.json or {}
    env_str = data.get("environment", "Smart Home")
    batch_size = int(data.get("batch_size", 50))

    # Match environment enum
    env = None
    for e in IoTEnvironment:
        if e.value == env_str:
            env = e
            break

    if env is None:
        env = IoTEnvironment.SMART_HOME

    # Execute
    res = pipeline_orchestrator.run_pipeline(
        environment=env,
        batch_size=batch_size,
        cache_rows=system_state["max_rows"]
    )

    # Convert dataclass to dict
    res_dict = {
        "timestamp": res.timestamp,
        "environment": res.environment,
        "total_flows": res.total_flows,
        "allowed_count": res.allowed_count,
        "blocked_count": res.blocked_count,
        "rate_limited_count": res.rate_limited_count,
        "terminated_count": res.terminated_count,
        "anomaly_count": res.anomaly_count,
        "threats_detected": res.threats_detected,
        "stages_status": res.stages_status,
        "decisions": res.decisions[:30], # limit return values size
        "threat_intel_summary": res.threat_intel_summary,
        "execution_time_ms": res.execution_time_ms
    }

    # Save execution history
    system_state["pipeline_history"].append(res_dict)
    if len(system_state["pipeline_history"]) > 20:
        system_state["pipeline_history"] = system_state["pipeline_history"][-20:]

    return jsonify(res_dict)


@app.route("/api/pipeline_history")
def get_pipeline_history():
    return jsonify(system_state["pipeline_history"])


@app.route("/api/logs")
def get_logs_endpoint():
    level = request.args.get("level")
    event_type = request.args.get("event_type")
    source = request.args.get("source")
    limit = int(request.args.get("limit", 100))

    logs = system_state["log_manager"].get_logs(
        level=level,
        event_type=event_type,
        source=source,
        limit=limit
    )
    return jsonify(logs)


@app.route("/api/logs/stats")
def get_logs_stats():
    return jsonify(system_state["log_manager"].get_stats())


@app.route("/api/logs/export/json")
def export_logs_json():
    filepath = os.path.join(BASE_DIR, "logs_export.json")
    system_state["log_manager"].export_json(filepath)
    return send_file(filepath, as_attachment=True, download_name="smart_ids_logs.json")


@app.route("/api/logs/export/csv")
def export_logs_csv():
    filepath = os.path.join(BASE_DIR, "logs_export.csv")
    system_state["log_manager"].export_csv(filepath)
    return send_file(filepath, as_attachment=True, download_name="smart_ids_logs.csv")


@app.route("/api/alerts")
def get_alerts_endpoint():
    status = request.args.get("status") # ACTIVE, ACKNOWLEDGED, RESOLVED
    severity = request.args.get("severity")

    alerts = system_state["alert_manager"].alerts
    if status:
        alerts = [a for a in alerts if a.status == status]
    if severity:
        alerts = [a for a in alerts if a.severity == severity]

    alerts_dict = [a.to_dict() for a in reversed(alerts[-100:])]
    return jsonify(alerts_dict)


@app.route("/api/alerts/stats")
def get_alerts_stats():
    return jsonify(system_state["alert_manager"].get_stats())


@app.route("/api/alerts/timeline")
def get_alerts_timeline_endpoint():
    return jsonify(system_state["alert_manager"].get_alert_timeline())


@app.route("/api/alerts/acknowledge", methods=["POST"])
def acknowledge_alerts():
    data = request.json or {}
    alert_id = data.get("alert_id")

    if alert_id == "all":
        system_state["alert_manager"].acknowledge_all()
        return jsonify({"status": "success", "message": "All alerts acknowledged"})
    elif alert_id:
        system_state["alert_manager"].acknowledge_alert(alert_id)
        return jsonify({"status": "success", "message": f"Alert {alert_id} acknowledged"})
    return jsonify({"status": "error", "message": "Missing alert_id"}), 400


@app.route("/api/alerts/resolve", methods=["POST"])
def resolve_alerts():
    data = request.json or {}
    alert_id = data.get("alert_id")

    if alert_id == "all":
        system_state["alert_manager"].resolve_all()
        return jsonify({"status": "success", "message": "All alerts resolved"})
    elif alert_id:
        system_state["alert_manager"].resolve_alert(alert_id)
        return jsonify({"status": "success", "message": f"Alert {alert_id} resolved"})
    return jsonify({"status": "error", "message": "Missing alert_id"}), 400


@app.route("/api/alerts/preferences", methods=["POST", "GET"])
def alert_preferences():
    am = system_state["alert_manager"]
    if request.method == "POST":
        data = request.json or {}
        am.preferences.email_enabled = bool(data.get("email_enabled", am.preferences.email_enabled))
        am.preferences.sms_enabled = bool(data.get("sms_enabled", am.preferences.sms_enabled))
        am.preferences.email_threshold = data.get("email_threshold", am.preferences.email_threshold)
        am.preferences.sms_threshold = data.get("sms_threshold", am.preferences.sms_threshold)
        return jsonify({"status": "success", "message": "Preferences updated", "preferences": {
            "email_enabled": am.preferences.email_enabled,
            "sms_enabled": am.preferences.sms_enabled,
            "email_threshold": am.preferences.email_threshold,
            "sms_threshold": am.preferences.sms_threshold
        }})
    else:
        return jsonify({
            "email_enabled": am.preferences.email_enabled,
            "sms_enabled": am.preferences.sms_enabled,
            "email_threshold": am.preferences.email_threshold,
            "sms_threshold": am.preferences.sms_threshold
        })


@app.route("/api/alerts/export/csv")
def export_alerts_csv_endpoint():
    filepath = os.path.join(BASE_DIR, "alerts_export.csv")
    system_state["alert_manager"].export_alerts_csv(filepath)
    return send_file(filepath, as_attachment=True, download_name="smart_ids_alerts.csv")


@app.route("/api/threat_intel")
def get_threat_intel():
    ti = system_state["threat_intel"]
    return jsonify({
        "summary": ti.get_threat_summary(),
        "reputations": ti.get_all_reputations(sort_by="score", limit=30),
        "iocs": ti.extract_iocs(),
        "patterns": ti.correlate_patterns([]) # run check
    })


@app.route("/api/explain/<attack_class>")
def explain_attack(attack_class):
    explanation = system_state["explainer"].explain(attack_class, 0.95)
    info = system_state["explainer"].get_attack_info(attack_class)
    return jsonify({
        "attack_class": attack_class,
        "narrative": explanation,
        "details": info
    })


@app.route("/api/run_federated", methods=["POST"])
def run_federated_endpoint():
    if not system_state["loaded"]:
        return jsonify({"status": "error", "message": "Dataset must be loaded first"}), 400

    data = request.json or {}
    rounds = int(data.get("rounds", 3))
    clients = int(data.get("clients", 4))

    try:
        # Run federated simulation
        results = run_federated_simulation(
            rounds=rounds,
            num_clients=clients,
            epochs_per_round=1,
            max_rows_per_class=system_state["max_rows"],
            device=DEVICE
        )
        
        system_state["log_manager"].log_fl_round(
            round_num=rounds,
            accuracy=results["global_accuracies"][-1],
            num_clients=clients
        )

        return jsonify({
            "status": "success",
            "rounds": rounds,
            "clients": clients,
            "accuracies": results["global_accuracies"],
            "client_metrics": results.get("client_accuracies", [])
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/run_adversarial", methods=["POST"])
def run_adversarial_endpoint():
    if not system_state["cnn_trained"]:
        return jsonify({"status": "error", "message": "CNN model must be trained first"}), 400

    data = request.json or {}
    method = data.get("method", "FGSM")
    epsilon = float(data.get("epsilon", 0.1))

    try:
        model = system_state["cnn_model"]
        X_te = system_state["X_te"][:200]
        y_te = system_state["y_te"][:200]

        if method == "FGSM":
            _, orig_acc, adv_acc = fgsm_attack(model, X_te, y_te, epsilon, DEVICE)
        else:
            _, orig_acc, adv_acc = pgd_attack(model, X_te, y_te, epsilon, 0.01, 10, DEVICE)

        return jsonify({
            "status": "success",
            "method": method,
            "epsilon": epsilon,
            "original_accuracy": float(orig_acc),
            "adversarial_accuracy": float(adv_acc),
            "drop": float(orig_acc - adv_acc)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/run_poison", methods=["POST"])
def run_poison_endpoint():
    if not system_state["loaded"]:
        return jsonify({"status": "error", "message": "Dataset must be loaded first"}), 400

    data = request.json or {}
    src_class = data.get("src_class", "Benign_Final")
    tgt_class = data.get("tgt_class", "Mirai-udpplain")
    poison_rate = float(data.get("poison_rate", 0.2))

    try:
        X_tr = system_state["X_tr"]
        y_tr = system_state["y_tr"]
        classes = list(system_state["le"].classes_)

        src_idx = classes.index(src_class)
        tgt_idx = classes.index(tgt_class)

        _, y_poisoned, poison_mask = label_flip_attack(
            X_tr, y_tr, src_idx, tgt_idx, poison_rate
        )
        
        detected = detect_poison_isolation_forest(X_tr, contamination=poison_rate)
        tp = int((poison_mask & detected).sum())
        total_poisoned = int(poison_mask.sum())
        detection_rate = tp / (total_poisoned + 1e-9)

        return jsonify({
            "status": "success",
            "src_class": src_class,
            "tgt_class": tgt_class,
            "poison_rate": poison_rate,
            "total_poisoned": total_poisoned,
            "detected_poisoned": tp,
            "detection_rate": float(detection_rate)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/attack_groups")
def get_attack_groups():
    return jsonify(ATTACK_GROUPS)


@app.route("/api/attack_classes")
def get_attack_classes():
    if system_state["le"]:
        return jsonify(list(system_state["le"].classes_))
    # Return default categories/classes
    return jsonify(list(ATTACK_GROUPS.keys()))


@app.route("/api/model_info")
def get_model_info():
    """Return metadata about the pre-trained models."""
    meta = system_state.get("training_meta", None)
    if meta:
        return jsonify({
            "pretrained": True,
            "trained_at": meta.get("trained_at"),
            "cnn_test_accuracy": meta.get("cnn_test_accuracy"),
            "ae_attack_detection_rate": meta.get("ae_attack_detection_rate"),
            "ae_threshold": meta.get("ae_threshold"),
            "num_classes": meta.get("num_classes"),
            "class_names": meta.get("class_names"),
            "train_samples": meta.get("train_samples"),
            "test_samples": meta.get("test_samples"),
            "num_features": meta.get("num_features"),
            "cnn_epochs": meta.get("cnn_epochs"),
            "ae_epochs": meta.get("ae_epochs"),
            "cnn_training_loss": meta.get("cnn_training_loss"),
            "cnn_training_acc": meta.get("cnn_training_acc"),
            "ae_training_loss": meta.get("ae_training_loss"),
            "classification_report": meta.get("classification_report")
        })
    return jsonify({
        "pretrained": False,
        "cnn_trained": system_state["cnn_trained"],
        "ae_trained": system_state["ae_trained"],
        "cnn_accuracy": system_state["cnn_acc"]
    })


@app.route("/api/upload_test_data", methods=["POST"])
def upload_test_data():
    """Accept a CSV file, format it as traffic flows, run it through the end-to-end 9-stage IDSPipeline, and update the dashboard."""
    if not system_state["cnn_trained"]:
        return jsonify({"status": "error", "message": "CNN model is not trained yet. Please train models first or run train_and_save.py"}), 400

    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded. Send a CSV file with key 'file'."}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"status": "error", "message": "Only .csv files are accepted."}), 400

    try:
        df = pd.read_csv(file, low_memory=False)
        total_rows = len(df)
        if total_rows == 0:
            return jsonify({"status": "error", "message": "Uploaded CSV file is empty."}), 400

        # Limit large files to 300 rows for real-time responsiveness
        if total_rows > 300:
            df = df.iloc[:300]
            total_rows = 300

        # Align columns with training features
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0
        X_raw = df[FEATURE_COLS].copy()
        X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_raw.fillna(0, inplace=True)

        # Scale and extract features
        # Devices metadata mapping
        devices_list = [
            ("SH-CAM-001", "Security Camera", "192.168.1.45"),
            ("SH-THRM-001", "Smart Thermostat", "192.168.1.12"),
            ("SH-LOCK-001", "Smart Lock", "192.168.1.88"),
            ("HC-MON-001", "Patient Monitor", "192.168.1.201"),
            ("IND-PLC-001", "PLC Controller", "192.168.1.130")
        ]

        def guess_protocol(row) -> str:
            if row.get("TCP", 0) > 0: return "TCP"
            if row.get("UDP", 0) > 0: return "UDP"
            if row.get("ICMP", 0) > 0: return "ICMP"
            if row.get("HTTP", 0) > 0: return "HTTP"
            if row.get("DNS", 0) > 0: return "DNS"
            return "OTHER"

        custom_flows = []
        import random
        for i, (_, row) in enumerate(X_raw.iterrows()):
            dev_id, dev_type, src_ip = random.choice(devices_list)
            custom_flows.append({
                "flow_id": f"FL-UP-{i+1:05d}",
                "timestamp": datetime.now() - timedelta(seconds=random.uniform(0, 10)),
                "device_id": dev_id,
                "device_type": dev_type,
                "src_ip": src_ip,
                "dst_ip": f"10.0.0.{random.randint(10, 254)}",
                "label": "Unknown",
                "features": row.values.astype(np.float32),
                "protocol": guess_protocol(row),
            })

        # Process custom flows through the IDSPipeline
        pipeline_res = pipeline_orchestrator.run_pipeline(
            environment=IoTEnvironment.SMART_HOME,
            custom_flows=custom_flows
        )

        res_dict = {
            "timestamp": pipeline_res.timestamp,
            "environment": "CSV Upload (" + file.filename + ")",
            "total_flows": pipeline_res.total_flows,
            "allowed_count": pipeline_res.allowed_count,
            "blocked_count": pipeline_res.blocked_count,
            "rate_limited_count": pipeline_res.rate_limited_count,
            "terminated_count": pipeline_res.terminated_count,
            "anomaly_count": pipeline_res.anomaly_count,
            "threats_detected": pipeline_res.threats_detected,
            "stages_status": pipeline_res.stages_status,
            "decisions": pipeline_res.decisions[:30],  # Keep size lightweight
            "threat_intel_summary": pipeline_res.threat_intel_summary,
            "execution_time_ms": pipeline_res.execution_time_ms
        }

        # Save to pipeline history
        system_state["pipeline_history"].append(res_dict)
        if len(system_state["pipeline_history"]) > 20:
            system_state["pipeline_history"] = system_state["pipeline_history"][-20:]

        # Create statistics response metrics for frontend backward compatibility
        pred_labels = [d["cnn_pred"] for d in pipeline_res.decisions]
        from collections import Counter
        distribution = dict(Counter(pred_labels))
        distribution = dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))

        group_distribution = {}
        for label, count in distribution.items():
            group = ATTACK_GROUPS.get(label, "Unknown")
            group_distribution[group] = group_distribution.get(group, 0) + count
        group_distribution = dict(sorted(group_distribution.items(), key=lambda x: x[1], reverse=True))

        sample_results = []
        for i in range(min(100, len(pipeline_res.decisions))):
            d = pipeline_res.decisions[i]
            sample_results.append({
                "row": i + 1,
                "predicted_class": d["cnn_pred"],
                "attack_group": d["action"],
                "is_attack": d["action"] != "ALLOW"
            })

        return jsonify({
            "status": "success",
            "total_rows": pipeline_res.total_flows,
            "benign_count": pipeline_res.allowed_count,
            "attack_count": pipeline_res.total_flows - pipeline_res.allowed_count,
            "anomaly_count": pipeline_res.anomaly_count,
            "anomaly_rate": round((pipeline_res.anomaly_count / pipeline_res.total_flows) * 100, 2) if pipeline_res.total_flows > 0 else 0,
            "class_distribution": distribution,
            "group_distribution": group_distribution,
            "sample_results": sample_results,
            "stages_status": pipeline_res.stages_status,
            "decisions": pipeline_res.decisions
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    # In production/deployment, Flask runs on 0.0.0.0:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
