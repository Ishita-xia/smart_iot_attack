"""
Traffic Filtering & Response Engine
=====================================
Multi-model decision engine that combines CNN predictions + Autoencoder anomaly
scores to classify traffic as ALLOW / BLOCK / RATE_LIMIT / TERMINATE.
"""

import numpy as np
import torch
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


class FilterAction(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    RATE_LIMIT = "rate_limit"
    TERMINATE = "terminate"


@dataclass
class FilterDecision:
    flow_id: str
    timestamp: datetime
    action: FilterAction
    cnn_prediction: str
    cnn_confidence: float
    anomaly_score: float
    is_anomaly: bool
    reason: str
    src_ip: str = ""
    dst_ip: str = ""


class TrafficFilter:
    """
    Combines CNN classifier + Autoencoder anomaly detector for multi-model
    traffic filtering decisions.
    """

    def __init__(
        self,
        cnn_model=None,
        autoencoder=None,
        label_encoder=None,
        scaler=None,
        anomaly_threshold: float = 0.5,
        confidence_threshold: float = 0.6,
        rate_limit_window: int = 60,
        rate_limit_max: int = 100,
    ):
        self.cnn_model = cnn_model
        self.autoencoder = autoencoder
        self.label_encoder = label_encoder
        self.scaler = scaler
        self.anomaly_threshold = anomaly_threshold
        self.confidence_threshold = confidence_threshold
        self.rate_limit_window = rate_limit_window
        self.rate_limit_max = rate_limit_max

        # Statistics
        self.stats = {
            "total": 0,
            "allowed": 0,
            "blocked": 0,
            "rate_limited": 0,
            "terminated": 0,
        }
        self.decision_history: List[FilterDecision] = []
        self.ip_request_counts: Dict[str, List[datetime]] = defaultdict(list)

        # Severity mapping for attack types
        self.attack_severity = {
            "Benign_Final": 0,
            "Recon-PingSweep": 2, "Recon-PortScan": 2,
            "Recon-HostDiscovery": 2, "Recon-OSScan": 2,
            "VulnerabilityScan": 3,
            "DictionaryBruteForce": 4,
            "BrowserHijacking": 4, "XSS": 4, "SqlInjection": 5,
            "CommandInjection": 5, "Uploading_Attack": 5,
            "DNS_Spoofing": 5, "MITM-ArpSpoofing": 6,
            "DoS-HTTP_Flood": 6, "DoS-SYN_Flood": 6,
            "DoS-TCP_Flood": 6, "DoS-UDP_Flood": 6,
            "DDoS-ACK_Fragmentation": 7, "DDoS-HTTP_Flood": 7,
            "DDoS-ICMP_Flood": 7, "DDoS-ICMP_Fragmentation": 7,
            "DDoS-PSHACK_FLOOD": 7, "DDoS-RSTFINFLOOD": 7,
            "DDoS-SYN_Flood": 7, "DDoS-SlowLoris": 7,
            "DDoS-SynonymousIP_Flood": 7, "DDoS-TCP_Flood": 7,
            "DDoS-UDP_Flood": 7, "DDoS-UDP_Fragmentation": 7,
            "Backdoor_Malware": 8,
            "Mirai-greeth_flood": 8, "Mirai-greip_flood": 8,
            "Mirai-udpplain": 8,
        }

    def set_models(self, cnn_model, autoencoder, label_encoder, scaler, anomaly_threshold):
        """Set or update the ML models used for filtering."""
        self.cnn_model = cnn_model
        self.autoencoder = autoencoder
        self.label_encoder = label_encoder
        self.scaler = scaler
        self.anomaly_threshold = anomaly_threshold

    def _check_rate_limit(self, src_ip: str) -> bool:
        """Returns True if the IP has exceeded the rate limit."""
        now = datetime.now()
        cutoff = now.timestamp() - self.rate_limit_window
        self.ip_request_counts[src_ip] = [
            t for t in self.ip_request_counts[src_ip] if t.timestamp() > cutoff
        ]
        self.ip_request_counts[src_ip].append(now)
        return len(self.ip_request_counts[src_ip]) > self.rate_limit_max

    def filter_flow(self, features: np.ndarray, flow_id: str = "",
                    src_ip: str = "", dst_ip: str = "") -> FilterDecision:
        """
        Make a multi-model filtering decision for a single traffic flow.
        """
        self.stats["total"] += 1
        timestamp = datetime.now()

        # ── CNN prediction ──
        cnn_prediction = "Unknown"
        cnn_confidence = 0.0
        if self.cnn_model is not None and self.label_encoder is not None:
            try:
                self.cnn_model.eval()
                x_tensor = torch.tensor(features.reshape(1, -1), dtype=torch.float32)
                with torch.no_grad():
                    logits = self.cnn_model(x_tensor)
                    probs = torch.softmax(logits, dim=1)
                    pred_idx = probs.argmax(1).item()
                    cnn_confidence = probs[0, pred_idx].item()
                    cnn_prediction = self.label_encoder.inverse_transform([pred_idx])[0]
            except Exception:
                pass

        # ── Autoencoder anomaly score ──
        anomaly_score = 0.0
        is_anomaly = False
        if self.autoencoder is not None:
            try:
                self.autoencoder.eval()
                x_tensor = torch.tensor(features.reshape(1, -1), dtype=torch.float32)
                with torch.no_grad():
                    anomaly_score = self.autoencoder.reconstruction_error(x_tensor).item()
                is_anomaly = anomaly_score > self.anomaly_threshold
            except Exception:
                pass

        # ── Multi-model decision logic ──
        action, reason = self._decide(
            cnn_prediction, cnn_confidence, is_anomaly, anomaly_score, src_ip
        )

        # Update stats
        action_key = action.value.replace(" ", "_")
        if action_key in self.stats:
            self.stats[action_key] += 1

        decision = FilterDecision(
            flow_id=flow_id,
            timestamp=timestamp,
            action=action,
            cnn_prediction=cnn_prediction,
            cnn_confidence=cnn_confidence,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            reason=reason,
            src_ip=src_ip,
            dst_ip=dst_ip,
        )
        self.decision_history.append(decision)

        # Keep only last 5000 decisions in memory
        if len(self.decision_history) > 5000:
            self.decision_history = self.decision_history[-5000:]

        return decision

    def _decide(self, cnn_pred: str, cnn_conf: float,
                is_anomaly: bool, anomaly_score: float,
                src_ip: str) -> Tuple[FilterAction, str]:
        """Core multi-model decision logic."""

        severity = self.attack_severity.get(cnn_pred, 5)

        # Rule 1: Benign with high confidence and no anomaly → ALLOW
        if cnn_pred == "Benign_Final" and cnn_conf > self.confidence_threshold and not is_anomaly:
            return FilterAction.ALLOW, "Legitimate traffic (CNN + AE agree)"

        # Rule 2: High-severity attack with high confidence → TERMINATE
        if severity >= 7 and cnn_conf > 0.7:
            return FilterAction.TERMINATE, f"Critical attack: {cnn_pred} (conf={cnn_conf:.2f})"

        # Rule 3: Malware / backdoor → BLOCK immediately
        if severity >= 8:
            return FilterAction.BLOCK, f"Malware detected: {cnn_pred}"

        # Rule 4: Rate limit check
        if self._check_rate_limit(src_ip):
            return FilterAction.RATE_LIMIT, f"Rate limit exceeded for {src_ip}"

        # Rule 5: DDoS/DoS attacks → BLOCK
        if "DDoS" in cnn_pred or "DoS" in cnn_pred:
            return FilterAction.BLOCK, f"Flood attack blocked: {cnn_pred}"

        # Rule 6: Anomaly detected but CNN says benign → RATE_LIMIT (suspicious)
        if is_anomaly and cnn_pred == "Benign_Final":
            return FilterAction.RATE_LIMIT, f"Anomaly detected (score={anomaly_score:.4f}), possible zero-day"

        # Rule 7: Known attack with moderate confidence → BLOCK
        if cnn_pred != "Benign_Final" and cnn_conf > 0.5:
            return FilterAction.BLOCK, f"Attack detected: {cnn_pred} (conf={cnn_conf:.2f})"

        # Rule 8: Low confidence → ALLOW but log
        if cnn_pred == "Benign_Final":
            return FilterAction.ALLOW, "Likely benign (low confidence)"

        # Default: block unknown
        return FilterAction.BLOCK, f"Suspicious traffic: {cnn_pred}"

    def get_stats(self) -> Dict:
        """Return current filtering statistics."""
        total = max(self.stats["total"], 1)
        return {
            **self.stats,
            "block_rate": round((self.stats["blocked"] + self.stats["terminated"]) / total * 100, 2),
            "allow_rate": round(self.stats["allowed"] / total * 100, 2),
        }

    def get_recent_decisions(self, n: int = 50) -> List[Dict]:
        """Return the most recent N decisions as dicts."""
        recent = self.decision_history[-n:]
        return [
            {
                "flow_id": d.flow_id,
                "time": d.timestamp.strftime("%H:%M:%S"),
                "action": d.action.value.upper(),
                "prediction": d.cnn_prediction,
                "confidence": f"{d.cnn_confidence:.2f}",
                "anomaly": f"{d.anomaly_score:.4f}",
                "is_anomaly": "⚠️" if d.is_anomaly else "✓",
                "reason": d.reason,
                "src_ip": d.src_ip,
            }
            for d in reversed(recent)
        ]

    def reset_stats(self):
        """Reset all filtering statistics."""
        self.stats = {k: 0 for k in self.stats}
        self.decision_history.clear()
        self.ip_request_counts.clear()
