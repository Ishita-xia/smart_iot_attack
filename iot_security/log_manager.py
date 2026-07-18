"""
Log Manager Module
==================
Centralized logging system for all IDS events.
Provides structured log entries, searching, filtering, and export.
"""

import os
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from enum import Enum
from collections import Counter


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventType(Enum):
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_STOP = "SYSTEM_STOP"
    TRAFFIC_CAPTURE = "TRAFFIC_CAPTURE"
    ATTACK_DETECTED = "ATTACK_DETECTED"
    ANOMALY_DETECTED = "ANOMALY_DETECTED"
    TRAFFIC_BLOCKED = "TRAFFIC_BLOCKED"
    TRAFFIC_ALLOWED = "TRAFFIC_ALLOWED"
    RATE_LIMITED = "RATE_LIMITED"
    CONNECTION_TERMINATED = "CONNECTION_TERMINATED"
    MODEL_TRAINED = "MODEL_TRAINED"
    FL_ROUND_COMPLETE = "FL_ROUND_COMPLETE"
    ALERT_GENERATED = "ALERT_GENERATED"
    THRESHOLD_UPDATED = "THRESHOLD_UPDATED"
    DEVICE_CONNECTED = "DEVICE_CONNECTED"
    DEVICE_DISCONNECTED = "DEVICE_DISCONNECTED"
    PIPELINE_RUN = "PIPELINE_RUN"
    PIPELINE_START = "PIPELINE_START"
    PIPELINE_COMPLETE = "PIPELINE_COMPLETE"
    THREAT_INTEL_COMPLETE = "THREAT_INTEL_COMPLETE"


@dataclass
class LogEntry:
    timestamp: str
    level: str
    event_type: str
    source: str
    message: str
    details: Dict = field(default_factory=dict)
    src_ip: str = ""
    dst_ip: str = ""
    flow_id: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


class LogManager:
    """
    Centralized logging system for the Smart IDS.
    Stores logs in-memory with optional file export.
    """

    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self.logs: List[LogEntry] = []
        self._counters = Counter()

        # Auto-log system start
        self.log(LogLevel.INFO, EventType.SYSTEM_START, "System",
                 "Smart IDS System initialized")

    def log(self, level: LogLevel, event_type: EventType, source: str,
            message: str, details: Dict = None, src_ip: str = "",
            dst_ip: str = "", flow_id: str = ""):
        """Add a new log entry."""
        entry = LogEntry(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            level=level.value,
            event_type=event_type.value,
            source=source,
            message=message,
            details=details or {},
            src_ip=src_ip,
            dst_ip=dst_ip,
            flow_id=flow_id,
        )
        self.logs.append(entry)
        self._counters[level.value] += 1
        self._counters[event_type.value] += 1

        # Trim if over limit
        if len(self.logs) > self.max_entries:
            self.logs = self.logs[-self.max_entries:]

    # ── Convenience methods ──

    def log_attack(self, attack_type: str, confidence: float,
                   src_ip: str = "", flow_id: str = ""):
        severity = LogLevel.CRITICAL if confidence > 0.9 else LogLevel.WARNING
        self.log(
            severity, EventType.ATTACK_DETECTED, "Detection Engine",
            f"Attack detected: {attack_type} (confidence: {confidence:.2f})",
            details={"attack_type": attack_type, "confidence": confidence},
            src_ip=src_ip, flow_id=flow_id,
        )

    def log_anomaly(self, anomaly_score: float, threshold: float,
                    src_ip: str = "", flow_id: str = ""):
        self.log(
            LogLevel.WARNING, EventType.ANOMALY_DETECTED, "Autoencoder",
            f"Anomaly detected (score={anomaly_score:.4f}, threshold={threshold:.4f})",
            details={"anomaly_score": anomaly_score, "threshold": threshold},
            src_ip=src_ip, flow_id=flow_id,
        )

    def log_filter_action(self, action: str, reason: str,
                          src_ip: str = "", flow_id: str = ""):
        event_map = {
            "block": EventType.TRAFFIC_BLOCKED,
            "allow": EventType.TRAFFIC_ALLOWED,
            "rate_limit": EventType.RATE_LIMITED,
            "terminate": EventType.CONNECTION_TERMINATED,
        }
        level_map = {
            "block": LogLevel.WARNING,
            "allow": LogLevel.INFO,
            "rate_limit": LogLevel.WARNING,
            "terminate": LogLevel.CRITICAL,
        }
        self.log(
            level_map.get(action, LogLevel.INFO),
            event_map.get(action, EventType.TRAFFIC_ALLOWED),
            "Traffic Filter",
            f"Action: {action.upper()} — {reason}",
            src_ip=src_ip, flow_id=flow_id,
        )

    def log_model_event(self, model_name: str, message: str, metrics: Dict = None):
        self.log(
            LogLevel.INFO, EventType.MODEL_TRAINED, model_name,
            message, details=metrics or {},
        )

    def log_fl_round(self, round_num: int, accuracy: float, num_clients: int):
        self.log(
            LogLevel.INFO, EventType.FL_ROUND_COMPLETE, "Federated Learning",
            f"FL Round {round_num} complete — Global Acc: {accuracy:.4f}",
            details={"round": round_num, "accuracy": accuracy, "clients": num_clients},
        )

    # ── Querying ──

    def get_logs(self, level: str = None, event_type: str = None,
                 source: str = None, limit: int = 100) -> List[Dict]:
        """Get filtered log entries as list of dicts."""
        filtered = self.logs
        if level:
            filtered = [l for l in filtered if l.level == level]
        if event_type:
            filtered = [l for l in filtered if l.event_type == event_type]
        if source:
            filtered = [l for l in filtered if source.lower() in l.source.lower()]
        return [l.to_dict() for l in filtered[-limit:]]

    def get_recent(self, n: int = 50) -> List[Dict]:
        """Get the N most recent log entries."""
        return [l.to_dict() for l in self.logs[-n:]]

    def get_stats(self) -> Dict:
        """Get log statistics."""
        return {
            "total_entries": len(self.logs),
            "by_level": {
                "DEBUG": self._counters.get("DEBUG", 0),
                "INFO": self._counters.get("INFO", 0),
                "WARNING": self._counters.get("WARNING", 0),
                "ERROR": self._counters.get("ERROR", 0),
                "CRITICAL": self._counters.get("CRITICAL", 0),
            },
            "by_event": {
                "attacks": self._counters.get("ATTACK_DETECTED", 0),
                "anomalies": self._counters.get("ANOMALY_DETECTED", 0),
                "blocked": self._counters.get("TRAFFIC_BLOCKED", 0),
                "allowed": self._counters.get("TRAFFIC_ALLOWED", 0),
                "rate_limited": self._counters.get("RATE_LIMITED", 0),
                "terminated": self._counters.get("CONNECTION_TERMINATED", 0),
            },
        }

    def get_threat_timeline(self, last_n: int = 200) -> List[Dict]:
        """Get timeline of threat events only."""
        threat_types = {
            EventType.ATTACK_DETECTED.value,
            EventType.ANOMALY_DETECTED.value,
            EventType.TRAFFIC_BLOCKED.value,
            EventType.CONNECTION_TERMINATED.value,
        }
        threats = [l for l in self.logs if l.event_type in threat_types]
        return [
            {
                "time": l.timestamp,
                "type": l.event_type,
                "level": l.level,
                "message": l.message,
                "src_ip": l.src_ip,
            }
            for l in threats[-last_n:]
        ]

    def export_json(self, filepath: str):
        """Export all logs to a JSON file."""
        with open(filepath, "w") as f:
            json.dump([l.to_dict() for l in self.logs], f, indent=2)

    def export_csv(self, filepath: str):
        """Export all logs to a CSV file."""
        import csv
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Level", "Event Type", "Source", "Message", "Src IP", "Dst IP", "Flow ID"])
            for l in self.logs:
                writer.writerow([
                    l.timestamp, l.level, l.event_type, l.source, l.message, l.src_ip, l.dst_ip, l.flow_id
                ])

    def get_attack_summary(self) -> Dict:
        """Return aggregate summary of detected attacks."""
        attack_logs = [l for l in self.logs if l.event_type == EventType.ATTACK_DETECTED.value]
        types = Counter()
        ips = Counter()
        for al in attack_logs:
            atk_type = al.details.get("attack_type", "Unknown")
            types[atk_type] += 1
            if al.src_ip:
                ips[al.src_ip] += 1
        return {
            "total_attack_logs": len(attack_logs),
            "by_type": dict(types.most_common(5)),
            "by_ip": dict(ips.most_common(5))
        }

    def clear(self):
        """Clear all logs."""
        self.logs.clear()
        self._counters.clear()
