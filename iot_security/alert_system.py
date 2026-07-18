"""
Alert & Notification System
============================
Real-time alert management with severity levels, aggregation,
dashboard notifications, and simulated email/SMS alerts.
"""

from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from collections import Counter, defaultdict


class AlertSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class AlertStatus(Enum):
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


@dataclass
class AlertPreferences:
    email_enabled: bool = True
    sms_enabled: bool = True
    email_threshold: str = "CRITICAL"
    sms_threshold: str = "EMERGENCY"
    auto_acknowledge_minutes: int = 0


@dataclass
class Alert:
    alert_id: str
    timestamp: str
    severity: str
    title: str
    message: str
    source: str
    status: str = "ACTIVE"
    attack_type: str = ""
    src_ip: str = ""
    count: int = 1
    details: Dict = field(default_factory=dict)
    acknowledged_at: str = ""
    resolved_at: str = ""

    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "status": self.status,
            "attack_type": self.attack_type,
            "src_ip": self.src_ip,
            "count": self.count,
            "details": self.details,
        }


class AlertManager:
    """
    Manages real-time alerts for the Smart IDS.
    Features: alert aggregation, severity classification,
    notification queue, and simulated email/SMS.
    """

    def __init__(self, aggregation_window_sec: int = 30):
        self.alerts: List[Alert] = []
        self.notification_queue: List[Dict] = []
        self._counter = 0
        self.aggregation_window = aggregation_window_sec
        self._recent_keys: Dict[str, Alert] = {}
        self.preferences = AlertPreferences()

        # Simulated notification channels
        self.email_log: List[Dict] = []
        self.sms_log: List[Dict] = []
        self.dashboard_notifications: List[Dict] = []

    def _next_id(self) -> str:
        self._counter += 1
        return f"ALT-{self._counter:05d}"


    def create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str = "Detection Engine",
        attack_type: str = "",
        src_ip: str = "",
        details: Dict = None,
    ) -> Alert:
        """Create a new alert or aggregate with an existing similar one."""

        # Aggregation key: combine attack_type + severity + src_ip
        agg_key = f"{attack_type}:{severity.value}:{src_ip}"
        now = datetime.now()

        # Check if we can aggregate
        if agg_key in self._recent_keys:
            existing = self._recent_keys[agg_key]
            existing_time = datetime.strptime(existing.timestamp, "%Y-%m-%d %H:%M:%S")
            if (now - existing_time).total_seconds() < self.aggregation_window:
                existing.count += 1
                existing.message = f"{message} (x{existing.count})"
                return existing

        # Create new alert
        alert = Alert(
            alert_id=self._next_id(),
            timestamp=now.strftime("%Y-%m-%d %H:%M:%S"),
            severity=severity.value,
            title=title,
            message=message,
            source=source,
            attack_type=attack_type,
            src_ip=src_ip,
            details=details or {},
        )
        self.alerts.append(alert)
        self._recent_keys[agg_key] = alert

        # Push to notification channels
        self._push_notifications(alert)

        # Keep only last 2000 alerts
        if len(self.alerts) > 2000:
            self.alerts = self.alerts[-2000:]

        return alert

    def _push_notifications(self, alert: Alert):
        """Push alert to all notification channels."""
        notif = {
            "alert_id": alert.alert_id,
            "time": alert.timestamp,
            "severity": alert.severity,
            "title": alert.title,
            "message": alert.message,
        }

        # Dashboard notification (always)
        self.dashboard_notifications.append(notif)
        if len(self.dashboard_notifications) > 500:
            self.dashboard_notifications = self.dashboard_notifications[-500:]

        severity_levels = {"INFO": 1, "WARNING": 2, "CRITICAL": 3, "EMERGENCY": 4}
        alert_level = severity_levels.get(alert.severity, 1)

        # Simulated Email
        if self.preferences.email_enabled:
            threshold_level = severity_levels.get(self.preferences.email_threshold, 3)
            if alert_level >= threshold_level:
                self.email_log.append({
                    **notif,
                    "to": "admin@iot-security.local",
                    "subject": f"[{alert.severity}] {alert.title}",
                    "sent_at": alert.timestamp,
                    "status": "SIMULATED",
                })

        # Simulated SMS
        if self.preferences.sms_enabled:
            threshold_level = severity_levels.get(self.preferences.sms_threshold, 4)
            if alert_level >= threshold_level:
                self.sms_log.append({
                    **notif,
                    "to": "+91-XXXX-XXXXXX",
                    "sent_at": alert.timestamp,
                    "status": "SIMULATED",
                })

    # ── Convenience alert creators ──

    def alert_attack_detected(self, attack_type: str, confidence: float,
                               src_ip: str = ""):
        """Create an alert for a detected attack."""
        if confidence > 0.9:
            severity = AlertSeverity.CRITICAL
        elif confidence > 0.7:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO

        # Upgrade DDoS / Malware to emergency
        if "DDoS" in attack_type or "Mirai" in attack_type or "Backdoor" in attack_type:
            severity = AlertSeverity.EMERGENCY if confidence > 0.8 else AlertSeverity.CRITICAL

        return self.create_alert(
            severity=severity,
            title=f"Attack Detected: {attack_type}",
            message=f"{attack_type} attack detected from {src_ip} (confidence: {confidence:.2f})",
            source="Detection Engine",
            attack_type=attack_type,
            src_ip=src_ip,
            details={"confidence": confidence},
        )

    def alert_anomaly(self, score: float, threshold: float, src_ip: str = ""):
        """Create alert for anomaly detection."""
        severity = AlertSeverity.CRITICAL if score > threshold * 2 else AlertSeverity.WARNING
        return self.create_alert(
            severity=severity,
            title="Anomaly Detected (Possible Zero-Day)",
            message=f"Anomalous traffic from {src_ip} (score={score:.4f}, threshold={threshold:.4f})",
            source="Autoencoder",
            attack_type="Anomaly",
            src_ip=src_ip,
            details={"anomaly_score": score, "threshold": threshold},
        )

    def alert_rate_limit(self, src_ip: str, request_count: int):
        """Create alert for rate limiting."""
        return self.create_alert(
            severity=AlertSeverity.WARNING,
            title="Rate Limit Triggered",
            message=f"IP {src_ip} exceeded rate limit ({request_count} requests)",
            source="Traffic Filter",
            attack_type="Rate Limit",
            src_ip=src_ip,
        )

    def alert_system(self, title: str, message: str,
                     severity: AlertSeverity = AlertSeverity.INFO):
        """Create a system alert."""
        return self.create_alert(
            severity=severity,
            title=title,
            message=message,
            source="System",
        )

    # ── Querying ──

    def get_active_alerts(self) -> List[Dict]:
        """Get all currently active alerts."""
        return [a.to_dict() for a in self.alerts if a.status == "ACTIVE"]

    def get_alerts_by_severity(self, severity: str) -> List[Dict]:
        return [a.to_dict() for a in self.alerts if a.severity == severity]

    def get_recent_alerts(self, n: int = 50) -> List[Dict]:
        return [a.to_dict() for a in self.alerts[-n:]]

    def get_notifications(self, n: int = 20) -> List[Dict]:
        return self.dashboard_notifications[-n:]

    def acknowledge_alert(self, alert_id: str):
        for a in self.alerts:
            if a.alert_id == alert_id:
                a.status = AlertStatus.ACKNOWLEDGED.value
                a.acknowledged_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                break

    def acknowledge_all(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for a in self.alerts:
            if a.status == AlertStatus.ACTIVE.value:
                a.status = AlertStatus.ACKNOWLEDGED.value
                a.acknowledged_at = now

    def resolve_alert(self, alert_id: str):
        for a in self.alerts:
            if a.alert_id == alert_id:
                a.status = AlertStatus.RESOLVED.value
                a.resolved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                break

    def resolve_all(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for a in self.alerts:
            if a.status in (AlertStatus.ACTIVE.value, AlertStatus.ACKNOWLEDGED.value):
                a.status = AlertStatus.RESOLVED.value
                a.resolved_at = now

    def get_alert_timeline(self) -> List[Dict]:
        """Aggregate alert counts per minute for visualization."""
        timeline = defaultdict(lambda: {"INFO": 0, "WARNING": 0, "CRITICAL": 0, "EMERGENCY": 0})
        for a in self.alerts:
            # Group by minute: YYYY-MM-DD HH:MM
            minute_str = a.timestamp[:16]
            if a.severity in timeline[minute_str]:
                timeline[minute_str][a.severity] += 1
            else:
                timeline[minute_str][a.severity] = 1
        
        sorted_timeline = []
        for time_key in sorted(timeline.keys()):
            sorted_timeline.append({
                "time": time_key,
                "INFO": timeline[time_key].get("INFO", 0),
                "WARNING": timeline[time_key].get("WARNING", 0),
                "CRITICAL": timeline[time_key].get("CRITICAL", 0),
                "EMERGENCY": timeline[time_key].get("EMERGENCY", 0),
            })
        return sorted_timeline

    def export_alerts_csv(self, filepath: str):
        """Export alerts to a CSV file."""
        import csv
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Alert ID", "Timestamp", "Severity", "Title", "Message",
                "Source", "Status", "Attack Type", "Source IP", "Count"
            ])
            for a in self.alerts:
                writer.writerow([
                    a.alert_id, a.timestamp, a.severity, a.title, a.message,
                    a.source, a.status, a.attack_type, a.src_ip, a.count
                ])

    def get_stats(self) -> Dict:
        """Alert statistics for dashboard widgets."""
        total = len(self.alerts)
        active = sum(1 for a in self.alerts if a.status == "ACTIVE")
        sev_counts = Counter(a.severity for a in self.alerts)
        attack_counts = Counter(
            a.attack_type for a in self.alerts if a.attack_type and a.attack_type != "Anomaly"
        )

        return {
            "total_alerts": total,
            "active_alerts": active,
            "acknowledged": sum(1 for a in self.alerts if a.status == "ACKNOWLEDGED"),
            "resolved": sum(1 for a in self.alerts if a.status == "RESOLVED"),
            "by_severity": {
                "INFO": sev_counts.get("INFO", 0),
                "WARNING": sev_counts.get("WARNING", 0),
                "CRITICAL": sev_counts.get("CRITICAL", 0),
                "EMERGENCY": sev_counts.get("EMERGENCY", 0),
            },
            "top_attacks": attack_counts.most_common(5),
            "emails_sent": len(self.email_log),
            "sms_sent": len(self.sms_log),
        }

    def clear(self):
        self.alerts.clear()
        self.notification_queue.clear()
        self.dashboard_notifications.clear()
        self.email_log.clear()
        self.sms_log.clear()
        self._recent_keys.clear()
        self._counter = 0
