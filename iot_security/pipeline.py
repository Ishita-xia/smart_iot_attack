"""
End-to-End Pipeline Orchestrator
=================================
Orchestrates the 9 stages of the Smart IDS flowchart:
1. IoT Network -> 2. Traffic Capture -> 3. Preprocessing -> 4. Detection Engines ->
5. Federated Learning -> 6. Traffic Filtering -> 7. Log Analysis & Threat Intelligence ->
8. Alerts & Notifications -> 9. Admin Dashboard
"""

import os
import time
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from traffic_simulator import TrafficCapture, IoTEnvironment
from traffic_filter import TrafficFilter, FilterAction
from log_manager import LogManager, LogLevel, EventType
from alert_system import AlertManager, AlertSeverity
from threat_intelligence import ThreatIntelligence


@dataclass
class PipelineResult:
    timestamp: str
    environment: str
    total_flows: int
    allowed_count: int
    blocked_count: int
    rate_limited_count: int
    terminated_count: int
    anomaly_count: int
    threats_detected: int
    stages_status: Dict[str, str] = field(default_factory=dict) # Stage -> Status (GREEN, RED, GREY)
    decisions: List[Dict] = field(default_factory=list)
    threat_intel_summary: Dict = field(default_factory=dict)
    execution_time_ms: float = 0.0


class IDSPipeline:
    """
    Orchestrates the entire Smart IDS Flowchart pipeline.
    """

    def __init__(
        self,
        log_manager: LogManager,
        alert_manager: AlertManager,
        traffic_filter: TrafficFilter,
        threat_intelligence: ThreatIntelligence,
        scaler = None,
        label_encoder = None,
    ):
        self.log_manager = log_manager
        self.alert_manager = alert_manager
        self.traffic_filter = traffic_filter
        self.threat_intel = threat_intelligence
        self.scaler = scaler
        self.label_encoder = label_encoder

    def run_pipeline(
        self,
        environment: IoTEnvironment,
        batch_size: int = 50,
        cache_rows: int = 1000,
        custom_flows: Optional[List[Dict]] = None
    ) -> PipelineResult:
        """
        Execute the 9-stage Smart IDS Pipeline.
        """
        start_time = time.time()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Initialize statuses
        stages = {
            "Stage 1: IoT Network": "GREY",
            "Stage 2: Traffic Capture": "GREY",
            "Stage 3: Preprocessing": "GREY",
            "Stage 4: Detection Engines": "GREY",
            "Stage 5: Federated Learning": "GREY",
            "Stage 6: Traffic Filtering": "GREY",
            "Stage 7: Log Analysis & Threat Intel": "GREY",
            "Stage 8: Alerts & Notifications": "GREY",
            "Stage 9: Admin Dashboard": "GREY",
        }

        self.log_manager.log(
            LogLevel.INFO,
            EventType.PIPELINE_START,
            "Pipeline",
            f"Executing end-to-end Smart IDS Flowchart pipeline in {environment.value} environment."
        )

        # ── Stage 1: IoT Network & Stage 2: Traffic Capture ──
        stages["Stage 1: IoT Network"] = "GREEN"
        stages["Stage 2: Traffic Capture"] = "GREEN"
        
        if custom_flows is not None:
            flows = custom_flows
            self.log_manager.log(
                LogLevel.INFO,
                EventType.TRAFFIC_CAPTURE,
                "Pipeline",
                f"Processing {len(flows)} uploaded custom CSV network flows."
            )
        else:
            try:
                capture = TrafficCapture(environment, cache_rows=cache_rows)
                capture.start()
                flows = capture.capture_batch(batch_size)
                capture.stop()
                
                self.log_manager.log(
                    LogLevel.INFO,
                    EventType.TRAFFIC_CAPTURE,
                    "Pipeline",
                    f"Captured {len(flows)} network flows from environment {environment.value}"
                )
            except Exception as e:
                stages["Stage 2: Traffic Capture"] = "RED"
                self.log_manager.log(
                    LogLevel.ERROR,
                    EventType.PIPELINE_RUN,
                    "Pipeline",
                    f"Capture stage failed: {str(e)}"
                )
                return PipelineResult(
                    timestamp=now_str,
                    environment=environment.value,
                    total_flows=0,
                    allowed_count=0,
                    blocked_count=0,
                    rate_limited_count=0,
                    terminated_count=0,
                    anomaly_count=0,
                    threats_detected=0,
                    stages_status=stages,
                    execution_time_ms=(time.time() - start_time) * 1000
                )

        if not flows:
            stages["Stage 2: Traffic Capture"] = "RED"
            return PipelineResult(
                timestamp=now_str,
                environment=environment.value,
                total_flows=0,
                allowed_count=0,
                blocked_count=0,
                rate_limited_count=0,
                terminated_count=0,
                anomaly_count=0,
                threats_detected=0,
                stages_status=stages,
                execution_time_ms=(time.time() - start_time) * 1000
            )

        # ── Stage 3: Preprocessing & Normalization ──
        stages["Stage 3: Preprocessing"] = "GREEN"
        # Done on features flow by flow using scaler

        # ── Stage 4: Detection Engines & Stage 6: Filtering ──
        # Update traffic filter models
        stages["Stage 4: Detection Engines"] = "GREEN" if (self.traffic_filter.cnn_model is not None or self.traffic_filter.autoencoder is not None) else "YELLOW"
        stages["Stage 6: Traffic Filtering"] = "GREEN"

        # Check if Federated Learning node simulation is active / has been run
        # FL is stage 5. If models are present, we indicate FL status.
        stages["Stage 5: Federated Learning"] = "GREEN" if self.traffic_filter.cnn_model is not None else "YELLOW"

        decisions = []
        allowed = 0
        blocked = 0
        rate_limited = 0
        terminated = 0
        anomalies = 0
        threats = 0

        for flow in flows:
            features = flow['features']
            if self.scaler is not None:
                try:
                    features = self.scaler.transform(features.reshape(1, -1)).flatten()
                except Exception:
                    pass

            # Make decision
            decision = self.traffic_filter.filter_flow(
                features,
                flow_id=flow['flow_id'],
                src_ip=flow['src_ip'],
                dst_ip=flow['dst_ip']
            )

            # Record decision metrics
            if decision.action == FilterAction.ALLOW:
                allowed += 1
            elif decision.action == FilterAction.BLOCK:
                blocked += 1
            elif decision.action == FilterAction.RATE_LIMIT:
                rate_limited += 1
            elif decision.action == FilterAction.TERMINATE:
                terminated += 1

            if decision.is_anomaly:
                anomalies += 1

            # Log traffic action
            self.log_manager.log_filter_action(
                decision.action.value,
                decision.reason,
                src_ip=flow['src_ip'],
                flow_id=flow['flow_id']
            )

            # ── Stage 7: Log Analysis & Threat Intelligence ──
            # Feed classification results into threat intelligence engine
            is_attack = (decision.action != FilterAction.ALLOW)
            attack_cat = "Benign"
            if is_attack:
                threats += 1
                if decision.cnn_prediction != "Benign_Final":
                    attack_cat = decision.cnn_prediction
                elif decision.is_anomaly:
                    attack_cat = "Anomaly"

            self.threat_intel.update_reputation(
                ip=flow['src_ip'],
                attack_type=decision.cnn_prediction,
                is_attack=is_attack,
                category=attack_cat
            )

            # ── Stage 8: Alerts & Notifications ──
            if is_attack:
                if decision.cnn_prediction != "Benign_Final":
                    self.log_manager.log_attack(
                        decision.cnn_prediction,
                        decision.cnn_confidence,
                        flow['src_ip'],
                        flow['flow_id']
                    )
                    self.alert_manager.alert_attack_detected(
                        decision.cnn_prediction,
                        decision.cnn_confidence,
                        flow['src_ip']
                    )
                if decision.is_anomaly:
                    self.log_manager.log_anomaly(
                        decision.anomaly_score,
                        self.traffic_filter.anomaly_threshold,
                        flow['src_ip'],
                        flow['flow_id']
                    )
                    self.alert_manager.alert_anomaly(
                        decision.anomaly_score,
                        self.traffic_filter.anomaly_threshold,
                        flow['src_ip']
                    )

            decisions.append({
                "flow_id": decision.flow_id,
                "src_ip": decision.src_ip,
                "dst_ip": decision.dst_ip,
                "protocol": flow["protocol"],
                "cnn_pred": decision.cnn_prediction,
                "cnn_conf": decision.cnn_confidence,
                "anomaly_score": decision.anomaly_score,
                "is_anomaly": decision.is_anomaly,
                "action": decision.action.value.upper(),
                "reason": decision.reason
            })

        # Correlate attack patterns with recent decisions
        recent_attacks = []
        for d in decisions:
            if d["action"] != "ALLOW":
                recent_attacks.append({
                    "attack_type": d["cnn_pred"],
                    "src_ip": d["src_ip"]
                })

        if recent_attacks:
            self.threat_intel.correlate_patterns(recent_attacks)
            self.threat_intel.extract_iocs()

        stages["Stage 7: Log Analysis & Threat Intel"] = "GREEN"
        stages["Stage 8: Alerts & Notifications"] = "GREEN"
        stages["Stage 9: Admin Dashboard"] = "GREEN"

        self.log_manager.log(
            LogLevel.INFO,
            EventType.PIPELINE_COMPLETE,
            "Pipeline",
            f"Pipeline complete. Handled {len(flows)} flows. Blocked: {blocked + terminated}, Allowed: {allowed}."
        )

        exec_time = (time.time() - start_time) * 1000

        return PipelineResult(
            timestamp=now_str,
            environment=environment.value,
            total_flows=len(flows),
            allowed_count=allowed,
            blocked_count=blocked,
            rate_limited_count=rate_limited,
            terminated_count=terminated,
            anomaly_count=anomalies,
            threats_detected=threats,
            stages_status=stages,
            decisions=decisions,
            threat_intel_summary=self.threat_intel.get_threat_summary(),
            execution_time_ms=round(exec_time, 2)
        )
