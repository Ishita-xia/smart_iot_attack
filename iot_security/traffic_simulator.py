"""
Traffic Simulator Module
========================
Simulates IoT network environments and generates realistic traffic flows
by sampling from the real dataset CSVs.

Components:
  1. IoT Network → Defines device profiles for 4 environments
  2. Traffic Capture → Samples real data to mimic live packet capture
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

# ── Dataset root (parent of iot_security/) ──
DATASET_ROOT = os.path.dirname(os.path.dirname(__file__))


# ═══════════════════════════════════════════════════════════════════════
#  1.  IoT Network — Environment & Device Definitions
# ═══════════════════════════════════════════════════════════════════════

class IoTEnvironment(Enum):
    SMART_HOME = "Smart Home"
    HEALTHCARE = "Healthcare"
    INDUSTRIAL = "Industrial Automation"
    SMART_CITY = "Smart City"


@dataclass
class IoTDevice:
    device_id: str
    device_type: str
    environment: IoTEnvironment
    ip_address: str
    mac_address: str
    status: str = "online"
    last_seen: datetime = field(default_factory=datetime.now)


# Device templates for each environment
IOT_DEVICE_TEMPLATES = {
    IoTEnvironment.SMART_HOME: [
        ("SH-CAM-{id}", "Security Camera"),
        ("SH-THRM-{id}", "Smart Thermostat"),
        ("SH-LOCK-{id}", "Smart Lock"),
        ("SH-LIGHT-{id}", "Smart Light"),
        ("SH-SPEAK-{id}", "Smart Speaker"),
        ("SH-TV-{id}", "Smart TV"),
    ],
    IoTEnvironment.HEALTHCARE: [
        ("HC-MON-{id}", "Patient Monitor"),
        ("HC-PUMP-{id}", "Infusion Pump"),
        ("HC-VENT-{id}", "Ventilator"),
        ("HC-SCAN-{id}", "MRI Scanner"),
        ("HC-WEAR-{id}", "Wearable Sensor"),
    ],
    IoTEnvironment.INDUSTRIAL: [
        ("IND-PLC-{id}", "PLC Controller"),
        ("IND-SENS-{id}", "Temperature Sensor"),
        ("IND-ACT-{id}", "Actuator"),
        ("IND-ROB-{id}", "Robot Arm"),
        ("IND-METER-{id}", "Power Meter"),
    ],
    IoTEnvironment.SMART_CITY: [
        ("SC-TRAF-{id}", "Traffic Light"),
        ("SC-CAM-{id}", "Surveillance Camera"),
        ("SC-AIR-{id}", "Air Quality Sensor"),
        ("SC-PARK-{id}", "Parking Sensor"),
        ("SC-WATER-{id}", "Water Flow Meter"),
    ],
}

# Map attack types likely per environment
ENVIRONMENT_ATTACK_PROFILE = {
    IoTEnvironment.SMART_HOME: {
        "benign_ratio": 0.70,
        "common_attacks": [
            "Mirai-greeth_flood", "Mirai-greip_flood", "Mirai-udpplain",
            "DDoS-TCP_Flood", "BrowserHijacking", "DictionaryBruteForce",
        ],
    },
    IoTEnvironment.HEALTHCARE: {
        "benign_ratio": 0.80,
        "common_attacks": [
            "Backdoor_Malware", "MITM-ArpSpoofing", "DNS_Spoofing",
            "Recon-PortScan", "SqlInjection", "XSS",
        ],
    },
    IoTEnvironment.INDUSTRIAL: {
        "benign_ratio": 0.75,
        "common_attacks": [
            "DDoS-SYN_Flood", "DDoS-UDP_Flood", "CommandInjection",
            "Recon-OSScan", "Recon-HostDiscovery", "VulnerabilityScan",
        ],
    },
    IoTEnvironment.SMART_CITY: {
        "benign_ratio": 0.65,
        "common_attacks": [
            "DDoS-HTTP_Flood", "DDoS-SlowLoris", "DDoS-ICMP_Flood",
            "DoS-HTTP_Flood", "DoS-SYN_Flood", "Uploading_Attack",
        ],
    },
}


def generate_mac():
    return ":".join(f"{random.randint(0, 255):02x}" for _ in range(6))


def generate_ip(subnet="192.168.1"):
    return f"{subnet}.{random.randint(10, 254)}"


def create_iot_network(environment: IoTEnvironment, devices_per_type: int = 2) -> List[IoTDevice]:
    """Generate a list of IoT devices for the given environment."""
    devices = []
    templates = IOT_DEVICE_TEMPLATES.get(environment, [])
    for template_id, device_type in templates:
        for i in range(1, devices_per_type + 1):
            dev = IoTDevice(
                device_id=template_id.format(id=f"{i:03d}"),
                device_type=device_type,
                environment=environment,
                ip_address=generate_ip(),
                mac_address=generate_mac(),
            )
            devices.append(dev)
    return devices


# ═══════════════════════════════════════════════════════════════════════
#  2.  Traffic Capture — Sample real CSV data to simulate live flows
# ═══════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count',
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH',
    'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv',
    'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size',
    'IAT', 'Number', 'Variance',
]


class TrafficCapture:
    """
    Simulates real-time IoT traffic capture by sampling from dataset CSVs.
    """

    def __init__(self, environment: IoTEnvironment, cache_rows: int = 2000):
        self.environment = environment
        self.cache_rows = cache_rows
        self.devices = create_iot_network(environment)
        self.profile = ENVIRONMENT_ATTACK_PROFILE[environment]
        self._cache: Dict[str, pd.DataFrame] = {}
        self._load_cache()
        self.is_capturing = False
        self.total_captured = 0

    # ── cache a small sample from each relevant folder ──
    def _load_cache(self):
        """Load small samples from Benign + environment-specific attack folders."""
        folders_to_load = ["Benign_Final"] + self.profile["common_attacks"]
        for folder in folders_to_load:
            folder_path = os.path.join(DATASET_ROOT, folder)
            if not os.path.isdir(folder_path):
                continue
            csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
            if not csv_files:
                continue
            try:
                fp = os.path.join(folder_path, csv_files[0])
                df = pd.read_csv(fp, nrows=self.cache_rows, low_memory=False)
                available = [c for c in FEATURE_COLS if c in df.columns]
                df = df[available].copy()
                for c in FEATURE_COLS:
                    if c not in df.columns:
                        df[c] = 0
                df = df[FEATURE_COLS]
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.fillna(0, inplace=True)
                self._cache[folder] = df
            except Exception:
                pass

    def start(self):
        self.is_capturing = True

    def stop(self):
        self.is_capturing = False

    def capture_batch(self, batch_size: int = 50) -> List[Dict]:
        """
        Return a batch of simulated traffic flows.
        Each flow has: features (np array), label, device, timestamp, flow_id.
        """
        if not self.is_capturing:
            return []

        flows = []
        benign_ratio = self.profile["benign_ratio"]
        attack_folders = [f for f in self.profile["common_attacks"] if f in self._cache]

        for _ in range(batch_size):
            is_benign = random.random() < benign_ratio

            if is_benign and "Benign_Final" in self._cache:
                label = "Benign_Final"
                src_df = self._cache["Benign_Final"]
            elif attack_folders:
                label = random.choice(attack_folders)
                src_df = self._cache[label]
            elif "Benign_Final" in self._cache:
                label = "Benign_Final"
                src_df = self._cache["Benign_Final"]
            else:
                continue

            row = src_df.sample(1).iloc[0]
            device = random.choice(self.devices)
            self.total_captured += 1

            flows.append({
                "flow_id": f"FL-{self.total_captured:06d}",
                "timestamp": datetime.now() - timedelta(seconds=random.uniform(0, 5)),
                "device_id": device.device_id,
                "device_type": device.device_type,
                "src_ip": device.ip_address,
                "dst_ip": generate_ip("10.0.0"),
                "label": label,
                "features": row.values.astype(np.float32),
                "protocol": self._guess_protocol(row),
            })

        return flows

    @staticmethod
    def _guess_protocol(row) -> str:
        if row.get("TCP", 0) > 0:
            return "TCP"
        if row.get("UDP", 0) > 0:
            return "UDP"
        if row.get("ICMP", 0) > 0:
            return "ICMP"
        if row.get("HTTP", 0) > 0:
            return "HTTP"
        if row.get("DNS", 0) > 0:
            return "DNS"
        return "OTHER"

    def get_network_info(self) -> Dict:
        return {
            "environment": self.environment.value,
            "num_devices": len(self.devices),
            "devices": [
                {
                    "id": d.device_id,
                    "type": d.device_type,
                    "ip": d.ip_address,
                    "mac": d.mac_address,
                    "status": d.status,
                }
                for d in self.devices
            ],
            "total_captured": self.total_captured,
            "is_capturing": self.is_capturing,
        }
