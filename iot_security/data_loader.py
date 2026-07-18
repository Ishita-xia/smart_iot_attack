"""
Data Loader Module
==================
Loads all IoT attack/benign CSV files from the dataset folders,
assigns labels, preprocesses, and returns train/test splits.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# path to the root dataset folder
DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)))

FEATURE_COLS = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count',
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH',
    'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv',
    'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size',
    'IAT', 'Number', 'Variance'
]

ATTACK_GROUPS = {
    'Benign_Final': 'Benign',
    'DDoS-ACK_Fragmentation': 'DDoS',
    'DDoS-HTTP_Flood': 'DDoS',
    'DDoS-ICMP_Flood': 'DDoS',
    'DDoS-ICMP_Fragmentation': 'DDoS',
    'DDoS-PSHACK_FLOOD': 'DDoS',
    'DDoS-RSTFINFLOOD': 'DDoS',
    'DDoS-SYN_Flood': 'DDoS',
    'DDoS-SlowLoris': 'DDoS',
    'DDoS-SynonymousIP_Flood': 'DDoS',
    'DDoS-TCP_Flood': 'DDoS',
    'DDoS-UDP_Flood': 'DDoS',
    'DDoS-UDP_Fragmentation': 'DDoS',
    'DoS-HTTP_Flood': 'DoS',
    'DoS-SYN_Flood': 'DoS',
    'DoS-TCP_Flood': 'DoS',
    'DoS-UDP_Flood': 'DoS',
    'Backdoor_Malware': 'Malware',
    'Mirai-greeth_flood': 'Malware',
    'Mirai-greip_flood': 'Malware',
    'Mirai-udpplain': 'Malware',
    'Recon-HostDiscovery': 'Recon',
    'Recon-OSScan': 'Recon',
    'Recon-PingSweep': 'Recon',
    'Recon-PortScan': 'Recon',
    'BrowserHijacking': 'Web Attack',
    'CommandInjection': 'Web Attack',
    'SqlInjection': 'Web Attack',
    'XSS': 'Web Attack',
    'Uploading_Attack': 'Web Attack',
    'DNS_Spoofing': 'Spoofing',
    'MITM-ArpSpoofing': 'Spoofing',
    'DictionaryBruteForce': 'Brute Force',
    'VulnerabilityScan': 'Recon',
}


def load_dataset(max_rows_per_class=5000, verbose=True):
    all_dfs = []
    folders = sorted(os.listdir(DATASET_ROOT))

    for folder in folders:
        folder_path = os.path.join(DATASET_ROOT, folder)
        if not os.path.isdir(folder_path):
            continue
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not csv_files:
            continue

        label = folder
        rows_loaded = 0
        for csv_file in csv_files:
            if rows_loaded >= max_rows_per_class:
                break
            filepath = os.path.join(folder_path, csv_file)
            try:
                chunk_size = max_rows_per_class - rows_loaded
                df = pd.read_csv(filepath, nrows=chunk_size, low_memory=False)
                df = df[[c for c in FEATURE_COLS if c in df.columns]]
                df['label'] = label
                all_dfs.append(df)
                rows_loaded += len(df)
                if verbose:
                    print(f"  [OK] {folder}/{csv_file}  ->  {len(df)} rows")
            except Exception as e:
                if verbose:
                    print(f"  [ERR] Error reading {filepath}: {e}")

    combined = pd.concat(all_dfs, ignore_index=True)

    for col in FEATURE_COLS:
        if col not in combined.columns:
            combined[col] = 0

    X = combined[FEATURE_COLS].copy()
    y_raw = combined['label'].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values.astype(np.float32))

    if verbose:
        print(f"\n[OK] Dataset loaded: {X_scaled.shape[0]} samples, "
              f"{X_scaled.shape[1]} features, {len(le.classes_)} classes")

    return X_scaled, y, le, scaler


def get_train_test(max_rows_per_class=5000, test_size=0.2, seed=42):
    X, y, le, scaler = load_dataset(max_rows_per_class=max_rows_per_class)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_tr, X_te, y_tr, y_te, le, scaler


def get_class_names(le):
    return list(le.classes_)


def get_group_for_label(label_name):
    return ATTACK_GROUPS.get(label_name, 'Unknown')
