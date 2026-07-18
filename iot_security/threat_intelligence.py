"""
Threat Intelligence Engine
===========================
Provides IP reputation scoring, multi-stage attack pattern correlation,
automated threat interpretation, and IOC (Indicators of Compromise) extraction.

Flowchart Stage 7: Log Analysis & Threat Intelligence
"""

from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ── Multi-stage attack patterns (attack chain signatures) ──
ATTACK_CHAINS = {
    "Full Reconnaissance → Exploit": {
        "stages": ["Recon-PingSweep", "Recon-HostDiscovery", "Recon-PortScan",
                    "Recon-OSScan", "VulnerabilityScan"],
        "description": "Systematic network mapping followed by vulnerability identification",
        "risk": "HIGH",
        "next_likely": ["CommandInjection", "SqlInjection", "Backdoor_Malware"],
    },
    "Recon → Brute Force → Backdoor": {
        "stages": ["Recon-PortScan", "DictionaryBruteForce", "Backdoor_Malware"],
        "description": "Port scan to find services, brute-force credentials, install backdoor",
        "risk": "CRITICAL",
        "next_likely": ["Mirai-greeth_flood", "Mirai-udpplain"],
    },
    "Spoofing → Man-in-the-Middle": {
        "stages": ["DNS_Spoofing", "MITM-ArpSpoofing"],
        "description": "DNS/ARP spoofing to intercept and manipulate IoT traffic",
        "risk": "CRITICAL",
        "next_likely": ["Backdoor_Malware", "CommandInjection"],
    },
    "Botnet Recruitment → DDoS": {
        "stages": ["Backdoor_Malware", "Mirai-greeth_flood", "Mirai-greip_flood",
                    "Mirai-udpplain"],
        "description": "Devices compromised by malware then used for DDoS attacks",
        "risk": "CRITICAL",
        "next_likely": ["DDoS-TCP_Flood", "DDoS-UDP_Flood", "DDoS-SYN_Flood"],
    },
    "Web Exploit Chain": {
        "stages": ["XSS", "SqlInjection", "CommandInjection", "Uploading_Attack"],
        "description": "Chained web application attacks escalating from injection to RCE",
        "risk": "HIGH",
        "next_likely": ["Backdoor_Malware"],
    },
    "DDoS Escalation": {
        "stages": ["DoS-SYN_Flood", "DoS-TCP_Flood", "DDoS-SYN_Flood",
                    "DDoS-TCP_Flood", "DDoS-HTTP_Flood"],
        "description": "Attack escalating from single-source DoS to distributed DDoS",
        "risk": "HIGH",
        "next_likely": ["DDoS-SlowLoris", "DDoS-UDP_Flood"],
    },
}


@dataclass
class IPReputation:
    ip_address: str
    score: float = 100.0          # 0 (malicious) – 100 (clean)
    total_flows: int = 0
    attack_count: int = 0
    benign_count: int = 0
    attack_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    first_seen: str = ""
    last_seen: str = ""
    is_blocked: bool = False

    def to_dict(self) -> Dict:
        return {
            "ip": self.ip_address,
            "score": round(self.score, 1),
            "rating": self._rating(),
            "total_flows": self.total_flows,
            "attack_count": self.attack_count,
            "benign_count": self.benign_count,
            "top_attacks": dict(
                sorted(self.attack_types.items(), key=lambda x: -x[1])[:5]
            ),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "is_blocked": self.is_blocked,
        }

    def _rating(self) -> str:
        if self.score >= 80:
            return "CLEAN"
        if self.score >= 60:
            return "SUSPICIOUS"
        if self.score >= 30:
            return "MALICIOUS"
        return "CRITICAL"


@dataclass
class ThreatPattern:
    pattern_name: str
    matched_stages: List[str]
    total_stages: int
    match_pct: float
    risk: str
    description: str
    next_likely: List[str]
    source_ips: List[str]
    timestamp: str


@dataclass
class IOC:
    ioc_type: str          # "IP", "ATTACK_SIGNATURE", "PORT", "PATTERN"
    value: str
    severity: str
    context: str
    first_seen: str
    count: int = 1


class ThreatIntelligence:
    """
    Threat intelligence engine that correlates attack detections,
    tracks IP reputation, detects multi-stage attacks, and extracts IOCs.
    """

    def __init__(self):
        self.ip_reputations: Dict[str, IPReputation] = {}
        self.detected_patterns: List[ThreatPattern] = []
        self.iocs: List[IOC] = []
        self.attack_timeline: List[Dict] = []

        # Severity → score penalty mapping
        self._severity_penalty = {
            "Benign": 0,
            "Recon": 5,
            "Brute Force": 15,
            "Web Attack": 20,
            "Spoofing": 25,
            "DoS": 25,
            "DDoS": 30,
            "Malware": 40,
        }

    # ── IP Reputation ──

    def update_reputation(self, ip: str, attack_type: str, is_attack: bool,
                          category: str = "Unknown"):
        """Update IP reputation based on a traffic classification."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if ip not in self.ip_reputations:
            self.ip_reputations[ip] = IPReputation(
                ip_address=ip, first_seen=now
            )
        rep = self.ip_reputations[ip]
        rep.last_seen = now
        rep.total_flows += 1

        if is_attack:
            rep.attack_count += 1
            rep.attack_types[attack_type] += 1
            penalty = self._severity_penalty.get(category, 10)
            rep.score = max(0.0, rep.score - penalty)

            # Track in timeline
            self.attack_timeline.append({
                "time": now,
                "ip": ip,
                "attack": attack_type,
                "category": category,
            })
            if len(self.attack_timeline) > 5000:
                self.attack_timeline = self.attack_timeline[-5000:]
        else:
            rep.benign_count += 1
            # Slow reputation recovery for benign traffic
            rep.score = min(100.0, rep.score + 0.5)

        # Auto-block threshold
        if rep.score < 20:
            rep.is_blocked = True

    def get_ip_reputation(self, ip: str) -> Dict:
        if ip in self.ip_reputations:
            return self.ip_reputations[ip].to_dict()
        return {"ip": ip, "score": 100, "rating": "UNKNOWN", "total_flows": 0}

    def get_all_reputations(self, sort_by: str = "score",
                            limit: int = 100) -> List[Dict]:
        reps = list(self.ip_reputations.values())
        if sort_by == "score":
            reps.sort(key=lambda r: r.score)
        elif sort_by == "attacks":
            reps.sort(key=lambda r: -r.attack_count)
        return [r.to_dict() for r in reps[:limit]]

    def get_blocked_ips(self) -> List[str]:
        return [ip for ip, r in self.ip_reputations.items() if r.is_blocked]

    # ── Multi-stage Attack Pattern Correlation ──

    def correlate_patterns(self, recent_attacks: List[Dict]) -> List[Dict]:
        """
        Detect multi-stage attack patterns by checking if recent attacks
        match known attack chain signatures.

        Args:
            recent_attacks: list of {"attack_type": str, "src_ip": str, ...}
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Group attacks by source IP
        attacks_by_ip: Dict[str, List[str]] = defaultdict(list)
        for a in recent_attacks:
            attacks_by_ip[a.get("src_ip", "unknown")].append(
                a.get("attack_type", "")
            )

        self.detected_patterns.clear()

        for ip, attack_list in attacks_by_ip.items():
            attack_set = set(attack_list)
            for chain_name, chain_info in ATTACK_CHAINS.items():
                chain_stages = set(chain_info["stages"])
                matched = attack_set & chain_stages
                if len(matched) >= 2:
                    match_pct = len(matched) / len(chain_stages) * 100
                    pattern = ThreatPattern(
                        pattern_name=chain_name,
                        matched_stages=list(matched),
                        total_stages=len(chain_stages),
                        match_pct=round(match_pct, 1),
                        risk=chain_info["risk"],
                        description=chain_info["description"],
                        next_likely=chain_info["next_likely"],
                        source_ips=[ip],
                        timestamp=now,
                    )
                    self.detected_patterns.append(pattern)

        return [
            {
                "pattern": p.pattern_name,
                "matched": p.matched_stages,
                "total_stages": p.total_stages,
                "match_pct": p.match_pct,
                "risk": p.risk,
                "description": p.description,
                "next_likely": p.next_likely,
                "source_ips": p.source_ips,
                "time": p.timestamp,
            }
            for p in self.detected_patterns
        ]

    # ── IOC Extraction ──

    def extract_iocs(self) -> List[Dict]:
        """Extract Indicators of Compromise from tracked data."""
        self.iocs.clear()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Malicious IPs
        for ip, rep in self.ip_reputations.items():
            if rep.score < 50:
                severity = "CRITICAL" if rep.score < 20 else "HIGH"
                top_attack = max(rep.attack_types, key=rep.attack_types.get) \
                    if rep.attack_types else "Unknown"
                self.iocs.append(IOC(
                    ioc_type="IP",
                    value=ip,
                    severity=severity,
                    context=f"Reputation {rep.score:.0f}/100 — "
                            f"{rep.attack_count} attacks, top: {top_attack}",
                    first_seen=rep.first_seen,
                    count=rep.attack_count,
                ))

        # Attack signature IOCs
        all_attacks = Counter()
        for rep in self.ip_reputations.values():
            for atk, cnt in rep.attack_types.items():
                all_attacks[atk] += cnt

        for atk, cnt in all_attacks.most_common(20):
            if cnt >= 3:
                self.iocs.append(IOC(
                    ioc_type="ATTACK_SIGNATURE",
                    value=atk,
                    severity="HIGH" if cnt >= 10 else "MEDIUM",
                    context=f"Detected {cnt} times across network",
                    first_seen=now,
                    count=cnt,
                ))

        # Pattern IOCs
        for pattern in self.detected_patterns:
            self.iocs.append(IOC(
                ioc_type="PATTERN",
                value=pattern.pattern_name,
                severity=pattern.risk,
                context=f"Multi-stage attack: {pattern.description}",
                first_seen=pattern.timestamp,
                count=len(pattern.matched_stages),
            ))

        return [
            {
                "type": i.ioc_type,
                "value": i.value,
                "severity": i.severity,
                "context": i.context,
                "first_seen": i.first_seen,
                "count": i.count,
            }
            for i in self.iocs
        ]

    # ── Threat Intelligence Summary ──

    def get_threat_summary(self) -> Dict:
        """Generate a consolidated threat intelligence report."""
        total_ips = len(self.ip_reputations)
        malicious_ips = sum(
            1 for r in self.ip_reputations.values() if r.score < 50
        )
        blocked_ips = sum(
            1 for r in self.ip_reputations.values() if r.is_blocked
        )
        total_attacks = sum(
            r.attack_count for r in self.ip_reputations.values()
        )
        total_benign = sum(
            r.benign_count for r in self.ip_reputations.values()
        )

        # Top attack types globally
        global_attacks = Counter()
        for rep in self.ip_reputations.values():
            for atk, cnt in rep.attack_types.items():
                global_attacks[atk] += cnt

        # Risk level
        if blocked_ips > 5 or len(self.detected_patterns) > 3:
            risk_level = "CRITICAL"
        elif malicious_ips > 3 or len(self.detected_patterns) > 1:
            risk_level = "HIGH"
        elif malicious_ips > 0:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "risk_level": risk_level,
            "total_ips_tracked": total_ips,
            "malicious_ips": malicious_ips,
            "blocked_ips": blocked_ips,
            "total_attacks": total_attacks,
            "total_benign": total_benign,
            "top_attacks": global_attacks.most_common(10),
            "active_patterns": len(self.detected_patterns),
            "total_iocs": len(self.iocs),
            "patterns": [
                {
                    "name": p.pattern_name,
                    "risk": p.risk,
                    "match_pct": p.match_pct,
                    "ips": p.source_ips,
                }
                for p in self.detected_patterns
            ],
        }

    def get_attack_timeline(self, last_n: int = 200) -> List[Dict]:
        return self.attack_timeline[-last_n:]

    def clear(self):
        self.ip_reputations.clear()
        self.detected_patterns.clear()
        self.iocs.clear()
        self.attack_timeline.clear()
