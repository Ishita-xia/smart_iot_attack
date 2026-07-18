"""
QwenCoder Explainer Module — Enhanced
======================================
Full attack knowledge base for all 34 classes + threat report generation.
"""
import json
from datetime import datetime
from collections import Counter

ATTACK_KNOWLEDGE = {
    # ── Benign ──
    "Benign_Final": {
        "description": "Normal, legitimate network traffic from IoT devices operating within expected parameters.",
        "severity": "NONE",
        "indicators": ["Normal flow patterns", "Standard packet sizes", "Expected protocols"],
        "mitigation": "No action required. Continue monitoring.",
        "iot_impact": "None — normal operation.",
        "category": "Benign",
    },
    # ── DDoS Attacks ──
    "DDoS-ACK_Fragmentation": {
        "description": "DDoS attack using fragmented ACK packets to overwhelm target's reassembly buffers and exhaust resources.",
        "severity": "HIGH",
        "indicators": ["High ACK flag count", "Fragmented packets", "Elevated rate", "Multiple source IPs"],
        "mitigation": "Enable anti-fragmentation rules, rate limiting, deploy DDoS mitigation appliance.",
        "iot_impact": "Can crash resource-constrained IoT devices attempting to reassemble fragments.",
        "category": "DDoS",
    },
    "DDoS-HTTP_Flood": {
        "description": "HTTP Flood DDoS attack sending massive HTTP GET/POST requests to overwhelm web servers.",
        "severity": "HIGH",
        "indicators": ["High HTTP count", "Elevated rate", "Many TCP connections", "High request frequency"],
        "mitigation": "Rate limiting, WAF rules, CDN protection, CAPTCHA challenges, IP blacklisting.",
        "iot_impact": "Can render IoT management interfaces and cloud dashboards unreachable.",
        "category": "DDoS",
    },
    "DDoS-ICMP_Flood": {
        "description": "ICMP Flood (Ping Flood) overwhelming targets with ICMP Echo Request packets.",
        "severity": "HIGH",
        "indicators": ["High ICMP count", "Elevated rate", "Large packet volume"],
        "mitigation": "Block excessive ICMP at firewall, rate-limit ICMP, enable ICMP flood protection.",
        "iot_impact": "Exhausts bandwidth on IoT networks, causing communication failures between devices.",
        "category": "DDoS",
    },
    "DDoS-ICMP_Fragmentation": {
        "description": "ICMP fragmentation attack sending oversized ICMP packets requiring reassembly.",
        "severity": "HIGH",
        "indicators": ["Fragmented ICMP packets", "High packet count", "Large payload sizes"],
        "mitigation": "Drop fragmented ICMP, limit ICMP packet sizes, enable anti-fragmentation.",
        "iot_impact": "Memory exhaustion on IoT devices with limited RAM during fragment reassembly.",
        "category": "DDoS",
    },
    "DDoS-PSHACK_FLOOD": {
        "description": "PSH+ACK flood attack sending TCP packets with PSH and ACK flags to force immediate data processing.",
        "severity": "HIGH",
        "indicators": ["High PSH flag count", "High ACK count", "Elevated rate"],
        "mitigation": "TCP flood protection, connection rate limiting, SYN proxy.",
        "iot_impact": "Forces IoT devices to process data immediately, exhausting CPU resources.",
        "category": "DDoS",
    },
    "DDoS-RSTFINFLOOD": {
        "description": "RST/FIN flood attack sending TCP reset and finish packets to disrupt active connections.",
        "severity": "HIGH",
        "indicators": ["High RST flag count", "High FIN flag count", "Abnormal connection teardowns"],
        "mitigation": "Stateful firewall inspection, rate limiting RST/FIN packets.",
        "iot_impact": "Forcefully terminates legitimate IoT device connections to cloud services.",
        "category": "DDoS",
    },
    "DDoS-SYN_Flood": {
        "description": "SYN Flood exploiting TCP three-way handshake by sending massive SYN requests without completing.",
        "severity": "HIGH",
        "indicators": ["High SYN count", "Low ACK count", "High rate", "Half-open connections"],
        "mitigation": "SYN cookies, SYN proxy, firewall rate limiting, increase backlog queue.",
        "iot_impact": "Exhausts TCP connection table on IoT gateways, preventing new connections.",
        "category": "DDoS",
    },
    "DDoS-SlowLoris": {
        "description": "Slowloris attack holding connections open by sending partial HTTP headers slowly.",
        "severity": "HIGH",
        "indicators": ["Long-lived connections", "Low data rate per connection", "Incomplete HTTP headers"],
        "mitigation": "Connection timeouts, limit connections per IP, use reverse proxy (Nginx/HAProxy).",
        "iot_impact": "Ties up all available connections on IoT web servers with minimal attacker resources.",
        "category": "DDoS",
    },
    "DDoS-SynonymousIP_Flood": {
        "description": "DDoS attack using spoofed IPs identical to the target's own IP to cause routing confusion.",
        "severity": "HIGH",
        "indicators": ["Source IP matches target IP", "High traffic volume", "Routing anomalies"],
        "mitigation": "Anti-spoofing filters (BCP38), ingress/egress filtering, uRPF.",
        "iot_impact": "Can cause IoT network routing loops and communication breakdown.",
        "category": "DDoS",
    },
    "DDoS-TCP_Flood": {
        "description": "Generic TCP flood attack overwhelming targets with high volumes of TCP packets.",
        "severity": "HIGH",
        "indicators": ["High TCP traffic volume", "Elevated connection rate", "Multiple flag combinations"],
        "mitigation": "TCP rate limiting, connection throttling, DDoS mitigation services.",
        "iot_impact": "Saturates IoT network bandwidth, causing widespread connectivity loss.",
        "category": "DDoS",
    },
    "DDoS-UDP_Flood": {
        "description": "UDP flood attack overwhelming targets with high-volume UDP packets on random ports.",
        "severity": "HIGH",
        "indicators": ["High UDP traffic", "Random destination ports", "ICMP Port Unreachable responses"],
        "mitigation": "UDP rate limiting, port filtering, enable UDP flood protection.",
        "iot_impact": "Saturates IoT network links, disrupts time-sensitive sensor communications.",
        "category": "DDoS",
    },
    "DDoS-UDP_Fragmentation": {
        "description": "UDP fragmentation attack sending oversized UDP datagrams requiring reassembly.",
        "severity": "HIGH",
        "indicators": ["Fragmented UDP packets", "Large datagrams", "High reassembly overhead"],
        "mitigation": "Limit UDP fragment size, drop oversized UDP, anti-fragmentation rules.",
        "iot_impact": "Memory exhaustion on resource-limited IoT devices during fragment reassembly.",
        "category": "DDoS",
    },
    # ── DoS Attacks ──
    "DoS-HTTP_Flood": {
        "description": "Single-source HTTP flood sending many requests to exhaust server resources.",
        "severity": "MEDIUM",
        "indicators": ["High HTTP request rate from single IP", "Elevated server CPU"],
        "mitigation": "Per-IP rate limiting, WAF rules, CAPTCHA.",
        "iot_impact": "Can take down individual IoT device web interfaces.",
        "category": "DoS",
    },
    "DoS-SYN_Flood": {
        "description": "Single-source SYN flood exhausting target's TCP connection table.",
        "severity": "MEDIUM",
        "indicators": ["High SYN rate from single source", "Half-open TCP connections"],
        "mitigation": "SYN cookies, per-IP connection limits.",
        "iot_impact": "Prevents the targeted IoT gateway from accepting new connections.",
        "category": "DoS",
    },
    "DoS-TCP_Flood": {
        "description": "Single-source TCP flood with mixed flag combinations.",
        "severity": "MEDIUM",
        "indicators": ["High TCP rate from single IP", "Mixed TCP flags"],
        "mitigation": "Connection rate limiting, stateful inspection.",
        "iot_impact": "Degrades performance of targeted IoT services.",
        "category": "DoS",
    },
    "DoS-UDP_Flood": {
        "description": "Single-source UDP flood targeting specific services.",
        "severity": "MEDIUM",
        "indicators": ["High UDP rate from single IP", "Targeted port flooding"],
        "mitigation": "Per-IP UDP rate limiting, port-based filtering.",
        "iot_impact": "Disrupts UDP-based IoT protocols (CoAP, MQTT-SN).",
        "category": "DoS",
    },
    # ── Malware / Botnet ──
    "Backdoor_Malware": {
        "description": "Backdoor malware establishing persistent unauthorized access to compromised IoT devices.",
        "severity": "CRITICAL",
        "indicators": ["Unusual outbound connections", "Command-and-control traffic", "Unexpected ports open"],
        "mitigation": "Isolate device, firmware reset, patch vulnerabilities, network segmentation.",
        "iot_impact": "Full device compromise — attacker gains persistent control over IoT devices.",
        "category": "Malware",
    },
    "Mirai-greeth_flood": {
        "description": "Mirai botnet variant performing GRE/ETH flood attacks from compromised IoT devices.",
        "severity": "CRITICAL",
        "indicators": ["GRE protocol traffic", "Botnet C&C patterns", "Known Mirai signatures"],
        "mitigation": "Change default credentials, firmware update, block C&C IPs, network segmentation.",
        "iot_impact": "IoT device becomes botnet zombie, used to attack others.",
        "category": "Malware",
    },
    "Mirai-greip_flood": {
        "description": "Mirai botnet variant performing GRE/IP flood attacks to generate volumetric DDoS.",
        "severity": "CRITICAL",
        "indicators": ["GRE/IP encapsulated flood traffic", "Botnet behavior patterns"],
        "mitigation": "Block GRE at perimeter, update IoT firmware, change default passwords.",
        "iot_impact": "Turns IoT devices into DDoS attack tools, consuming all bandwidth.",
        "category": "Malware",
    },
    "Mirai-udpplain": {
        "description": "Mirai botnet variant performing plain UDP flood attacks from compromised IoT devices.",
        "severity": "CRITICAL",
        "indicators": ["High volume UDP from IoT devices", "Known Mirai payload patterns"],
        "mitigation": "Credential hardening, firmware patches, egress filtering.",
        "iot_impact": "IoT device fully controlled by attacker, participating in large-scale DDoS.",
        "category": "Malware",
    },
    # ── Reconnaissance ──
    "Recon-HostDiscovery": {
        "description": "Network reconnaissance scanning to discover active hosts and IoT devices on the network.",
        "severity": "LOW",
        "indicators": ["ARP scans", "ICMP sweeps", "Sequential IP probing"],
        "mitigation": "Disable unnecessary protocols, implement IDS rules, network segmentation.",
        "iot_impact": "Attacker maps the IoT network topology — precursor to targeted attacks.",
        "category": "Recon",
    },
    "Recon-OSScan": {
        "description": "OS fingerprinting scan to identify operating systems and firmware on IoT devices.",
        "severity": "LOW",
        "indicators": ["TCP/IP stack probing", "Unusual flag combinations", "TTL analysis probes"],
        "mitigation": "OS fingerprint obfuscation, firewall rules blocking probes.",
        "iot_impact": "Reveals IoT device firmware versions, enabling targeted exploit selection.",
        "category": "Recon",
    },
    "Recon-PingSweep": {
        "description": "ICMP ping sweep scanning a range of IPs to find responsive hosts.",
        "severity": "LOW",
        "indicators": ["Sequential ICMP Echo requests", "Rapid IP scanning"],
        "mitigation": "Block ICMP at perimeter, enable ping sweep detection.",
        "iot_impact": "Identifies active IoT devices for further attack targeting.",
        "category": "Recon",
    },
    "Recon-PortScan": {
        "description": "Port scanning to discover open services on IoT devices.",
        "severity": "LOW",
        "indicators": ["Sequential port probing", "SYN scans", "Multiple ports per host"],
        "mitigation": "Close unnecessary ports, enable port scan detection, use port knocking.",
        "iot_impact": "Exposes vulnerable services running on IoT devices.",
        "category": "Recon",
    },
    "VulnerabilityScan": {
        "description": "Automated vulnerability scanning probing for known CVEs and misconfigurations.",
        "severity": "MEDIUM",
        "indicators": ["Known vulnerability probe patterns", "Multiple exploit attempts", "Scanner user-agents"],
        "mitigation": "Patch management, WAF rules, disable version disclosure.",
        "iot_impact": "Identifies exploitable vulnerabilities in IoT firmware and services.",
        "category": "Recon",
    },
    # ── Web Attacks ──
    "BrowserHijacking": {
        "description": "Browser hijacking attack redirecting IoT web interfaces to malicious content.",
        "severity": "MEDIUM",
        "indicators": ["DNS redirection", "Modified HTTP responses", "Injected JavaScript"],
        "mitigation": "HTTPS enforcement, Content Security Policy, browser security headers.",
        "iot_impact": "Compromises IoT device management interfaces, enabling configuration theft.",
        "category": "Web Attack",
    },
    "CommandInjection": {
        "description": "OS command injection exploiting IoT device web interfaces to execute system commands.",
        "severity": "HIGH",
        "indicators": ["Shell metacharacters in requests", "Unusual process execution", "Command chaining"],
        "mitigation": "Input sanitization, parameterized commands, least privilege, WAF.",
        "iot_impact": "Full device takeover — attacker gains shell access to IoT device OS.",
        "category": "Web Attack",
    },
    "SqlInjection": {
        "description": "SQL injection attacking database-backed IoT management systems.",
        "severity": "HIGH",
        "indicators": ["SQL syntax in parameters", "Error-based responses", "Union-based queries"],
        "mitigation": "Parameterized queries, ORM usage, input validation, WAF rules.",
        "iot_impact": "Data exfiltration from IoT databases — device credentials, sensor data, configs.",
        "category": "Web Attack",
    },
    "XSS": {
        "description": "Cross-Site Scripting injecting malicious scripts into IoT web dashboards.",
        "severity": "MEDIUM",
        "indicators": ["Script tags in parameters", "Event handler injection", "Encoded payloads"],
        "mitigation": "Output encoding, Content Security Policy, input sanitization.",
        "iot_impact": "Session hijacking of IoT admin panels, unauthorized device control.",
        "category": "Web Attack",
    },
    "Uploading_Attack": {
        "description": "Malicious file upload exploiting IoT firmware update or configuration upload interfaces.",
        "severity": "HIGH",
        "indicators": ["Unusual file uploads", "Executable content in uploads", "Modified firmware images"],
        "mitigation": "File type validation, sandbox uploads, integrity verification, signed firmware.",
        "iot_impact": "Malicious firmware installation — complete persistent device compromise.",
        "category": "Web Attack",
    },
    # ── Spoofing ──
    "DNS_Spoofing": {
        "description": "DNS spoofing/poisoning redirecting IoT device DNS queries to malicious servers.",
        "severity": "HIGH",
        "indicators": ["Mismatched DNS responses", "Unexpected DNS servers", "TTL anomalies"],
        "mitigation": "DNSSEC, DNS over HTTPS, pin DNS servers, verify DNS responses.",
        "iot_impact": "Redirects IoT devices to malicious update/C&C servers.",
        "category": "Spoofing",
    },
    "MITM-ArpSpoofing": {
        "description": "ARP spoofing enabling Man-in-the-Middle interception of IoT network traffic.",
        "severity": "HIGH",
        "indicators": ["Duplicate ARP entries", "MAC address changes", "Gratuitous ARP packets"],
        "mitigation": "Static ARP entries, Dynamic ARP Inspection, 802.1X, encrypted communications.",
        "iot_impact": "All IoT traffic intercepted — sensor data, credentials, commands visible to attacker.",
        "category": "Spoofing",
    },
    # ── Brute Force ──
    "DictionaryBruteForce": {
        "description": "Dictionary-based brute force attack targeting IoT device login interfaces.",
        "severity": "MEDIUM",
        "indicators": ["Repeated failed login attempts", "Dictionary password patterns", "High auth rate"],
        "mitigation": "Account lockout, CAPTCHA, MFA, strong password policies, fail2ban.",
        "iot_impact": "Compromises default/weak IoT credentials — many IoT devices use factory passwords.",
        "category": "Brute Force",
    },
}

DEFAULT_KNOWLEDGE = {
    "description": "Network attack detected. Traffic pattern does not match known benign behavior.",
    "severity": "MEDIUM",
    "indicators": ["Anomalous traffic patterns", "Deviation from baseline"],
    "mitigation": "Investigate the source IP. Isolate affected devices. Review logs.",
    "iot_impact": "Potential disruption to IoT device operations.",
    "category": "Unknown",
}


class QwenExplainer:
    """IoT Security threat explainer with full attack knowledge base."""

    def __init__(self):
        self.knowledge = ATTACK_KNOWLEDGE
        self.report_history = []

    def explain(self, predicted_class: str, confidence: float, sample_features: dict = None) -> str:
        """Generate a detailed explanation for a detected attack."""
        info = self.knowledge.get(predicted_class, DEFAULT_KNOWLEDGE)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"╔══════════════════════════════════════════════════════╗",
            f"║  🤖 QwenCoder IoT Security Analysis                 ║",
            f"║  Timestamp: {timestamp}                    ║",
            f"╚══════════════════════════════════════════════════════╝",
            f"",
            f"  🎯 Detected Attack : {predicted_class}",
            f"  📊 Confidence      : {confidence * 100:.1f}%",
            f"  ⚡ Severity        : {info['severity']}",
            f"  📁 Category        : {info.get('category', 'Unknown')}",
            f"",
            f"  🔍 What is this attack?",
            f"  {info['description']}",
            f"",
            f"  ⚠️  Key Indicators:",
        ]
        for ind in info['indicators']:
            lines.append(f"    • {ind}")
        lines += [
            f"",
            f"  🌐 IoT Impact:",
            f"  {info['iot_impact']}",
            f"",
            f"  🛡️  Recommended Mitigation:",
            f"  {info['mitigation']}",
            f"",
            f"  ─────────────────────────────────────────────────────",
        ]
        report = "\n".join(lines)
        self.report_history.append({
            "timestamp": timestamp,
            "attack": predicted_class,
            "confidence": confidence,
        })
        return report

    def get_attack_info(self, attack_class: str) -> dict:
        """Get structured attack information."""
        return self.knowledge.get(attack_class, DEFAULT_KNOWLEDGE)

    def get_all_attack_categories(self) -> dict:
        """Return attacks grouped by category."""
        categories = {}
        for name, info in self.knowledge.items():
            cat = info.get("category", "Unknown")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(name)
        return categories

    def generate_threat_report(self, detections: list) -> dict:
        """Generate comprehensive threat report from a list of detections."""
        classes = [d['class'] for d in detections]
        counts = Counter(classes)
        total = len(detections)
        benign = counts.get('Benign_Final', 0)
        attacks = total - benign

        severity_counts = {'NONE': 0, 'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        category_counts = Counter()
        for cls, cnt in counts.items():
            info = self.knowledge.get(cls, DEFAULT_KNOWLEDGE)
            sev = info['severity']
            severity_counts[sev] += cnt
            category_counts[info.get('category', 'Unknown')] += cnt

        top_attacks = counts.most_common(10)

        report = {
            'generated_at': datetime.now().isoformat(),
            'total_flows': total,
            'benign_flows': benign,
            'attack_flows': attacks,
            'attack_rate_pct': round(attacks / (total + 1e-9) * 100, 2),
            'top_attacks': top_attacks,
            'severity_distribution': severity_counts,
            'category_distribution': dict(category_counts),
            'risk_level': self._compute_risk(severity_counts),
            'summary': self._narrative_summary(attacks, total, top_attacks),
            'recommendations': self._generate_recommendations(top_attacks),
        }
        return report

    def _compute_risk(self, severity_counts):
        if severity_counts.get('CRITICAL', 0) > 0:
            return 'CRITICAL'
        if severity_counts.get('HIGH', 0) > 0:
            return 'HIGH'
        if severity_counts.get('MEDIUM', 0) > 0:
            return 'MEDIUM'
        if severity_counts.get('LOW', 0) > 0:
            return 'LOW'
        return 'NONE'

    def _narrative_summary(self, attacks, total, top_attacks):
        if attacks == 0:
            return "✅ All network traffic appears normal. No security threats detected."
        rate = attacks / (total + 1e-9) * 100
        top = top_attacks[0][0] if top_attacks else 'Unknown'
        return (
            f"🚨 Security Alert: {attacks}/{total} flows ({rate:.1f}%) identified as malicious. "
            f"Primary threat: {top}. Immediate action recommended."
        )

    def _generate_recommendations(self, top_attacks) -> list:
        """Generate actionable recommendations based on top attacks."""
        recs = []
        seen_categories = set()
        for attack_name, count in top_attacks:
            info = self.knowledge.get(attack_name, DEFAULT_KNOWLEDGE)
            cat = info.get('category', 'Unknown')
            if cat not in seen_categories and cat != 'Benign':
                seen_categories.add(cat)
                recs.append({
                    'category': cat,
                    'action': info['mitigation'],
                    'priority': info['severity'],
                    'top_attack': attack_name,
                    'occurrences': count,
                })
        return sorted(recs, key=lambda r: {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}.get(r['priority'], 4))
