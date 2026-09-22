import os
import sys
import csv
import json
import math
import re
import hashlib
import argparse
import random
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import ollama
from sentence_transformers import SentenceTransformer
import chromadb
import pypdf



OLLAMA_MODEL    = "qwen2.5:3b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH     = "./chroma_db"
DATA_PATH       = "./enterprise_data"

CHUNK_SIZE      = 400
CHUNK_OVERLAP   = 80
TOP_K           = 5
HYBRID_ALPHA    = 0.7
CONFIDENCE_THRESH = 0.25


CATEGORIES = ["hr", "finance", "it", "compliance", "general"]


ROLES: Dict[str, List[str]] = {
    "admin":              ["hr", "finance", "it", "compliance", "general"],
    "hr_manager":         ["hr", "general"],
    "finance_analyst":    ["finance", "general"],
    "it_engineer":        ["it", "general"],
    "compliance_officer": ["compliance", "hr", "finance", "it", "general"],
}


USERS: Dict[str, Dict] = {
    "alice": {"role": "admin",              "name": "Alice Johnson",  "dept": "Management"},
    "bob":   {"role": "hr_manager",         "name": "Bob Smith",      "dept": "HR"},
    "carol": {"role": "finance_analyst",    "name": "Carol Davis",    "dept": "Finance"},
    "dave":  {"role": "it_engineer",        "name": "Dave Wilson",    "dept": "IT"},
    "eve":   {"role": "compliance_officer", "name": "Eve Martinez",   "dept": "Compliance"},
}


def get_allowed_categories(username: str) -> List[str]:
    """Return the data categories this user is permitted to access."""
    user = USERS.get(username)
    if not user:
        raise PermissionError(
            f"Access denied: unknown user '{username}'. "
            f"Valid users: {', '.join(USERS)}"
        )
    return ROLES[user["role"]]


def rbac_check(username: str, required_category: str) -> bool:
    """Return True if user has permission for the given category."""
    try:
        return required_category in get_allowed_categories(username)
    except PermissionError:
        return False



def _mkdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def generate_synthetic_data(base: str = DATA_PATH, force: bool = False):
    """
    Creates a realistic synthetic enterprise dataset across 5 silos:
      hr/         → Employee records, HR policies (TXT)
      finance/    → Revenue reports, budget tables (TXT + CSV)
      it/         → System logs, infrastructure report (JSON + TXT)
      compliance/ → Audit trail, compliance policy (JSON + TXT)
      general/    → Company overview (TXT)

    Skips generation if data already exists unless force=True.
    """
    marker = Path(base) / ".generated"
    if marker.exists() and not force:
        return
    random.seed(42)

    # ── HR ────────────────────────────────────────────────────────────────────
    hr_dir = f"{base}/hr"
    _mkdir(hr_dir)

    employees = [
        {"id": "E001", "name": "Alice Johnson",  "dept": "Management", "salary": 120000, "level": "VP",        "manager": "Board"},
        {"id": "E002", "name": "Bob Smith",       "dept": "HR",         "salary":  85000, "level": "Manager",   "manager": "Alice Johnson"},
        {"id": "E003", "name": "Carol Davis",     "dept": "Finance",    "salary":  92000, "level": "Analyst",   "manager": "Alice Johnson"},
        {"id": "E004", "name": "Dave Wilson",     "dept": "IT",         "salary":  98000, "level": "Engineer",  "manager": "Alice Johnson"},
        {"id": "E005", "name": "Eve Martinez",    "dept": "Compliance", "salary":  88000, "level": "Officer",   "manager": "Alice Johnson"},
        {"id": "E006", "name": "Frank Lee",       "dept": "HR",         "salary":  72000, "level": "Associate", "manager": "Bob Smith"},
        {"id": "E007", "name": "Grace Kim",       "dept": "Finance",    "salary":  80000, "level": "Analyst",   "manager": "Carol Davis"},
        {"id": "E008", "name": "Henry Chen",      "dept": "IT",         "salary":  95000, "level": "Senior Eng","manager": "Dave Wilson"},
        {"id": "E009", "name": "Irene Patel",     "dept": "HR",         "salary":  68000, "level": "Associate", "manager": "Bob Smith"},
        {"id": "E010", "name": "James O'Brien",   "dept": "Compliance", "salary":  76000, "level": "Analyst",   "manager": "Eve Martinez"},
    ]

    with open(f"{hr_dir}/employee_records.txt", "w",encoding="utf-8") as f:
        f.write("ACME CORP — EMPLOYEE RECORDS (CONFIDENTIAL — HR ACCESS ONLY)\n")
        f.write("=" * 65 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d')} | Total Employees: {len(employees)}\n\n")
        for e in employees:
            hire = (datetime(2017, 1, 1) + timedelta(days=random.randint(0, 2500))).strftime("%Y-%m-%d")
            perf = random.choice(["Exceeds Expectations", "Meets Expectations", "Meets Expectations", "Exceeds Expectations"])
            f.write(f"Employee ID    : {e['id']}\n")
            f.write(f"Full Name      : {e['name']}\n")
            f.write(f"Department     : {e['dept']}\n")
            f.write(f"Level          : {e['level']}\n")
            f.write(f"Annual Salary  : ${e['salary']:,}\n")
            f.write(f"Manager        : {e['manager']}\n")
            f.write(f"Hire Date      : {hire}\n")
            f.write(f"Employment Type: Full-Time\n")
            f.write(f"Status         : Active\n")
            f.write(f"Last Review    : {perf}\n")
            f.write(f"Benefits       : Health, Dental, Vision, 401k (4% match)\n")
            f.write("-" * 45 + "\n\n")

    with open(f"{hr_dir}/hr_policy.txt", "w",encoding="utf-8") as f:
        f.write("ACME CORP — HR POLICY HANDBOOK v2.5 (CONFIDENTIAL)\n")
        f.write("=" * 65 + "\n")
        f.write("Effective Date: January 1, 2025 | Last Updated: March 15, 2025\n\n")

        f.write("SECTION 1: LEAVE POLICY\n")
        f.write("-" * 30 + "\n")
        f.write("Annual Leave: 20 days per year (pro-rated for new hires).\n")
        f.write("Sick Leave: 10 days per calendar year. Medical certificate required for absences exceeding 3 consecutive days.\n")
        f.write("Maternity Leave: 26 weeks fully paid, followed by optional 13 weeks unpaid.\n")
        f.write("Paternity Leave: 4 weeks fully paid within 3 months of birth.\n")
        f.write("Bereavement Leave: 5 days for immediate family, 2 days for extended family.\n\n")

        f.write("SECTION 2: REMOTE WORK POLICY\n")
        f.write("-" * 30 + "\n")
        f.write("Employees may work remotely up to 3 days per week with manager approval.\n")
        f.write("Core hours for remote workers: 10:00 AM – 3:00 PM local time.\n")
        f.write("IT Security Policy compliance is mandatory for all remote access.\n")
        f.write("Temporary full-remote arrangements (up to 4 weeks/year) require VP approval.\n\n")

        f.write("SECTION 3: PERFORMANCE REVIEW PROCESS\n")
        f.write("-" * 30 + "\n")
        f.write("Bi-annual reviews: June (mid-year check-in) and December (annual review).\n")
        f.write("Rating Scale: Exceeds Expectations | Meets Expectations | Needs Improvement.\n")
        f.write("Salary increments: 0–3% for Meets, 3–8% for Exceeds, 0% for Needs Improvement.\n")
        f.write("Promotion eligibility: Minimum 2 consecutive 'Exceeds Expectations' reviews.\n\n")

        f.write("SECTION 4: COMPENSATION & BENEFITS\n")
        f.write("-" * 30 + "\n")
        f.write("Salary bands are reviewed annually against market benchmarks.\n")
        f.write("Bonus pool: 15% of base salary for VP level; 10% for Manager; 7% for Analyst/Associate.\n")
        f.write("401k: Company matches 4% of employee contribution. Vesting: 3-year cliff.\n")
        f.write("Health benefits: Medical, Dental, Vision fully covered for employee; 50% for dependents.\n\n")

        f.write("SECTION 5: RECRUITMENT & ONBOARDING\n")
        f.write("-" * 30 + "\n")
        f.write("All open positions must be approved by department VP and HR Manager.\n")
        f.write("Interview process: Recruiter screen → Technical/functional round → HR round → Offer.\n")
        f.write("Onboarding duration: 90-day structured program with assigned buddy.\n")
        f.write("Background check and reference check required for all hires.\n\n")

        f.write("SECTION 6: CODE OF CONDUCT\n")
        f.write("-" * 30 + "\n")
        f.write("Zero tolerance policy on harassment, discrimination, and retaliation.\n")
        f.write("Conflicts of interest must be disclosed to HR within 15 days of arising.\n")
        f.write("Disciplinary actions: Verbal warning → Written warning → Suspension → Termination.\n")
        f.write("Whistleblower protections apply for good-faith reporting of policy violations.\n")

    # ── FINANCE ───────────────────────────────────────────────────────────────
    finance_dir = f"{base}/finance"
    _mkdir(finance_dir)

    with open(f"{finance_dir}/quarterly_revenue.csv", "w", newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Quarter", "Year", "Revenue_USD", "COGS_USD", "Gross_Profit_USD",
                    "Operating_Expenses_USD", "Net_Profit_USD", "YoY_Growth_Pct", "Region"])
        records = [
            ("Q1", 2023, 3_400_000, 1_200_000, 2_200_000, 1_800_000,   400_000, 0.0,  "North America"),
            ("Q2", 2023, 3_700_000, 1_300_000, 2_400_000, 1_900_000,   500_000, 0.0,  "North America"),
            ("Q3", 2023, 3_900_000, 1_350_000, 2_550_000, 2_000_000,   550_000, 0.0,  "North America"),
            ("Q4", 2023, 4_100_000, 1_400_000, 2_700_000, 2_100_000,   600_000, 0.0,  "North America"),
            ("Q1", 2024, 4_200_000, 1_450_000, 2_750_000, 2_150_000,   600_000, 23.5, "North America"),
            ("Q2", 2024, 4_800_000, 1_600_000, 3_200_000, 2_300_000,   900_000, 29.7, "North America"),
            ("Q3", 2024, 5_100_000, 1_700_000, 3_400_000, 2_400_000, 1_000_000, 30.8, "North America"),
            ("Q4", 2024, 5_600_000, 1_850_000, 3_750_000, 2_600_000, 1_150_000, 36.6, "North America"),
            ("Q1", 2025, 5_900_000, 1_950_000, 3_950_000, 2_700_000, 1_250_000, 40.5, "North America"),
            ("Q2", 2025, 6_200_000, 2_050_000, 4_150_000, 2_850_000, 1_300_000, 29.2, "North America"),
        ]
        for r in records:
            w.writerow(r)

    with open(f"{finance_dir}/budget_allocation_fy2025.csv", "w", newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Department", "Headcount", "Salary_Budget_USD", "OpEx_Budget_USD",
                    "CapEx_Budget_USD", "Total_Budget_USD", "YTD_Spent_USD",
                    "Remaining_USD", "Utilization_Pct"])
        budgets = [
            ("Management",  5, 600_000,   150_000,  50_000,  800_000,  620_000,  180_000, 77.5),
            ("HR",         12, 950_000,   200_000,  30_000, 1_180_000, 890_000,  290_000, 75.4),
            ("Finance",     8, 720_000,   150_000,  20_000,  890_000,  700_000,  190_000, 78.7),
            ("IT",         20,1_960_000,  400_000, 300_000, 2_660_000,2_100_000, 560_000, 78.9),
            ("Compliance",  7, 616_000,   120_000,  20_000,  756_000,  580_000,  176_000, 76.7),
            ("Sales",      40,3_200_000,  800_000, 100_000, 4_100_000,3_200_000, 900_000, 78.0),
            ("Engineering",35,3_850_000,  500_000, 250_000, 4_600_000,3_500_000,1_100_000, 76.1),
        ]
        for b in budgets:
            w.writerow(b)

    with open(f"{finance_dir}/financial_report_q2_2025.txt", "w",encoding="utf-8") as f:
        f.write("ACME CORP — FINANCIAL REPORT Q2 2025 (CONFIDENTIAL — FINANCE ACCESS ONLY)\n")
        f.write("=" * 65 + "\n")
        f.write("Prepared by: Carol Davis, Finance Analyst\n")
        f.write("Approved by: Alice Johnson, VP Management\n")
        f.write(f"Date: {datetime(2025, 7, 5).strftime('%B %d, %Y')}\n\n")

        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write("Q2 2025 consolidated revenue: $6,200,000 (+29.2% YoY vs Q2 2024 $4,800,000).\n")
        f.write("Net profit: $1,300,000 (margin: 21.0%), up from $900,000 in Q2 2024.\n")
        f.write("Total operating expenses: $2,850,000. EBITDA: $1,680,000.\n\n")

        f.write("REVENUE BREAKDOWN BY PRODUCT LINE\n")
        f.write("-" * 30 + "\n")
        f.write("AcmeFlow (workflow automation): $3,100,000 (50% of total revenue)\n")
        f.write("AcmeInsight (BI dashboards): $1,860,000 (30% of total revenue)\n")
        f.write("AcmeSecure (compliance tools): $1,240,000 (20% of total revenue)\n\n")

        f.write("KEY FINANCIAL HIGHLIGHTS\n")
        f.write("-" * 30 + "\n")
        f.write("1. Sales team exceeded Q2 target of $5.8M by 6.9%.\n")
        f.write("2. IT infrastructure upgrade completed $220K under the $1.2M budget.\n")
        f.write("3. Customer acquisition cost (CAC) reduced by 12% via organic growth channels.\n")
        f.write("4. Annual Recurring Revenue (ARR) stands at $22.4M as of June 2025.\n")
        f.write("5. Net Revenue Retention (NRR): 118%, indicating strong upsell momentum.\n\n")

        f.write("COST ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write("Cost of Goods Sold (COGS): $2,050,000 — gross margin improved to 66.9%.\n")
        f.write("R&D spend: $780,000 (12.6% of revenue), consistent with industry norms.\n")
        f.write("Sales & Marketing: $1,100,000 (17.7% of revenue).\n")
        f.write("G&A: $420,000 (6.8% of revenue).\n\n")

        f.write("FORECAST H2 2025\n")
        f.write("-" * 30 + "\n")
        f.write("H2 2025 revenue projection: $13,500,000 (full year: $25,600,000).\n")
        f.write("New enterprise deals pipeline: $8.2M, 65% close probability.\n")
        f.write("Cost optimization initiative projected to save $500,000 in H2.\n")
        f.write("Series B fundraising round targeting $15M, process begins Q3 2025.\n\n")

        f.write("RISKS\n")
        f.write("-" * 30 + "\n")
        f.write("1. Competitive pressure from larger vendors (ServiceNow, Salesforce).\n")
        f.write("2. Currency risk: 15% of revenue in EUR and GBP.\n")
        f.write("3. Key-person dependency: 3 strategic accounts managed by single AE.\n")

    # ── IT ────────────────────────────────────────────────────────────────────
    it_dir = f"{base}/it"
    _mkdir(it_dir)

    severities = ["INFO", "INFO", "INFO", "WARNING", "WARNING", "ERROR", "CRITICAL"]
    services   = ["auth-service", "payment-api", "data-pipeline", "user-service",
                  "report-engine", "notification-svc", "billing-api"]
    msg_bank = {
        "INFO":     [
            "Service started successfully on port 8080",
            "Health check passed — all dependencies reachable",
            "Cache invalidated and refreshed (TTL: 3600s)",
            "Scheduled job completed: daily_report_generation",
            "New user session created — session_id: sess_a8f2",
            "Database connection pool utilization: 42%",
        ],
        "WARNING":  [
            "High memory usage detected: 87% on prod-server-03",
            "Slow database query detected: 3,218ms (threshold: 500ms)",
            "Retry attempt 2/3 for downstream payment-api",
            "JWT token expiry warning: token expires in 5 minutes",
            "Disk usage at 78% on prod-server-02 — monitor required",
            "API response time degraded: p99 latency = 2.1s",
        ],
        "ERROR":    [
            "Database connection timeout after 30s — PostgreSQL unreachable",
            "Authentication failed: invalid credentials for user carol@acme.com",
            "API rate limit exceeded: 1000 req/min threshold hit on billing-api",
            "NullPointerException in PaymentProcessor.java:line 247",
            "Failed to send email notification: SMTP server returned 550",
            "S3 bucket upload failed: AccessDenied — check IAM permissions",
        ],
        "CRITICAL": [
            "ALERT: payment-api service DOWN — 0/3 instances healthy",
            "ALERT: Data corruption detected in shard-3 of user-service DB",
            "ALERT: Security breach attempt blocked — 247 failed logins from 45.33.12.14",
            "ALERT: Disk space critically low — prod-server-01 at 96% capacity",
            "ALERT: SSL certificate expiring in 3 days — auto-renewal failed",
        ],
    }

    logs = []
    for i in range(60):
        sev = random.choice(severities)
        service = random.choice(services)
        ts = (datetime(2025, 5, 1) + timedelta(hours=i * 2, minutes=random.randint(0, 59)))
        logs.append({
            "log_id":          f"LOG-{3000 + i}",
            "timestamp":       ts.isoformat(),
            "severity":        sev,
            "service":         service,
            "host":            f"prod-server-{random.randint(1, 5):02d}",
            "message":         random.choice(msg_bank[sev]),
            "response_time_ms": random.randint(80, 4500),
            "request_id":      hashlib.md5(f"{i}{sev}".encode()).hexdigest()[:12],
            "environment":     "production",
        })

    with open(f"{it_dir}/system_logs.json", "w",encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    with open(f"{it_dir}/infrastructure_report_2025.txt", "w",encoding="utf-8") as f:
        f.write("ACME CORP — IT INFRASTRUCTURE REPORT 2025 (CONFIDENTIAL — IT ACCESS ONLY)\n")
        f.write("=" * 65 + "\n")
        f.write("Prepared by: Dave Wilson, IT Engineer\n\n")

        f.write("1. SERVER INFRASTRUCTURE\n")
        f.write("-" * 30 + "\n")
        f.write("Production fleet: 5 bare-metal servers (prod-server-01 to prod-server-05).\n")
        f.write("OS: Ubuntu 22.04 LTS. Uptime SLA achieved: 99.97% YTD (target: 99.9%).\n")
        f.write("Specs per server: 64GB RAM, 32-core AMD EPYC, 10TB NVMe SSD RAID-10.\n")
        f.write("Load balancer: HAProxy v2.8 with active-passive failover.\n")
        f.write("Container orchestration: Docker Swarm (migration to K8s planned Q3 2025).\n\n")

        f.write("2. SECURITY INCIDENTS & RESPONSES\n")
        f.write("-" * 30 + "\n")
        f.write("Jan 2025: Brute-force attempt on SSH — 3 IPs blocked via fail2ban. No breach.\n")
        f.write("Feb 2025: Insider threat investigation — ex-contractor credentials revoked. No data exfil.\n")
        f.write("Mar 2025: Phishing campaign targeting Finance team — 12 emails quarantined.\n")
        f.write("         2 employees clicked malicious link; accounts reset within 20 minutes.\n")
        f.write("Apr 2025: Log4Shell vulnerability scan — all systems patched. 0 vulnerable instances.\n")
        f.write("May 2025: DDoS attack on payment-api — mitigated via Cloudflare within 4 minutes.\n")
        f.write("         Peak attack traffic: 120,000 requests/second. Zero customer impact.\n\n")

        f.write("3. NETWORK ARCHITECTURE\n")
        f.write("-" * 30 + "\n")
        f.write("Public-facing: Cloudflare CDN → WAF → Load Balancer.\n")
        f.write("Internal: VLAN-segmented network (VLAN 10: Prod, VLAN 20: Dev, VLAN 30: Mgmt).\n")
        f.write("VPN: Wireguard for remote access; 2FA enforced for all VPN connections.\n")
        f.write("Firewall: iptables with default-deny inbound; only ports 80/443/22 exposed.\n\n")

        f.write("4. PLANNED UPGRADES (H2 2025)\n")
        f.write("-" * 30 + "\n")
        f.write("Q3 2025: Full Kubernetes migration — estimated effort 6 weeks, $180K cost.\n")
        f.write("Q3 2025: Implement centralized SIEM (Elasticsearch + Kibana).\n")
        f.write("Q4 2025: Zero-trust network architecture — microsegmentation of all services.\n")
        f.write("Q4 2025: Hardware refresh for prod-server-01 and prod-server-02 (EOL Dec 2025).\n\n")

        f.write("5. MONITORING & ALERTING\n")
        f.write("-" * 30 + "\n")
        f.write("Tools: Prometheus + Grafana for metrics; PagerDuty for on-call alerting.\n")
        f.write("SLO: API p99 latency < 500ms. Current p99: 312ms. ✓\n")
        f.write("SLO: Error rate < 0.1%. Current error rate: 0.04%. ✓\n")
        f.write("Alert escalation: P1 (15 min) → P2 (1 hr) → P3 (4 hr) → P4 (next business day).\n")

    # ── COMPLIANCE ────────────────────────────────────────────────────────────
    compliance_dir = f"{base}/compliance"
    _mkdir(compliance_dir)

    actions = ["USER_LOGIN", "DOCUMENT_VIEW", "DATA_EXPORT", "PERMISSION_CHANGE",
               "FAILED_LOGIN", "FILE_DOWNLOAD", "SETTING_CHANGE", "ADMIN_ACTION"]
    resources = [
        "employee_records.txt", "quarterly_revenue.csv", "system_logs.json",
        "financial_report_q2_2025.txt", "compliance_policy.txt", "hr_policy.txt",
        "infrastructure_report_2025.txt", "budget_allocation_fy2025.csv"
    ]

    audit_records = []
    for i in range(45):
        user = random.choice(list(USERS.keys()))
        action = random.choice(actions)
        ts = (datetime(2025, 4, 1) + timedelta(hours=i * 4, minutes=random.randint(0, 59)))
        resource = random.choice(resources)
        status = "FAILED" if action == "FAILED_LOGIN" else "SUCCESS"
        audit_records.append({
            "audit_id":    f"AUD-{5000 + i}",
            "timestamp":   ts.isoformat(),
            "user":        user,
            "role":        USERS[user]["role"],
            "department":  USERS[user]["dept"],
            "action":      action,
            "resource":    resource,
            "ip_address":  f"10.{random.randint(0, 5)}.{random.randint(1, 20)}.{random.randint(2, 254)}",
            "user_agent":  random.choice(["Chrome/124", "Firefox/126", "Safari/17", "curl/8.4"]),
            "status":      status,
            "risk_score":  random.randint(0, 30) if status == "SUCCESS" else random.randint(40, 90),
            "geo":         random.choice(["San Francisco, CA", "New York, NY", "Austin, TX", "Remote"]),
        })

    with open(f"{compliance_dir}/audit_trail.json", "w",encoding="utf-8") as f:
        json.dump(audit_records, f, indent=2)

    with open(f"{compliance_dir}/compliance_policy.txt", "w",encoding="utf-8") as f:
        f.write("ACME CORP — COMPLIANCE & REGULATORY POLICY v3.1 (CONFIDENTIAL)\n")
        f.write("=" * 65 + "\n")
        f.write("Effective: January 1, 2025 | Owner: Eve Martinez, Compliance Officer\n\n")

        f.write("SECTION 1: DATA PROTECTION COMPLIANCE (GDPR / CCPA)\n")
        f.write("-" * 40 + "\n")
        f.write("All personal employee and customer data must be encrypted at rest (AES-256)\n")
        f.write("and in transit (TLS 1.3 minimum).\n")
        f.write("Data retention schedules:\n")
        f.write("  - Financial records: 7 years minimum\n")
        f.write("  - HR/employee records: 5 years after separation\n")
        f.write("  - Audit logs: 3 years minimum\n")
        f.write("  - Customer communications: 2 years\n")
        f.write("Data Subject Access Requests (DSARs) must be fulfilled within 30 days.\n")
        f.write("Right to erasure requests must be assessed within 72 hours.\n\n")

        f.write("SECTION 2: ACCESS CONTROL POLICY\n")
        f.write("-" * 40 + "\n")
        f.write("Role-Based Access Control (RBAC) is enforced across all internal systems.\n")
        f.write("Privileged access (admin-level) requires Multi-Factor Authentication (MFA).\n")
        f.write("Access reviews are conducted quarterly — unused accounts deactivated within 30 days.\n")
        f.write("Contractor accounts must be deactivated within 24 hours of engagement end.\n")
        f.write("Password policy: minimum 12 characters, rotation every 90 days, no reuse of last 10.\n\n")

        f.write("SECTION 3: AUDIT REQUIREMENTS\n")
        f.write("-" * 40 + "\n")
        f.write("All system access, data modifications, and permission changes must be logged.\n")
        f.write("Audit logs must be immutable and stored in a tamper-evident system.\n")
        f.write("Internal audit: quarterly by Compliance team.\n")
        f.write("External audit: annual by Big 4 certified auditors (next: November 2025).\n")
        f.write("SOC 2 Type II certification renewal: December 2025.\n\n")

        f.write("SECTION 4: INCIDENT RESPONSE PLAN\n")
        f.write("-" * 40 + "\n")
        f.write("Data breaches must be reported to regulatory authorities within 72 hours (GDPR Art. 33).\n")
        f.write("Incident response team notification: within 1 hour of detection.\n")
        f.write("Containment target: within 4 hours of confirmed breach.\n")
        f.write("Post-incident review required within 5 business days.\n")
        f.write("Incident severity levels: P1 (Critical) → P2 (High) → P3 (Medium) → P4 (Low).\n\n")

        f.write("SECTION 5: VENDOR & THIRD-PARTY RISK\n")
        f.write("-" * 40 + "\n")
        f.write("All vendors with data access require signed Data Processing Agreement (DPA).\n")
        f.write("Annual security questionnaire mandatory for Tier 1 vendors.\n")
        f.write("Vendor access limited to minimum necessary data (principle of least privilege).\n")
        f.write("Third-party penetration test required for any new critical vendor integrations.\n")


    general_dir = f"{base}/general"
    _mkdir(general_dir)

    with open(f"{general_dir}/company_overview.txt", "w",encoding="utf-8") as f:
        f.write("ACME CORP — COMPANY OVERVIEW & MISSION\n")
        f.write("=" * 65 + "\n")
        f.write("Founded: 2010 | Headquarters: 555 Market Street, San Francisco, CA 94105\n")
        f.write("Total Employees: ~127 globally (as of June 2025)\n")
        f.write("Annual Revenue FY2024: $20,400,000 | ARR June 2025: $22,400,000\n")
        f.write("Investors: Series A ($6M) — Sequoia Capital, Andreessen Horowitz (2019)\n\n")

        f.write("MISSION STATEMENT\n")
        f.write("-" * 30 + "\n")
        f.write("Delivering secure, intelligent enterprise workflow automation that empowers\n")
        f.write("organizations to operate with clarity, compliance, and competitive advantage.\n\n")

        f.write("CORE PRODUCTS\n")
        f.write("-" * 30 + "\n")
        f.write("1. AcmeFlow (Enterprise Workflow Automation)\n")
        f.write("   - No-code/low-code workflow builder\n")
        f.write("   - 200+ pre-built enterprise integrations (Salesforce, SAP, Workday)\n")
        f.write("   - 850 active enterprise customers | NPS: 67\n\n")
        f.write("2. AcmeInsight (Business Intelligence Dashboard)\n")
        f.write("   - Real-time data visualization and reporting\n")
        f.write("   - AI-powered anomaly detection and forecasting\n")
        f.write("   - 420 active customers | NPS: 72\n\n")
        f.write("3. AcmeSecure (Compliance & Audit Management)\n")
        f.write("   - Automated compliance monitoring for SOC2, GDPR, HIPAA\n")
        f.write("   - Audit trail management and risk scoring\n")
        f.write("   - 310 active customers | NPS: 79\n\n")

        f.write("ORGANIZATIONAL STRUCTURE\n")
        f.write("-" * 30 + "\n")
        f.write("Executive Team: Alice Johnson (VP/Acting CEO), Board of Directors (5 members)\n")
        f.write("Product & Engineering (55 employees): Builds and maintains all three products\n")
        f.write("Sales & Marketing (40 employees): Customer acquisition, retention, partnerships\n")
        f.write("Finance (8 employees): Accounting, FP&A, investor relations\n")
        f.write("HR (12 employees): Talent acquisition, L&D, employee experience\n")
        f.write("Compliance & Legal (7 employees): Regulatory adherence, risk, contract review\n")
        f.write("IT (20 employees): Infrastructure, security, DevOps, internal tools\n\n")

        f.write("STRATEGIC PRIORITIES 2025\n")
        f.write("-" * 30 + "\n")
        f.write("1. Raise Series B ($15M target) by Q4 2025 to fund EMEA expansion.\n")
        f.write("2. Launch AcmeFlow 3.0 with AI-native workflow generation (Q3 2025).\n")
        f.write("3. Achieve SOC 2 Type II recertification by December 2025.\n")
        f.write("4. Grow ARR to $30M by end of 2025 via upsell + new enterprise wins.\n")
        f.write("5. Open London office (EMEA HQ) by Q1 2026.\n")

    marker.touch()



def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_csv_as_text(path: str) -> str:

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return ""

    headers = list(rows[0].keys())
    stem = Path(path).stem.replace("_", " ").title()
    lines = [
        f"Table: {stem}",
        f"Columns: {', '.join(headers)}",
        f"Total records: {len(rows)}",
        "",
    ]
    for row in rows:
        # Build a natural language sentence per row
        parts = [f"{k.replace('_', ' ')}: {v}" for k, v in row.items() if v]
        lines.append(". ".join(parts) + ".")
    return "\n".join(lines)


def load_json_as_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        lines = [f"Log entries: {len(data)} total records", ""]
        for entry in data[:200]:  # cap to avoid oversized context
            if isinstance(entry, dict):
                parts = [f"{k}: {v}" for k, v in entry.items()]
                lines.append(" | ".join(parts))
            else:
                lines.append(str(entry))
        return "\n".join(lines)

    elif isinstance(data, dict):
        return json.dumps(data, indent=2)

    return str(data)


def load_pdf(path: str) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    except ImportError:
        return load_txt(path)
    except Exception as e:
        return ""


def load_document(path: str) -> Optional[str]:
    ext = Path(path).suffix.lower()
    loaders = {
        ".txt":  load_txt,
        ".csv":  load_csv_as_text,
        ".json": load_json_as_text,
        ".pdf":  load_pdf,
    }
    loader = loaders.get(ext)
    if not loader:
        return None
    try:
        return loader(path)
    except Exception as e:
        return None



def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping word-level chunks.
    Overlap ensures cross-sentence context is preserved.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


class EnterpriseVectorStore:
    def __init__(self, persist_path: str = CHROMA_PATH):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        self.chroma = chromadb.PersistentClient(path=persist_path)
        self.collection = self.chroma.get_or_create_collection(
            name="enterprise_docs",
            metadata={"hnsw:space": "cosine"},
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.encode(texts, show_progress_bar=False).tolist()

    def add_documents(
        self,
        chunks: List[str],
        metadatas: List[Dict],
        ids: List[str],
    ) -> int:
        try:
            existing_result = self.collection.get(ids=ids, include=[])
            existing_ids = set(existing_result["ids"])
        except Exception:
            existing_ids = set()

        new_ids, new_chunks, new_meta = [], [], []
        for cid, chunk, meta in zip(ids, chunks, metadatas):
            if cid not in existing_ids:
                new_ids.append(cid)
                new_chunks.append(chunk)
                new_meta.append(meta)

        if not new_ids:
            return 0

        embeddings = self.embed(new_chunks)
        self.collection.add(
            ids=new_ids,
            documents=new_chunks,
            embeddings=embeddings,
            metadatas=new_meta,
        )
        return len(new_ids)

    def semantic_search(
        self,
        query: str,
        allowed_categories: List[str],
        top_k: int = TOP_K,
    ) -> List[Dict]:
        """
        Search the collection using cosine similarity.
        Hard filters on `category` field to enforce RBAC.
        """
        total = self.collection.count()
        if total == 0:
            return []

        n_results = min(top_k * 3, total)
        q_embed = self.embed([query])[0]

        # ChromaDB `where` filter — only return docs the user may access
        where_filter: Dict = {"category": {"$in": allowed_categories}}

        try:
            results = self.collection.query(
                query_embeddings=[q_embed],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            return []

        hits = []
        docs      = results["documents"][0]
        dists     = results["distances"][0]
        metas     = results["metadatas"][0]

        for doc, dist, meta in zip(docs, dists, metas):
            similarity = max(0.0, 1.0 - dist)  # cosine distance → similarity
            hits.append({
                "text":      doc,
                "source":    meta.get("source", "unknown"),
                "category":  meta.get("category", "general"),
                "file_type": meta.get("file_type", ""),
                "chunk_id":  meta.get("chunk_id", 0),
                "similarity": round(similarity, 4),
                "score":      round(similarity, 4),
            })
        return hits

    def total_chunks(self) -> int:
        return self.collection.count()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — HYBRID RETRIEVAL (Semantic + Keyword BM25-style)
# ══════════════════════════════════════════════════════════════════════════════

def _keyword_score(query: str, text: str) -> float:
    """
    Lightweight term-frequency based keyword matching.
    Normalised by log of document length to avoid length bias.
    """
    query_terms = set(re.findall(r"\b\w{3,}\b", query.lower()))
    doc_words   = re.findall(r"\b\w{3,}\b", text.lower())
    if not doc_words or not query_terms:
        return 0.0
    match_count = sum(1 for w in doc_words if w in query_terms)
    return match_count / (1.0 + math.log(1 + len(doc_words)))


def hybrid_search(
    query: str,
    allowed_categories: List[str],
    store: EnterpriseVectorStore,
    top_k: int = TOP_K,
    alpha: float = HYBRID_ALPHA,
) -> List[Dict]:
    """
    Combines:
      - Semantic score (cosine similarity via ChromaDB)
      - Keyword score (TF-style term matching)
    Final score = alpha * semantic + (1-alpha) * keyword

    alpha=1.0 → pure semantic | alpha=0.0 → pure keyword
    """
    candidates = store.semantic_search(query, allowed_categories, top_k=top_k * 4)

    for hit in candidates:
        kw = _keyword_score(query, hit["text"])
        hit["keyword_score"] = round(kw, 4)
        hit["score"] = round(
            alpha * hit["similarity"] + (1.0 - alpha) * kw, 4
        )

    # Sort by combined score, filter below threshold
    candidates.sort(key=lambda x: x["score"], reverse=True)
    qualified = [h for h in candidates if h["score"] >= CONFIDENCE_THRESH]
    return qualified[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — QUERY ROUTING
# ══════════════════════════════════════════════════════════════════════════════

ROUTE_KEYWORDS: Dict[str, List[str]] = {
    "hr": [
        "employee", "staff", "hire", "fired", "salary", "payroll", "leave",
        "vacation", "sick", "performance", "review", "promotion", "hr",
        "onboard", "headcount", "bonus", "benefits", "recruit", "conduct",
        "handbook", "policy", "termination", "resign",
    ],
    "finance": [
        "revenue", "budget", "profit", "expense", "cost", "financial",
        "quarter", "fiscal", "invoice", "forecast", "cash", "ebitda",
        "margin", "arr", "mrr", "spend", "allocation", "accounting",
        "report", "balance", "fund", "investment", "series",
    ],
    "it": [
        "server", "log", "error", "system", "security", "incident", "database",
        "api", "network", "cpu", "memory", "disk", "crash", "latency",
        "service", "outage", "breach", "firewall", "kubernetes", "deploy",
        "infrastructure", "monitoring", "alert", "ssl", "patch",
    ],
    "compliance": [
        "audit", "gdpr", "ccpa", "regulation", "policy", "compliance",
        "risk", "legal", "data protection", "access control", "incident",
        "violation", "report", "soc2", "hipaa", "dpa", "retention",
        "investigation", "penalty", "breach", "mfa", "rbac",
    ],
    "general": [
        "company", "product", "overview", "mission", "division", "acme",
        "office", "team", "strategy", "goal", "vision", "investor",
        "customer", "roadmap", "market", "competitor",
    ],
}


def route_query(query: str) -> List[str]:
    """
    Score each category against query keywords.
    Returns categories with at least 1 keyword match.
    Falls back to all categories if no keywords match.
    """
    q_lower = query.lower()
    scores: Dict[str, int] = {cat: 0 for cat in ROUTE_KEYWORDS}

    for cat, keywords in ROUTE_KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower:
                scores[cat] += 1

    # Rank by score
    scored = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    routed = [cat for cat, sc in scored if sc > 0]
    return routed if routed else list(ROUTE_KEYWORDS.keys())  # fallback = all


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — PROMPT ENGINEERING & OLLAMA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _build_context_block(chunks: List[Dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        header = (
            f"[Source {i}] "
            f"FILE: {c['source']} | "
            f"CATEGORY: {c['category'].upper()} | "
            f"TYPE: {c['file_type'].upper()} | "
            f"RELEVANCE: {c['score']:.3f}"
        )
        parts.append(f"{header}\n{c['text']}")
    return "\n\n" + ("─" * 50 + "\n\n").join(parts)


def build_prompt(
    query: str,
    chunks: List[Dict],
    user_name: str,
    user_role: str,
) -> str:
    context = _build_context_block(chunks)
    return f"""You are a secure enterprise AI assistant for ACME Corp.
The authenticated user is '{user_name}' with role '{user_role}'.

STRICT INSTRUCTIONS:
1. Answer ONLY using information present in the CONTEXT below.
2. Cite every claim using [Source N] notation where N matches the context source number.
3. If the context does not contain enough information, respond with:
   "The documents accessible to your role ({user_role}) do not contain sufficient information to answer this question."
4. NEVER reveal information from sources not present in the context.
5. NEVER invent or speculate beyond what the context states.
6. Be concise and factual. Avoid filler phrases.
7. If multiple sources agree, cite all relevant ones.
8. Structure your answer clearly. Use bullet points if listing multiple items.

RETRIEVED CONTEXT:
{context}

USER QUESTION: {query}

ANSWER (use [Source N] citations):"""


def generate_with_ollama(prompt: str) -> str:
    """Call Ollama qwen2.5 for answer generation."""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.05,   # low temp for factual grounding
                "num_predict": 600,    # max tokens in response
                "top_p": 0.9,
            },
        )
        return response["message"]["content"].strip()
    except ImportError:
        return "[ERROR] ollama package missing."
    except Exception as e:
        return (
            f"[ERROR] Could not reach Ollama. Is it running?\n"
            f"  Try: ollama serve\n"
            f"  Then: ollama pull {OLLAMA_MODEL}\n"
            f"  Error details: {e}"
        )


def generate_response(
    query: str,
    chunks: List[Dict],
    username: str,
) -> Dict:
    """Build prompt, call Ollama, package result with citations."""
    user   = USERS[username]
    prompt = build_prompt(query, chunks, user["name"], user["role"])
    answer = generate_with_ollama(prompt)

    citations = []
    for i, c in enumerate(chunks, 1):
        citations.append({
            "ref":           f"[Source {i}]",
            "source":        c["source"],
            "category":      c["category"],
            "file_type":     c["file_type"],
            "relevance":     c["score"],
            "semantic_sim":  c.get("similarity", 0),
            "keyword_score": c.get("keyword_score", 0),
        })

    return {"answer": answer, "citations": citations, "prompt_length": len(prompt)}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — INGESTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

# Maps subdirectory name → RBAC category label
DIR_TO_CATEGORY: Dict[str, str] = {
    "hr":         "hr",
    "finance":    "finance",
    "it":         "it",
    "compliance": "compliance",
    "general":    "general",
}

SUPPORTED_EXTENSIONS = {".txt", ".csv", ".json", ".pdf"}


def ingest_all(
    store: EnterpriseVectorStore,
    base: str = DATA_PATH,
    force: bool = False,
) -> int:
    """
    Walk enterprise_data/ and ingest all documents into ChromaDB.
    Uses SHA-based IDs to avoid duplicate ingestion.
    Returns total number of new chunks added.
    """
    total_added = 0

    for category, label in DIR_TO_CATEGORY.items():
        cat_dir = Path(base) / category
        if not cat_dir.exists():
            print(f"Directory not found, skipping: {cat_dir}")
            continue

        files = [f for f in cat_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not files:
            continue


        for filepath in files:
            text = load_document(str(filepath))
            if not text or not text.strip():
                print(f"  ↳ Empty or failed: {filepath.name}")
                continue

            chunks = chunk_text(text)
            if not chunks:
                continue

            ids, metas = [], []
            for idx, chunk in enumerate(chunks):
                # Deterministic ID from filename + chunk index
                raw_id = f"{filepath.name}::{idx}"
                chunk_id = hashlib.sha256(raw_id.encode()).hexdigest()[:32]
                ids.append(chunk_id)
                metas.append({
                    "source":    filepath.name,
                    "category":  label,
                    "file_type": filepath.suffix.lstrip("."),
                    "chunk_id":  idx,
                    "path":      str(filepath),
                })

            added = store.add_documents(chunks, metas, ids)
            total_added += added

            status = f"{added} new" if added > 0 else "all cached"

    print(f"Ingestion complete. New chunks added: {total_added}")
    print(f"Total chunks in vector store: {store.total_chunks()}")
    return total_added


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — MAIN RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class EnterpriseRAG:
    """
    Top-level orchestrator for the enterprise RAG system.

    Flow per query:
      1. RBAC check → get allowed categories for user
      2. Query routing → identify which silos are relevant
      3. Intersection of (routed silos) ∩ (allowed silos)
      4. Hybrid search on narrowed scope
      5. Prompt construction with retrieved context
      6. Ollama qwen2.5 generation
      7. Return structured result with citations + metadata
    """

    def __init__(self, chroma_path: str = CHROMA_PATH, data_path: str = DATA_PATH):
        self.data_path   = data_path
        self.chroma_path = chroma_path
        self.store: Optional[EnterpriseVectorStore] = None

    def setup(self, force_regenerate: bool = False, force_reingest: bool = False):
        """One-time setup: generate synthetic data + ingest into ChromaDB."""
        generate_synthetic_data(base=self.data_path, force=force_regenerate)
        self.store = EnterpriseVectorStore(persist_path=self.chroma_path)
        ingest_all(self.store, base=self.data_path, force=force_reingest)
        print(f"Enterprise RAG ready. {self.store.total_chunks()} chunks indexed.")

    def query(
        self,
        question: str,
        username: str,
        top_k: int = TOP_K,
    ) -> Dict:
        """
        Execute a full RAG query on behalf of a user.

        Returns a dict with:
          answer       : generated response with [Source N] citations
          citations    : list of sources used, with relevance scores
          user         : username
          role         : user role
          allowed      : categories user can access
          routed_to    : categories the query was routed to
          effective    : final intersection used for search
          chunks_used  : number of context chunks fed to LLM
          access_denied: True if RBAC blocked the query
        """
        assert self.store is not None, "Call setup() first."

        # ── Step 1: RBAC ──────────────────────────────────────────────────────
        try:
            allowed = get_allowed_categories(username)
        except PermissionError as e:
            return {
                "answer":       str(e),
                "citations":    [],
                "user":         username,
                "role":         "unknown",
                "allowed":      [],
                "access_denied": True,
            }

        user_info = USERS[username]

        # ── Step 2: Route query to relevant silos ─────────────────────────────
        routed = route_query(question)

        # ── Step 3: Intersect routed with allowed (RBAC enforcement) ──────────
        effective = [c for c in routed if c in allowed]
        if not effective:
            # All routed categories are blocked → fall back to all allowed
            effective = allowed
        else:
            print(f"  Effective search scope: {effective}")

        # ── Step 4: Hybrid retrieval ──────────────────────────────────────────
        chunks = hybrid_search(question, effective, self.store, top_k=top_k)

        if not chunks:
            return {
                "answer": (
                    f"No relevant documents found within your access scope ({effective}). "
                    "Try rephrasing or ask about a different topic."
                ),
                "citations":   [],
                "user":        username,
                "role":        user_info["role"],
                "allowed":     allowed,
                "routed_to":   routed,
                "effective":   effective,
                "chunks_used": 0,
                "access_denied": False,
            }

        print(f"  Retrieved {len(chunks)} chunks (top score: {chunks[0]['score']:.3f})")

        # ── Step 5+6: Generate answer ─────────────────────────────────────────
        result = generate_response(question, chunks, username)

        return {
            "answer":      result["answer"],
            "citations":   result["citations"],
            "user":        username,
            "role":        user_info["role"],
            "allowed":     allowed,
            "routed_to":   routed,
            "effective":   effective,
            "chunks_used": len(chunks),
            "access_denied": False,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — DEMO QUERIES
# ══════════════════════════════════════════════════════════════════════════════

DEMO_QUERIES = [
    # (username, question, description)
    ("alice",  "What is ACME Corp's Q2 2025 revenue and net profit?",
     "Admin querying finance data"),
    ("carol",  "What was the total budget allocated to IT department in FY2025?",
     "Finance analyst querying finance data"),
    ("bob",    "What is the company's maternity leave policy?",
     "HR manager querying HR data"),
    ("dave",   "Were there any CRITICAL log events in the system logs? Summarize them.",
     "IT engineer querying IT logs"),
    ("eve",    "What are the data retention requirements according to compliance policy?",
     "Compliance officer querying compliance data"),
    ("bob",    "What were the Q2 2025 financial results?",
     "HR manager trying to access finance data (RBAC block expected)"),
    ("carol",  "What are the employee salary details?",
     "Finance analyst trying to access HR data (RBAC block expected)"),
    ("alice",  "Summarize all security incidents in 2025 from IT reports and compliance audit logs.",
     "Admin doing cross-silo query"),
]


def run_demo(rag: EnterpriseRAG):
    print("\n" + "═" * 70)
    print("  DEMO MODE — Running predefined queries across all user roles")
    print("═" * 70)

    for i, (username, question, desc) in enumerate(DEMO_QUERIES, 1):
        print(f"\n{'─' * 70}")
        print(f"[DEMO {i}/{len(DEMO_QUERIES)}] {desc}")
        print(f"User : {username} ({USERS[username]['role']})")
        print(f"Query: {question}")
        print("─" * 70)

        result = rag.query(question, username)
        _print_result(result)

        if i < len(DEMO_QUERIES):
            input("\nPress ENTER for next query…")



BANNER = """
╔══════════════════════════════════════════════════════════════╗
║       ACME CORP — Enterprise RAG Intelligence System         ║
║       LLM: Ollama qwen2.5  |  DB: ChromaDB  |  RBAC: On     ║
╚══════════════════════════════════════════════════════════════╝
"""

CLI_HELP = """
Commands:
  <question>   Ask anything — RBAC is enforced automatically
  switch       Change the active user
  whoami       Show current user and permissions
  stats        Show vector store statistics
  help         Show this help
  exit / quit  Exit the system
"""


def _print_result(result: Dict):
    if result.get("access_denied"):
        print(f"\n🚫 ACCESS DENIED: {result['answer']}")
        return

    print(f"\n{'─' * 60}")
    print(f"USER   : {result.get('user')} ({result.get('role')})")
    print(f"SCOPE  : {result.get('effective', result.get('routed_to', []))}")
    print(f"CHUNKS : {result.get('chunks_used', 0)} retrieved")
    print(f"{'─' * 60}")
    print("ANSWER:")
    print(result.get("answer", "No answer generated."))
    print(f"\n{'─' * 60}")
    print("SOURCES:")
    for c in result.get("citations", []):
        bar = "█" * int(c["relevance"] * 20)
        print(
            f"  {c['ref']:12s} {c['source']:<45s} "
            f"[{c['category'].upper():<12s}] "
            f"score={c['relevance']:.3f} {bar}"
        )
    print(f"{'─' * 60}\n")


def _select_user(prompt_text: str = "Login as") -> str:
    print(f"\nAvailable users:")
    for uname, udata in USERS.items():
        cats = ROLES[udata["role"]]
        print(f"  {uname:<10} → {udata['role']:<22} | Access: {', '.join(cats)}")
    while True:
        choice = input(f"\n{prompt_text}: ").strip().lower()
        if choice in USERS:
            return choice
        print(f"  Invalid. Choose from: {', '.join(USERS.keys())}")


def interactive_cli(rag: EnterpriseRAG, start_user: Optional[str] = None):
    print(BANNER)
    print(CLI_HELP)

    username = start_user if start_user and start_user in USERS else _select_user()
    user     = USERS[username]
    print(f"\n✓ Logged in as {user['name']} ({user['role']})")
    print(f"  Access scope: {get_allowed_categories(username)}")

    while True:
        try:
            raw = input(f"\n[{username}] Ask > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not raw:
            continue

        cmd = raw.lower()

        if cmd in ("exit", "quit"):
            print("Goodbye.")
            break

        elif cmd == "switch":
            username = _select_user("Switch to")
            user     = USERS[username]
            print(f"✓ Now operating as {user['name']} ({user['role']})")
            print(f"  Access scope: {get_allowed_categories(username)}")

        elif cmd == "whoami":
            allowed = get_allowed_categories(username)
            print(f"\nUser      : {user['name']} ({username})")
            print(f"Role      : {user['role']}")
            print(f"Department: {user['dept']}")
            print(f"Access    : {', '.join(allowed)}")

        elif cmd == "stats":
            total = rag.store.total_chunks() if rag.store else 0
            print(f"\nVector store: {total} chunks indexed")
            print(f"LLM         : Ollama {OLLAMA_MODEL}")
            print(f"Embedder    : {EMBEDDING_MODEL}")
            print(f"ChunkSize   : {CHUNK_SIZE} words (overlap: {CHUNK_OVERLAP})")
            print(f"Top-K       : {TOP_K}")
            print(f"Alpha       : {HYBRID_ALPHA} (semantic/keyword balance)")

        elif cmd == "help":
            print(CLI_HELP)

        else:
            result = rag.query(raw, username)
            _print_result(result)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enterprise RAG System — Ollama qwen2.5 + ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--demo",       action="store_true", help="Run demo queries across all roles")
    parser.add_argument("--reingest",   action="store_true", help="Force re-ingest all documents")
    parser.add_argument("--regenerate", action="store_true", help="Force regenerate synthetic data")
    parser.add_argument("--user",       type=str, default=None, help="Start as a specific user (e.g. --user bob)")
    parser.add_argument("--query",      type=str, default=None, help="Run a single query non-interactively")
    return parser.parse_args()


def main():
    args = parse_args()

    rag = EnterpriseRAG()
    rag.setup(
        force_regenerate=args.regenerate,
        force_reingest=args.reingest,
    )

    if args.demo:
        run_demo(rag)
        return

    if args.query:
        username = args.user or "alice"
        if username not in USERS:
            print(f"Unknown user: {username}. Valid: {', '.join(USERS.keys())}")
            sys.exit(1)
        result = rag.query(args.query, username)
        _print_result(result)
        return

    interactive_cli(rag, start_user=args.user)


if __name__ == "__main__":
    main()
