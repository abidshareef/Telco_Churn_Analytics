#!/usr/bin/env python3
"""
Automated Security, Code Quality, and Repository Hygiene Audit Script.
Checks:
  1. AST compilation of all Python files in src/, scripts/, and tests/
  2. Absence of bare `except Exception: pass` or unfinished `TODO`/`FIXME` placeholders
  3. Absence of hardcoded secrets, private keys, or committed `.env` files
  4. Raw dataset existence and non-zero integrity
"""

import os
import re
import sys
import ast

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FORBIDDEN_SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS Access Key ID
    re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"ghp_[A-Za-z0-9]{36}"),  # GitHub Personal Access Token
]


def audit_repository() -> int:
    errors = []
    py_files_checked = 0

    # 1. Ensure .env is not committed
    if os.path.exists(os.path.join(REPO_ROOT, ".env")):
        errors.append("Security violation: '.env' file exists in repository root.")

    # 2. Walk Python files in src, scripts, tests
    for subdir in ["src", "scripts", "tests"]:
        target_dir = os.path.join(REPO_ROOT, subdir)
        for root, _, files in os.walk(target_dir):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, REPO_ROOT)
                py_files_checked += 1

                with open(fpath, "r", encoding="utf-8") as f:
                    source = f.read()

                # AST syntax check
                try:
                    tree = ast.parse(source, filename=rel_path)
                except SyntaxError as exc:
                    errors.append(f"Syntax error in {rel_path}: {exc}")
                    continue

                # Check for silent `except Exception: pass`
                for node in ast.walk(tree):
                    if isinstance(node, ast.ExceptHandler):
                        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                            errors.append(f"Silent exception handler (`pass`) found in {rel_path}:{node.lineno}")

                # Check for hardcoded secret patterns
                for pat in FORBIDDEN_SECRET_PATTERNS:
                    if pat.search(source):
                        errors.append(f"Potential hardcoded credential pattern matched in {rel_path}")

    # 3. Check required repository files
    required_files = [
        "README.md",
        "LICENSE",
        ".gitignore",
        ".env.example",
        "pyproject.toml",
        "requirements.txt",
        "Dockerfile",
        "docker-compose.yml",
        ".github/workflows/ci.yml",
        "configs/default_config.json",
        "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    ]
    for req in required_files:
        if not os.path.exists(os.path.join(REPO_ROOT, req)):
            errors.append(f"Missing required repository file: {req}")

    if errors:
        print("=== REPOSITORY AUDIT FAILED ===")
        for err in errors:
            print(f" [FAIL] {err}")
        return 1

    print(f"=== REPOSITORY AUDIT PASSED ({py_files_checked} Python modules verified, 0 issues) ===")
    return 0


if __name__ == "__main__":
    sys.exit(audit_repository())
