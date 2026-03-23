# -*- coding: utf-8 -*-
"""
migrate_sessions.py — Import all prior project_output sessions into memory_layer

Scans project_output/ for existing session files (by date) and populates:
  - SQLite sessions table (via save_session)
  - SQLite agent_memory table (via save_agent_memory)
  - ChromaDB vector index (background, via _index_to_chroma)
  - GitHub private repo (via github_push_async)

Usage:
    python scripts/migrate_sessions.py
    python scripts/migrate_sessions.py --dry-run    # preview only, no writes
    python scripts/migrate_sessions.py --no-github  # skip GitHub push
"""

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_OUTPUT = Path("./project_output")
WORKSPACE      = Path(__file__).parent.parent

# ── Member name → display name mapping ───────────────────────
MEMBER_NAMES = {
    "darvin":   "Darvin",
    "felipe":   "Felipe",
    "felix":    "Felix",
    "finman":   "Finman",
    "ilya":     "Ilya",
    "jose":     "Jose",
    "marvin":   "Marvin",
    "schwertz": "Schwertz",
    "viktor":   "Viktor",
}


def _read(path: Path) -> str:
    """Read file, return empty string on error."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _find(date_str: str, prefix: str) -> str:
    """Find and read the best matching file for a given date + prefix."""
    pattern = f"{prefix}*{date_str}*.md"
    matches = sorted(PROJECT_OUTPUT.glob(pattern))
    if not matches:
        return ""
    return _read(matches[-1])   # use latest (highest revision)


def _find_agenda(date_str: str) -> str:
    """Find agenda archive for this date."""
    # Try exact date match e.g. agenda[Fin_Version_2026-03-21].md
    for p in WORKSPACE.glob(f"agenda*{date_str}*.md"):
        text = _read(p)
        if text.strip():
            return text
    return ""


def _get_session_dates() -> list[str]:
    """Derive session dates from final_report files."""
    dates = set()
    for f in PROJECT_OUTPUT.glob("final_report-Demis-Executive-*.md"):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", f.name)
        if m:
            dates.add(m.group(1))
    # Also check Result files for any extra dates
    for f in PROJECT_OUTPUT.glob("*_Result.md"):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", f.name)
        if m:
            dates.add(m.group(1))
    return sorted(dates)


def _get_member_best_result(member_key: str, date_str: str) -> tuple[str, str]:
    """Return (task_summary, result_text) for member's best (last approved) result."""
    # Try revision_2 → revision_1 → initial
    for attempt in ("revision_2", "revision_1", "initial"):
        text = _find(date_str, f"{member_key}_{attempt}")
        if text.strip():
            # Extract task from first non-empty line
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            task_hint = lines[0][:150] if lines else f"{member_key} task on {date_str}"
            return task_hint, text
    return "", ""


def migrate(dry_run: bool = False, no_github: bool = False) -> None:
    from memory_layer import (
        save_session, save_agent_memory, github_push_async,
        get_recent_sessions,
    )

    dates = _get_session_dates()
    if not dates:
        print("No session files found in project_output/")
        return

    print(f"Found {len(dates)} session date(s): {', '.join(dates)}\n")

    for i, date_str in enumerate(dates, 1):
        session_id = f"migrated-{date_str}-01"
        print(f"[{i}/{len(dates)}] Processing session {date_str} → {session_id}")

        # ── Core session files ────────────────────────────────
        agenda       = _find_agenda(date_str)
        plan         = _find(date_str, "demis_strategy_plan")
        final_report = _find(date_str, "final_report-Demis-Executive")
        risk_debate  = _find(date_str, "risk_debate_report")
        adversarial  = _find(date_str, "pre_council_adversarial_debate")
        alpha_sprint = _find(date_str, "sprint_completion_team_alpha")
        beta_sprint  = _find(date_str, "sprint_completion_team_beta")
        cto_report   = _find(date_str, "cto_validation_report")

        if not final_report:
            print(f"  [SKIP] No final_report found for {date_str}")
            continue

        # Extract risk verdict from risk debate
        risk_verdict = ""
        if risk_debate:
            m = re.search(r"VERDICT[:\s]+(INCREASE|MAINTAIN|REDUCE)", risk_debate, re.I)
            if m:
                risk_verdict = m.group(1).upper()

        # Extract key disputes from adversarial debate
        disputes = ""
        if adversarial:
            disputes = adversarial[:500]

        sprint_results = {
            "alpha": alpha_sprint[:300] if alpha_sprint else "",
            "beta":  beta_sprint[:300]  if beta_sprint  else "",
            "cto":   cto_report[:300]   if cto_report   else "",
        }

        print(f"  agenda: {len(agenda)} chars | plan: {len(plan)} chars | report: {len(final_report)} chars")
        print(f"  risk_verdict: {risk_verdict or '(none)'} | disputes: {len(disputes)} chars")

        if not dry_run:
            save_session(
                session_id   = session_id,
                agenda       = agenda or f"Session {date_str}",
                plan         = plan,
                final_report = final_report,
                sprint_results = sprint_results,
                disputes     = disputes,
                risk_verdict = risk_verdict,
            )
            print(f"  [OK] Session saved to SQLite")

            if not no_github and final_report:
                github_push_async(session_id, final_report)
                print(f"  [OK] GitHub push queued")
        else:
            print(f"  [DRY-RUN] Would save session {session_id}")

        # ── Member results ────────────────────────────────────
        member_count = 0
        for key, name in MEMBER_NAMES.items():
            task_hint, result = _get_member_best_result(key, date_str)
            if not result:
                continue
            member_count += 1
            if not dry_run:
                save_agent_memory(
                    agent_name = name,
                    task       = task_hint or f"Sprint task on {date_str}",
                    result     = result[:500],
                    session_id = session_id,
                )
            else:
                print(f"  [DRY-RUN] Would save {name}: {task_hint[:60]}")

        print(f"  [OK] {member_count} member results saved\n")

    if not dry_run:
        # Brief wait for background ChromaDB indexing threads
        print("Waiting 8s for background ChromaDB indexing...")
        time.sleep(8)

        # Verify
        sessions = get_recent_sessions(10)
        print(f"\nVerification: {len(sessions)} sessions now in memory.db")
        for s in sessions:
            print(f"  {s['id']} | {s['date']} | risk={s['risk_verdict'] or '-'}")

    print("\nMigration complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate prior sessions into memory_layer")
    parser.add_argument("--dry-run",   action="store_true", help="Preview only, no writes")
    parser.add_argument("--no-github", action="store_true", help="Skip GitHub push")
    args = parser.parse_args()

    migrate(dry_run=args.dry_run, no_github=args.no_github)
