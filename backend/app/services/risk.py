"""Risk and severity of each detected object for an aircraft, from severity.json (single source of truth)."""

import json
from pathlib import Path

from app.vision.types import UNKNOWN_KEY, Risk

SEVERITY: dict = json.loads(Path(__file__).with_name("severity.json").read_text(encoding="utf-8"))

RISK_ORDER: dict[str, int] = {"critic": 0, "mediu": 1, "de verificat": 2}


def severity(class_key: str, size_cm: float | None = None, unknown_critical_cm: float = 15.0) -> dict | None:
    """Level 1-5, its name and action, the hazards in words and why; None for a class not in the table."""
    if class_key == UNKNOWN_KEY:
        big = size_cm is not None and size_cm >= unknown_critical_cm
        entry = SEVERITY["obiect_necunoscut"]["mare" if big else "mic"]
    else:
        entry = SEVERITY["obiecte"].get(class_key)
        if entry is None:
            return None
    level = SEVERITY["niveluri"][str(entry["nivel"])]
    return {
        "nivel": entry["nivel"],
        "nume": level["nume"],
        "actiune": level["actiune"],
        "timp_maxim_min": level["timp_maxim_min"],
        "pericole": [SEVERITY["pericole"][h] for h in entry["pericole"]],
        "motiv": entry["motiv"],
    }


def assess_risk(class_key: str, size_cm: float | None, unknown_critical_cm: float) -> Risk:
    if class_key == UNKNOWN_KEY:
        return "critic" if size_cm is not None and size_cm >= unknown_critical_cm else "de verificat"
    entry = SEVERITY["obiecte"].get(class_key)
    return SEVERITY["niveluri"][str(entry["nivel"])]["risc"] if entry else "de verificat"
