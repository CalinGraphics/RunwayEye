from app.vision.types import UNKNOWN_KEY, Risk

CRITICAL_CLASSES = {"surub", "piulita", "cheie", "metal"}
MEDIUM_CLASSES = {"plastic"}

RISK_ORDER: dict[str, int] = {"critic": 0, "mediu": 1, "de verificat": 2}


def assess_risk(class_key: str, size_cm: float | None, unknown_critical_cm: float) -> Risk:
    if class_key in CRITICAL_CLASSES:
        return "critic"
    if class_key in MEDIUM_CLASSES:
        return "mediu"
    if class_key == UNKNOWN_KEY and size_cm is not None and size_cm >= unknown_critical_cm:
        return "critic"
    return "de verificat"
