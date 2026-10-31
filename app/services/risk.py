from app.vision.types import CLASS_LABELS, UNKNOWN_KEY, Risk, class_group

# Soft or light FOD (the "plastic" group: plastic, tape, labels, wood...): tyre/engine damage is
# unlikely, but it still has to be picked up. Every other known class (metal, tools, stones) is
# critical: engine ingestion and tyre cuts.
MEDIUM_GROUPS = {"plastic"}

RISK_ORDER: dict[str, int] = {"critic": 0, "mediu": 1, "de verificat": 2}


def assess_risk(class_key: str, size_cm: float | None, unknown_critical_cm: float) -> Risk:
    if class_key == UNKNOWN_KEY:
        return "critic" if size_cm is not None and size_cm >= unknown_critical_cm else "de verificat"
    group = class_group(class_key)
    if group in MEDIUM_GROUPS:
        return "mediu"
    return "critic" if class_key in CLASS_LABELS else "de verificat"
