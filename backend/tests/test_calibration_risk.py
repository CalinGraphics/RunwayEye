import pytest

from app.services.risk import assess_risk
from app.vision.calibration import Calibration, RunwayGeometry
from app.vision.types import CLASS_LABELS, FOD_CLASSES, class_group, class_key

GEOM = RunwayGeometry(length_m=2500, width_m=45, sectors_along=6, sectors_across=3)
# Oblique view: far end (threshold 27) narrower, like a camera on a pole.
CORNERS = [(100, 500), (860, 500), (620, 120), (340, 120)]


@pytest.fixture
def cal():
    return Calibration(CORNERS, (960, 540), GEOM)


def test_corners_map_to_runway_corners(cal):
    expected = [(0, 0), (0, 45), (2500, 45), (2500, 0)]
    for (x, y), (u, v) in zip(CORNERS, expected):
        ru, rv = cal.to_runway(x, y)
        assert ru == pytest.approx(u, abs=0.5)
        assert rv == pytest.approx(v, abs=0.05)


def test_runway_center_maps_to_centerline_middle(cal):
    x, y = cal.to_pixel(1250, 22.5)
    pos = cal.pixel_to_runway(x, y)
    assert pos.distance_m == pytest.approx(1250, abs=0.5)
    assert pos.offset_m == pytest.approx(0, abs=0.05)
    assert pos.inside


def test_rectangle_center_is_runway_center():
    rect = Calibration([(0, 0), (0, 100), (600, 100), (600, 0)], (600, 100), GEOM)
    pos = rect.pixel_to_runway(300, 50)
    assert pos.distance_m == pytest.approx(1250)
    assert pos.offset_m == pytest.approx(0)
    assert pos.sector == "C2" or pos.sector == "D2"


@pytest.mark.parametrize(
    "u,v,sector",
    [
        (0, 0, "A1"),
        (416, 14.9, "A1"),
        (2500 / 6, 15, "B2"),
        (1250, 22.5, "D2"),
        (2499.9, 44.9, "F3"),
        (2500, 45, "F3"),
        (1000, 30, "C3"),
    ],
)
def test_sectors_at_edges(u, v, sector):
    assert GEOM.sector(u, v) == sector


def test_left_of_centerline_is_negative(cal):
    x, y = cal.to_pixel(800, 5)
    assert cal.pixel_to_runway(x, y).offset_m == pytest.approx(-17.5, abs=0.1)


def test_mask_contains_inside_points(cal):
    m = cal.mask(960, 540)
    assert m[400, 480] == 255
    assert m[20, 20] == 0
    assert cal.contains(480, 400)
    assert not cal.contains(20, 20)


def test_calibration_save_load_roundtrip(tmp_path, cal):
    path = tmp_path / "calibration.json"
    cal.save(path)
    loaded = Calibration.load(path, GEOM, (480, 270))
    assert loaded.corners[0] == pytest.approx((50, 250))


def test_mockup_size_estimate():
    geom = RunwayGeometry(2500, 45, mockup_length_cm=100, mockup_width_cm=60)
    rect = Calibration([(0, 0), (0, 600), (1000, 600), (1000, 0)], (1000, 600), geom)
    from app.vision.types import BBox

    # 50 px along = 5 cm on a 100 cm mock-up; 30 px across = 3 cm.
    assert rect.size_cm(BBox(x=100, y=100, w=50, h=30)) == pytest.approx(5.0, abs=0.01)


def test_upright_object_is_measured_by_its_height():
    # Real runway seen at an angle: the far edge (y=0) is squeezed into fewer pixels than the near one.
    geom = RunwayGeometry(100, 40)
    cal = Calibration([(300, 100), (0, 500), (1000, 500), (700, 100)], (1000, 600), geom)
    from app.vision.types import BBox

    # Projected flat, the box's top lands somewhere on the asphalt and the size depends on the
    # camera angle; measured upright, height = box height at the scale of the feet.
    box = BBox(x=490, y=380, w=20, h=100)
    upright = cal.size_cm(box, upright=True)
    (u1, v1), (u2, v2) = cal.to_runway(490, 480), cal.to_runway(510, 480)
    base_cm = ((u2 - u1) ** 2 + (v2 - v1) ** 2) ** 0.5 * 100
    assert upright == pytest.approx(base_cm * 5, rel=0.01)
    assert upright != pytest.approx(cal.size_cm(box), rel=0.05)


@pytest.mark.parametrize(
    "key,size,risk",
    [
        ("cheie", None, "critic"),
        ("metal", None, "critic"),
        ("surub", 2, "critic"),
        ("piulita", None, "critic"),
        ("plastic", 30, "mediu"),
        # FOD-A classes
        ("bolt", None, "critic"),
        ("wrench", None, "critic"),
        ("rock", 1, "critic"),
        ("plastic_part", None, "mediu"),
        ("luggage_tag", 20, "mediu"),
        ("unknown", None, "de verificat"),
        ("unknown", 5, "de verificat"),
        ("unknown", 20, "critic"),
    ],
)
def test_risk(key, size, risk):
    assert assess_risk(key, size, unknown_critical_cm=15) == risk


def test_every_fod_class_has_a_label_and_a_risk():
    for key in FOD_CLASSES:
        assert CLASS_LABELS[key]
        assert assess_risk(key, None, unknown_critical_cm=15) in ("critic", "mediu")
    assert len(FOD_CLASSES) == 31


def test_class_group():
    assert class_group("bolt") == "surub"
    assert class_group("nut") == "piulita"
    assert class_group("surub") == "surub"
    assert class_group("unknown") == "unknown"


def test_class_key_normalizes_model_names():
    assert class_key("BoltNutSet") == "bolt_nut_set"
    assert class_key("metal part") == "metal_part"
    assert class_key("Piuliță ") == "piulita"
    assert class_key("bolt_nut_set") == "bolt_nut_set"


def test_rock_is_its_own_group_and_critical():
    assert class_group("rock") == "piatra"
    assert assess_risk("rock", None, unknown_critical_cm=15) == "critic"
    assert assess_risk("something_else", None, unknown_critical_cm=15) == "de verificat"


def test_general_classes_risk():
    assert assess_risk("pasare", None, unknown_critical_cm=15) == "critic"
    assert assess_risk("anvelopa", None, unknown_critical_cm=15) == "critic"
    assert assess_risk("punga", None, unknown_critical_cm=15) == "mediu"
    assert CLASS_LABELS["pasare"] == "pasăre"


def test_every_class_has_a_severity_entry():
    # severity.json is the single source of the risk: a class missing there would be "de verificat".
    from app.services.risk import SEVERITY, severity

    for key in CLASS_LABELS:
        if key == "unknown":
            continue
        assert key in SEVERITY["obiecte"], key
        s = severity(key)
        assert 1 <= s["nivel"] <= 5 and s["pericole"] and s["motiv"] and s["actiune"]


def test_severity_examples():
    from app.services.risk import severity

    assert severity("bolt")["nivel"] == 5 and severity("bolt")["timp_maxim_min"] == 5
    assert severity("paint_chip")["nume"] == "Neglijabil"
    assert severity("persoana")["nivel"] == 5
    assert severity("unknown", 30, 15)["nivel"] == 4 and severity("unknown", 3, 15)["nivel"] == 2
    assert severity("no_such_class") is None


def test_clip_scene_brings_its_own_runway(tmp_path):
    import json

    from app.vision.scene import Scene

    clip = tmp_path / "real.mp4"
    clip.write_bytes(b"")
    (tmp_path / "real.scene.json").write_text(json.dumps({
        "runway": {"length_m": 400, "width_m": 45, "sectors_along": 4, "sectors_across": 3},
        "corners": [[100, 50], [0, 200], [400, 200], [300, 50]],
        "frame_size": [400, 200],
        "tuning": {"min_area_px": 10},
    }))
    scene = Scene.for_source("file", str(clip))
    assert scene.tuning == {"min_area_px": 10}
    assert scene.geometry.length_m == 400 and scene.geometry.mockup_length_cm == 0
    cal = scene.calibration((800, 400))  # frame twice as large: corners rescale
    assert cal.corners[2] == (800.0, 400.0)
    assert Scene.for_source("camera", 0) is None
    assert Scene.for_source("file", str(tmp_path / "other.mp4")) is None
