from app.vision.calibration import Calibration, RunwayGeometry
from app.vision.engine import LargeObject, ObjectGone, ObjectReturned
from app.vision.types import BBox
from tests.synthetic import FakeYolo, Scene, incidents, learn, make_engine, run

BOLT = (150, 110, 14, 10, 220)


def test_static_object_detected_after_persist_frames():
    scene, engine = Scene(), make_engine()
    learn(engine, scene)
    n = engine.params.persist_frames

    early = run(engine, [scene.frame([BOLT]) for _ in range(n - 1)])
    assert incidents(early) == []

    late = run(engine, [scene.frame([BOLT])])
    found = incidents(late)
    assert len(found) == 1
    det = found[0].detection
    assert det.source == "change"
    assert det.object_class == "obiect necunoscut"
    assert det.risk == "de verificat"
    assert det.bbox.contains(*det.center_px)
    assert det.crop_jpeg_b64


def test_thin_colored_object_is_detected():
    # A paper clip: 2 px wide, coloured, about the same grey level as the asphalt.
    scene, engine = Scene(), make_engine()
    learn(engine, scene)
    events = []
    for _ in range(12):
        f = scene.frame()
        f[100:102, 120:160] = (60, 60, 190)
        f[100:112, 158:160] = (60, 60, 190)
        f[110:112, 128:160] = (60, 60, 190)
        events += engine.process(f).events
    assert len(incidents(events)) == 1


def test_moving_object_is_ignored():
    scene, engine = Scene(), make_engine()
    learn(engine, scene)
    frames = [scene.frame([(10 + 16 * i, 100, 14, 10, 220)]) for i in range(18)]
    assert incidents(run(engine, frames)) == []


def test_hand_placing_object_gives_one_alert_and_no_large_alarm():
    scene, engine = Scene(), make_engine()
    learn(engine, scene)
    # A big "hand" sweeps in, drops the bolt and leaves.
    frames = [scene.frame([(20 + 12 * i, 60, 70, 60, 60)]) for i in range(10)]
    frames += [scene.frame([BOLT]) for _ in range(20)]
    events = run(engine, frames)
    assert len(incidents(events)) == 1
    assert not any(isinstance(e, LargeObject) for e in events)


def test_slow_light_change_raises_no_alerts():
    scene, engine = Scene(), make_engine()
    learn(engine, scene)
    events = []
    for _ in range(300):
        scene.brightness += 0.25
        events.extend(engine.process(scene.frame()).events)
    assert incidents(events) == []
    assert not any(isinstance(e, LargeObject) for e in events)


def test_object_outside_runway_mask_is_ignored():
    geometry = RunwayGeometry(length_m=2500, width_m=45)
    cal = Calibration([(100, 60), (100, 180), (300, 180), (300, 60)], (320, 240), geometry)
    scene, engine = Scene(), make_engine(calibration=cal)
    learn(engine, scene)

    outside = run(engine, [scene.frame([(20, 20, 14, 10, 220)]) for _ in range(15)])
    assert incidents(outside) == []

    inside = run(engine, [scene.frame([(20, 20, 14, 10, 220), BOLT]) for _ in range(15)])
    found = incidents(inside)
    assert len(found) == 1
    assert found[0].detection.sector is not None
    assert found[0].detection.runway_distance_m is not None


def test_no_duplicate_incidents():
    scene, engine = Scene(), make_engine()
    learn(engine, scene)

    events = run(engine, [scene.frame([BOLT]) for _ in range(100)])
    # Short flicker (object hidden for a few frames) keeps the same incident.
    events += run(engine, [scene.frame() for _ in range(3)])
    events += run(engine, [scene.frame([BOLT]) for _ in range(30)])
    assert len(incidents(events)) == 1
    incident_id = incidents(events)[0].detection.id

    # Picked up: incident marked "obiect dispărut"; put back in the same spot: same incident.
    gone = run(engine, [scene.frame() for _ in range(engine.params.gone_frames + 2)])
    assert [e.incident_id for e in gone if isinstance(e, ObjectGone)] == [incident_id]
    back = run(engine, [scene.frame([BOLT]) for _ in range(20)])
    assert incidents(back) == []
    assert [e.incident_id for e in back if isinstance(e, ObjectReturned)] == [incident_id]


def test_closed_false_alarm_does_not_retrigger():
    scene, engine = Scene(), make_engine()
    learn(engine, scene)
    events = run(engine, [scene.frame([BOLT]) for _ in range(15)])
    (inc,) = incidents(events)
    engine.close_incident(inc.detection.id, absorb=True)
    assert incidents(run(engine, [scene.frame([BOLT]) for _ in range(40)])) == []


def test_object_removed_after_learning_is_one_false_alarm():
    # Known limitation (documented): a bolt present while the reference is learned and then
    # picked up gives one "obiect necunoscut"; after "Alarmă falsă" it stays quiet.
    scene, engine = Scene(), make_engine()
    for _ in range(engine.params.reference_frames):
        engine.process(scene.frame([BOLT]))
    found = incidents(run(engine, [scene.frame() for _ in range(15)]))
    assert len(found) == 1
    engine.close_incident(found[0].detection.id, absorb=True)
    assert incidents(run(engine, [scene.frame() for _ in range(30)])) == []


def test_false_alarm_object_removed_later_leaves_no_ghost():
    scene, engine = Scene(), make_engine()
    learn(engine, scene)
    (inc,) = incidents(run(engine, [scene.frame([BOLT]) for _ in range(15)]))
    engine.close_incident(inc.detection.id, absorb=True)
    run(engine, [scene.frame([BOLT]) for _ in range(5)])
    assert incidents(run(engine, [scene.frame() for _ in range(30)])) == []


def test_uniform_object_over_markings_is_not_a_ghost():
    # A plain piece of plastic lying on the threshold stripes must still be reported.
    scene, engine = Scene(), make_engine()
    for x in range(100, 220, 12):
        scene.base[60:180, x : x + 6] = 230
    learn(engine, scene)
    events = run(engine, [scene.frame([(130, 100, 30, 24, 105)]) for _ in range(15)])
    assert len(incidents(events)) == 1


def test_very_large_object_is_a_vehicle_event():
    scene, engine = Scene(), make_engine()
    learn(engine, scene)
    events = run(engine, [scene.frame([(60, 60, 120, 80, 40)]) for _ in range(20)])
    assert incidents(events) == []
    large = [e for e in events if isinstance(e, LargeObject)]
    assert len(large) == 1 and large[0].active


def test_yolo_plus_change_gives_source_both():
    yolo = FakeYolo([(BBox(x=146, y=106, w=22, h=18), "surub", "șurub", 0.91)])
    scene, engine = Scene(), make_engine(params={"yolo_every_n": 1})
    learn(engine, scene)
    engine.set_yolo(yolo)
    found = incidents(run(engine, [scene.frame([BOLT]) for _ in range(10)]))
    assert len(found) == 1
    det = found[0].detection
    assert det.source == "both"
    assert det.class_key == "surub"
    assert det.object_class == "șurub"
    assert det.risk == "critic"


def test_yolo_without_change_keeps_source_yolo():
    # YOLO sees an object that is part of the reference (no change region).
    yolo = FakeYolo([(BBox(x=40, y=40, w=20, h=20), "cheie", "cheie", 0.8)])
    scene, engine = Scene(), make_engine(yolo=yolo, params={"yolo_every_n": 1})
    found = incidents(run(engine, [scene.frame() for _ in range(25)]))
    assert len(found) == 1
    assert found[0].detection.source == "yolo"
