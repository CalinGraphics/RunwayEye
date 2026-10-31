from app.vision.fusion import Candidate
from app.vision.types import BBox
from app.vision.yolo_detector import CombinedDetector


class FakeDetector:
    def __init__(self, cands, name):
        self.cands, self.model_name, self.conf = cands, name, 0.35
        self.loaded, self.provisional, self.error = True, False, None

    def detect(self, frame):
        return list(self.cands)


def _cand(x, key, conf=0.8):
    b = BBox(x=x, y=10, w=40, h=40)
    return Candidate(b, b.center, b.area, "yolo", key, key, conf)


def test_general_model_adds_only_new_objects():
    fod = FakeDetector([_cand(10, "bolt")], "best.pt")
    general = FakeDetector([_cand(12, "carton"), _cand(300, "pasare")], "general.pt")
    det = CombinedDetector(fod, general)
    assert [c.class_key for c in det.detect(None)] == ["bolt", "pasare"]
    assert det.model_name == "best.pt + general.pt"


def test_tuning_threshold_reaches_the_fod_model_only():
    fod, general = FakeDetector([], "best.pt"), FakeDetector([], "general.pt")
    det = CombinedDetector(fod, general)
    det.conf = 0.5
    assert fod.conf == 0.5 and general.conf == 0.35


def test_more_confident_bird_beats_the_fod_models_guess():
    # FOD-A has no birds: the FOD model can only guess a hardware class for a bird.
    fod = FakeDetector([_cand(10, "clamp_part", 0.5), _cand(200, "bolt")], "best.pt")
    general = FakeDetector([_cand(12, "pasare", 0.7)], "general.pt")
    assert sorted(c.class_key for c in CombinedDetector(fod, general).detect(None)) == ["bolt", "pasare"]


def test_less_confident_bird_stays_in_the_label():
    fod = FakeDetector([_cand(10, "clamp_part", 0.75)], "best.pt")
    [c] = CombinedDetector(fod, FakeDetector([_cand(12, "pasare", 0.54)], "general.pt")).detect(None)
    assert c.class_key == "clamp_part" and c.label == "clamp_part / posibil pasare"
    fod = FakeDetector([_cand(10, "screw", 0.97)], "best.pt")
    [c] = CombinedDetector(fod, FakeDetector([_cand(12, "pasare", 0.35)], "general.pt")).detect(None)
    assert c.label == "screw"


def test_general_model_runs_every_nth_call_and_its_boxes_are_reused():
    class Counting(FakeDetector):
        calls = 0

        def detect(self, frame):
            Counting.calls += 1
            return super().detect(frame)

    det = CombinedDetector(FakeDetector([], "best.pt"), Counting([_cand(300, "pasare")], "general.pt"), general_every=2)
    results = [det.detect_stream(None, "live") for _ in range(4)]
    assert Counting.calls == 2
    assert all(r and r[0].class_key == "pasare" for r in results)
    # A photo on its own and another stream never get the live stream's boxes.
    det.detect(None)
    det.detect_stream(None, "clip")
    assert Counting.calls == 4
