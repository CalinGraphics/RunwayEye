import random

from training.prepare_fod_a import GAP, split_video, videos


def test_videos_follow_shared_classes():
    # A nut enters a bolt video (same video), then a new video with only pliers starts.
    frames = [("a", [(3, 0, 0, 0, 0)])] * 5 + [("b", [(3, 0, 0, 0, 0), (17, 0, 0, 0, 0)])] * 5
    frames += [("c", [(21, 0, 0, 0, 0)])] * 4
    assert [len(v) for v in videos(frames)] == [10, 4]


def test_split_video_has_all_splits_and_a_gap():
    idx = list(range(1000))
    split = split_video(idx, val=0.1, test=0.15, rng=random.Random(0))
    assert {"train", "val", "test"} <= set(split.values())
    train = sorted(i for i, p in split.items() if p == "train")
    for i, p in split.items():
        if p != "train":
            # No held-out frame closer than GAP frames to a training frame.
            assert min(abs(i - t) for t in train) > GAP


def test_short_video_still_reaches_test():
    split = split_video(list(range(127)), val=0.1, test=0.15, rng=random.Random(0))
    assert "test" in split.values() and "val" in split.values()
