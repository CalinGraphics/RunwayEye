"""Train the FOD detector on FOD-A on the local GPU, then score it on the held-out test split.

    venv-train\\Scripts\\python training/train_local.py                      # defaults below
    venv-train\\Scripts\\python training/train_local.py --epochs 60 --model yolo11s.pt
    venv-train\\Scripts\\python training/train_local.py --resume              # continue an interrupted run

Needs a CUDA build of torch (see training/README.md). Writes runs/fod-a/<name>/, copies the
best weights to models/best.pt and the test metrics to models/metrics.json.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets/fod-a/yolo/data.yaml"


def test_metrics(weights: Path, data: Path, imgsz: int, batch: int) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    m = model.val(data=str(data), split="test", imgsz=imgsz, batch=batch, plots=True,
                  project=str(weights.parent.parent), name="test", exist_ok=True)
    per_class = {model.names[int(c)]: {"mAP50": round(float(m.box.ap50[i]), 4), "mAP50_95": round(float(m.box.ap[i]), 4)}
                 for i, c in enumerate(m.box.ap_class_index)}
    return {
        "split": "test",
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "inference_ms_per_image": round(float(m.speed["inference"]), 2),
        "per_class": per_class,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--model", default="yolo11n.pt", help="punctul de plecare, pre-antrenat pe COCO")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=320, help="FOD-A are imagini de 300x300")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2, help="mai mult = mai multă memorie RAM")
    ap.add_argument("--name", default="yolo11n")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        sys.exit("torch nu vede placa video. Instalați varianta CUDA (training/README.md) sau folosiți Colab.")
    if not args.data.exists():
        sys.exit(f"Lipsește {args.data}. Rulați întâi training/prepare_fod_a.py.")

    run_dir = ROOT / "runs/fod-a" / args.name
    if args.resume:
        YOLO(str(run_dir / "weights/last.pt")).train(resume=True, workers=args.workers, batch=args.batch)
    else:
        local = ROOT / "models" / args.model  # reuse the copy the app already downloaded
        YOLO(str(local) if local.exists() else args.model).train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            patience=10,
            cos_lr=True,
            close_mosaic=10,
            # The camera sees the runway from above at any angle: flips are safe for FOD.
            flipud=0.5,
            fliplr=0.5,
            # Smaller objects than in FOD-A's close-ups, as seen from a fixed runway camera.
            scale=0.6,
            project=str(run_dir.parent),
            name=args.name,
            exist_ok=True,
            seed=0,
            deterministic=False,
            plots=True,
        )

    best = run_dir / "weights/best.pt"
    metrics = test_metrics(best, args.data, args.imgsz, args.batch)
    split = json.loads((args.data.parent / "split.json").read_text(encoding="utf-8"))
    metrics.update({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "FOD-A v2.1 (Munyer et al., ICMLA 2021), 31 clase",
        "dataset_split": {p: split[p]["images"] for p in ("train", "val", "test")},
        "split_method": split["split"],
        "base_model": args.model,
        "epochs_run": _epochs_run(run_dir),
        "imgsz": args.imgsz,
        "gpu": torch.cuda.get_device_name(0),
    })
    models = ROOT / "models"
    shutil.copy2(best, models / "best.pt")
    (models / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    # The plots worth a slide, committed next to the model (runs/ is not).
    report = models / "report"
    report.mkdir(exist_ok=True)
    for src, name in ((run_dir / "results.png", "antrenare_curbe.png"),
                      (run_dir / "results.csv", "antrenare_epoci.csv"),
                      (run_dir / "test/confusion_matrix_normalized.png", "test_matrice_confuzie.png"),
                      (run_dir / "test/BoxPR_curve.png", "test_curba_PR.png"),
                      (run_dir / "test/val_batch0_pred.jpg", "test_exemple_predictii.jpg")):
        if src.exists():
            shutil.copy2(src, report / name)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=2, ensure_ascii=False))
    print(f"Copiat: {models / 'best.pt'}, metrics.json și graficele în {report}")


def _epochs_run(run_dir: Path) -> int | None:
    results = run_dir / "results.csv"
    return max(0, len(results.read_text().splitlines()) - 1) if results.exists() else None


if __name__ == "__main__":
    main()
