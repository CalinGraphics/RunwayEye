# Antrenarea modelului YOLO

`models/best.pt` vine deja cu repo-ul. Pașii de mai jos sunt doar pentru a-l reface (alt număr de
epoci, alt model de bază, poze proprii cu macheta).

## Setul de date: FOD-A

Munyer et al., *FOD-A: A Dataset for Foreign Object Debris in Airports*, ICMLA 2021 ·
https://github.com/FOD-UNOmaha/FOD-data (licență MIT). 33.793 de imagini de 300×300, 31 de clase
(șuruburi, piulițe, chei, clești, baterii, pietre, etichete…), cadre din ~55 de clipuri filmate pe asfalt.

1. Descărcați varianta **Pascal VOC** (~412 MB) și dezarhivați-o în `datasets/fod-a/raw/`
   (`datasets/` nu intră în git).
2. Conversia în format YOLO:

   ```powershell
   venv\Scripts\python training\prepare_fod_a.py datasets\fod-a\raw\FODPascalVOCFormat-V.2.1\VOC2007
   ```

   Rezultatul e în `datasets/fod-a/yolo/` (`data.yaml`, `split.json`). Durează ~1 minut.
   Copiază și 3 imagini de test pe clasă în `samples/fod-a-test/` (pentru `scripts/test_samples.py`).

**De ce nu împărțirea oficială FOD-A:** listele `trainval.txt` / `test.txt` amestecă cadre la întâmplare,
deci aproape fiecare imagine de test are un cadru vecin aproape identic în train. mAP-ul iese umflat.
Aici fiecare clip e tăiat în blocuri de cadre consecutive, blocurile merg în train / val / test, iar
15 cadre (~0,5 s) de la fiecare graniță dintre seturi sunt aruncate. Toate cele 31 de clase ajung în test.

## Antrenarea pe placa video (Windows, NVIDIA)

Un mediu separat, cu torch pentru CUDA (aplicația folosește `venv/` cu torch pe CPU):

```powershell
py -3 -m venv venv-train
venv-train\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
venv-train\Scripts\python -m pip install -r requirements.txt
venv-train\Scripts\python training\train_local.py
```

Valorile implicite (`yolo11n.pt`, 20 de epoci, `imgsz=320`, batch 32, 2 procese de încărcare) merg
pe o placă de 4 GB și 8 GB RAM, în ~2 ore pe un GTX 1650 Ti. La sfârșit scriptul evaluează pe setul de **test**
și copiază `best.pt` și `metrics.json` în `models/`. Graficele (matricea de confuzie, curbele) sunt în
`runs/fod-a/yolo11n/`. O antrenare întreruptă se continuă cu `--resume`.

Dacă rămâne fără memorie: `--batch 16` (placa video) sau `--workers 1` (memoria RAM).

## Fără placă video NVIDIA: Google Colab

`training/train_colab.ipynb`: arhivați conținutul lui `datasets/fod-a/yolo/` ca `.zip`, încărcați-l în
notebook (GPU T4 gratuit), apoi copiați `best.pt` și `metrics.json` descărcate în `models/`.

## Verificarea modelului

```powershell
venv\Scripts\python scripts\test_samples.py      # imaginile din samples/fod-a-test, nevăzute la antrenare
```
