# RunwayEye

Supraveghere video a pistei care detectează obiectele străine (FOD) și alertează echipa de intervenție.
Craiova Hackathon V2.0 · track GROUND CONTROL.

**video → YOLO + detecția schimbărilor → persistență → alertă (ecran roșu, sunet) → sector și metri pe pistă →
telefonul echipei (`/team`) → „Am preluat” / „Obiect eliminat” → istoric**

Totul rulează local, fără internet: detecția, panoul, alertele și pagina echipei. Supabase și Claude sunt opționale.

## Pornire rapidă (Windows)

Cerințe: Python 3.14 (testat; 3.12+ merge), Node.js (doar pentru compilarea CSS), o cameră web sau un clip video.

```powershell
cd "D:\Runaway Eye"
py -3.14 -m venv venv
venv\Scripts\python -m pip install -r requirements.txt   # ~5 min (torch, ultralytics)
npm install                                              # Tailwind + DaisyUI + HTMX + Alpine, compilate local
copy .env.example .env                                   # apoi verificați VIDEO_SOURCE
start_demo.bat
```

`start_demo.bat` activează venv-ul, pornește serverul pe `0.0.0.0:8000`, afișează adresa pentru telefon
și deschide panoul în browser. Manual: `venv\Scripts\activate` apoi `python -m app`.

Înainte de fiecare prezentare: `python scripts\demo_check.py`. Pentru fiecare problemă spune ce setare din `.env` o ocolește.

## Paginile

| Adresă | Ce e |
|---|---|
| `/` | Command Center: video live, harta pistei cu sectoare, alertele, bara de stare |
| `/team` | Telefonul echipei de intervenție (alertă roșie, sunet, vibrație, „Am preluat”, „Obiect eliminat”) |
| `/incidents` | Istoricul, cu filtre și cronologia fiecărui incident |
| `/calibrate` | Cele 4 colțuri ale pistei: masca pistei, sectoarele și metrii |
| `/upload` | Test pe o poză sau un clip, fără să creeze incidente |
| `/health`, `/api/status` | Verificări rapide |

## Sursa video (`VIDEO_SOURCE` în `.env`)

| Valoare | Sursă |
|---|---|
| `0`, `1`… | Camera web cu acel index |
| `data/videos/macheta.mp4` | Fișier, redat în buclă la viteza lui |
| `rtsp://…` | Cameră IP |
| `http://<ip-telefon>:8080/video` | Telefon cu o aplicație de tip „IP Webcam” |

Codul e același pentru toate. Dacă sursa nu răspunde, aplicația trece automat pe `FALLBACK_VIDEO`
(filmați un clip cu macheta!), iar bara de stare arată **REZERVĂ**.

Fără cameră și fără clip real: `VIDEO_SOURCE=data/videos/synthetic_demo.mp4`, o machetă desenată în care
o „mână” lasă un șurub și o agrafă. E doar pentru test, nu pentru cifre.

## Telefonul echipei

1. Laptopul și telefonul în aceeași rețea (cel mai sigur: hotspot de pe un telefon).
2. Portul permis în firewall (o singură dată, PowerShell ca administrator):
   `netsh advfirewall firewall add rule name="RunwayEye 8000" dir=in action=allow protocol=TCP localport=8000`
3. Pe panou: **Telefon echipă** → cod QR cu `http://<IP-laptop>:8000/team`.
4. Pe telefon: apăsați o dată **Activează alertele** (browserele blochează sunetul și vibrația până la prima atingere).
   Vibrația merge pe Android (Chrome); iPhone nu permite vibrația din pagini web.

## Calibrarea

`/calibrate` → clic pe colțuri în ordinea: prag 09 stânga, prag 09 dreapta, prag 27 dreapta, prag 27 stânga
(stânga/dreapta privit de la pragul 09). Se salvează în `data/calibration.json`. Dimensiunile pistei
(`RUNWAY_LENGTH_M`, `RUNWAY_WIDTH_M`) și ale machetei (`MOCKUP_*_CM`) sunt în `.env` și **trebuie verificate**.
Fără calibrare, toată imaginea e considerată pistă și alertele nu au sector.

## Detecția pe scurt

- **YOLO** (`models/best.pt`, antrenat de noi cu `training/train_colab.ipynb` în Google Colab). Fără `best.pt`
  se folosește `yolo11n.pt` pre-antrenat, marcat **provizoriu** în bara de stare.
- **Detecția schimbărilor** față de pista curată (butonul **Pista e curată acum**) prinde orice alt obiect, ca „obiect necunoscut”.
- **Persistență:** alerta pornește doar după `PERSIST_FRAMES` cadre aproape în același loc: mâna care pune obiectul nu dă alarmă.
- **Obiect mare** (peste `MAX_AREA_PX`): eveniment separat „Vehicul / obiect mare pe pistă”, fără ecran roșu.
- **Alarmă falsă:** zona e învățată ca fundal și nu mai declanșează alerta.
- Reglaj live (doar cu `DEMO_MODE=true`): **Mai mult → Reglaj detecție**.

## Opriri din `.env`

| Setare | Efect |
|---|---|
| `ENABLE_YOLO=false` | Doar detecția schimbărilor (nu are nevoie de model) |
| `ENABLE_SUPABASE=false` sau `USE_LOCAL_DATA=true` | Doar copia locală (`data/runwayeye.db` + poze în `data/images/`) |
| `ENABLE_CLAUDE=false` | Fără descrierea AI (bonus) |
| `DEMO_MODE=false` | Mod prezentare: fără panou de reglaj și fără mesaje tehnice |
| `DEMO_FAKE_DETECTIONS=true` | Detecții false la ~15 s, ca să arătați panoul fără cameră |

## Supabase (opțional)

1. Proiectul Supabase are deja tabelele `cameras`, `incidents`, `incident_events` și bucket-ul `fod-images`.
2. În `.env`: `ENABLE_SUPABASE=true`, `USE_LOCAL_DATA=false`, `SUPABASE_URL`, `SUPABASE_KEY` (cheia secretă / service role, rămâne pe laptop).
3. Aplicația scrie întâi local, apoi sincronizează în fundal. Dacă pică internetul, datele rămân în coadă
   (bara de stare arată „N în coadă”) și se trimit când revine conexiunea.

## Evaluarea (cifrele pentru juriu)

```powershell
python scripts\annotate_clip.py data\eval\macheta_01.mp4   # marcați momentul și zona fiecărui obiect
python scripts\evaluate.py                                 # rata de detecție, alarme false/oră, timp până la alertă
```

Rezultatul e în `data/eval/summary.md`. mAP-ul modelului vine din notebook-ul de antrenare (`metrics.json`).

## Dezvoltare

```powershell
venv\Scripts\python -m pytest        # teste pe cadre sintetice + API
npm run css:watch                    # recompilează CSS-ul la modificarea șabloanelor
python scripts\browser_check.py http://127.0.0.1:8000/ panou.png --wait 4   # captură headless
```

Structura: `app/vision/` (captură, detecție, fuziune, calibrare, desenare), `app/services/` (incidente, risc,
date, SSE, telefonul echipei), `app/routers/`, `app/templates/`, `scripts/`, `tests/`, `training/` (notebook-ul Colab).

## Probleme frecvente

- **Imagine neagră / puține cadre pe secundă:** camera are nevoie de lumină (lampă pe machetă); verificați indexul camerei.
- **Alarme la umbre:** lumină constantă, apoi **Pista e curată acum**; măriți pragul din **Reglaj**.
- **Telefonul nu se conectează:** aceeași rețea, regula de firewall, `HOST=0.0.0.0`; `demo_check.py` verifică toate trei.
- **Fără sunet pe panou:** clic oriunde în pagină sau pe **Activează sunetul**.
- **Un obiect ridicat lasă o alertă „necunoscut”:** se întâmplă dacă obiectul era pe pistă când s-a învățat referința.
  Apăsați **Alarmă falsă** (zona se învață) sau **Pista e curată acum** cu pista goală.
