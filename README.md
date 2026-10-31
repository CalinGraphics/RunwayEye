# RunwayEye

Supraveghere video a pistei: detectează obiectele străine (șuruburi, unelte, pietre, deșeuri, păsări și alte
animale, persoane, orice obiect nou), spune cât de grave sunt pentru o aeronavă și alertează imediat echipa de
intervenție pe telefon.
Craiova Hackathon V2.0 · track GROUND CONTROL.

**cameră → detecție (2 modele YOLO + detecția schimbărilor) → alertă pe panou și pe telefon → echipa preia și
elimină obiectul → istoric în Supabase**

## Pornire

Cerință: Python 3.11 sau mai nou, de pe python.org (pe Windows bifați „Add to PATH”). Nu e nevoie de placă video,
de Node.js sau de cameră: modelele antrenate și interfața vin gata cu proiectul.

```powershell
git clone <repo> RunwayEye
cd RunwayEye
start_demo.bat          # Windows: dublu clic pe fișier
./start_demo.sh         # macOS / Linux
```

Prima pornire instalează tot (~5 minute, o singură dată). Apoi panoul se deschide singur în browser, la
http://localhost:8000. Un al doilea dublu clic, cu serverul pornit, doar redeschide panoul. Serverul se oprește
închizând fereastra lui.

Implicit rulează **clipul demo** `data/videos/synthetic_demo.mp4`: o pistă pe care o mână lasă pe rând obiecte
reale fotografiate (doză, cutter, baterie, clește, o bucată de plastic), apoi ridică unul. Camera web, o cameră
de supraveghere sau alt clip se aleg din pagina **Sursă**.

## Cum se folosește

| Pagina | La ce folosește |
|---|---|
| **Panou** (`/`) | Video live, harta pistei pe sectoare, alertele active cu severitatea lor. Fiecare alertă: **Confirm** → **Echipă trimisă** → **Obiect eliminat**, sau **Alarmă falsă**; clic pe poză pentru detalii (pericole, acțiune, timp maxim) |
| **Sursă** (`/source`) | Ce filmează sistemul: un clip (din listă sau încărcat de pe calculator), camera web sau o cameră de supraveghere IP (`rtsp://…`), cu test de conexiune. Alegerea rămâne și după repornire |
| **Telefon echipă** (butonul de sus) | Codul QR pentru telefoanele echipei și câte telefoane sunt conectate |
| **Istoric** (`/incidents`) | Toate incidentele, cu filtre, cronologia fiecăruia și export CSV pentru Excel |
| **Calibrare** (`/calibrate`) | Cele 4 colțuri ale pistei: de aici vin sectorul și distanța în metri ale fiecărei alerte |
| **Test** (`/upload`) | Verificare pe o poză sau un clip; cu „Trimite ca incident real” pornește alerta ca o cameră live |

Primii pași, o singură dată:
1. **Calibrare:** clic pe colțurile pistei, în ordinea prag 09 stânga, prag 09 dreapta, prag 27 dreapta,
   prag 27 stânga (stânga/dreapta văzute de la pragul 09). Dimensiunile pistei sunt în `.env` (`RUNWAY_*`).
2. Cu pista goală: **Pista e curată acum** (pe imaginea video). Tot ce apare nou după asta e verificat.
3. **Telefoanele echipei:** vezi mai jos.

## Telefonul echipei

1. Pe panou: **Telefon echipă**, apoi scanați codul QR cu camera telefonului (cheia de acces e inclusă în cod).
2. Pe telefon apăsați o dată **Activează alertele**, ca să sune și să vibreze (vibrația doar pe Android).
3. În fereastra de pe panou apare **✓ telefon conectat**. Probă: **Test** → o poză din `samples\fod-a-test\`
   cu „Trimite ca incident real”: telefonul arată alerta roșie.

**Dacă telefonul nu deschide pagina:**
- Rețelele de facultate sau de firmă blochează des legătura dintre dispozitive, chiar în aceeași rețea Wi-Fi.
  Porniți pe laptop **Hotspot mobil** (Setări → Rețea și internet → Hotspot mobil), conectați telefonul la el și
  alegeți în fereastra **Telefon echipă** adresa „Hotspotul laptopului”. E și varianta cea mai sigură pentru o
  demonstrație: nu depinde de rețeaua sălii.
- Firewall-ul Windows, o singură dată, în PowerShell deschis ca administrator:
  `netsh advfirewall firewall add rule name="RunwayEye 8000" dir=in action=allow protocol=TCP localport=8000 profile=any`
- Telefonul pe date mobile nu vede laptopul: trebuie să fie pe același Wi-Fi sau pe hotspotul laptopului.

Laptopul pe care rulează serverul nu are nevoie de cheie. Alt laptop (panou secundar):
`http://<IP-laptop>:8000/?key=<cheia>`; cheia e în `data/access_key.txt` sau în `ACCESS_KEY` din `.env`.

## Ce detectează

- **Modelul FOD** (`backend/models/best.pt`, YOLO11n antrenat de noi pe cele 31 de clase ale setului FOD-A):
  șuruburi, piulițe, șaibe, chei, clești, ciocane, baterii, cuie, sârmă, pietre, etichete, bucăți de plastic…
- **Modelul general** (`backend/models/general.pt`, YOLOE cu vocabular deschis): **păsări, și moarte**, câini,
  pisici, vulpi, iepuri, bucăți de anvelopă, bagaje, pungi și sticle de plastic, cutii, mănuși. Lista e în
  `backend/app/vision/types.py` (`GENERAL_PROMPTS`, în engleză); după o schimbare:
  `venv\Scripts\python backend\scripts\build_general_model.py`.
- **Persoane** pe pistă (incursiune), tot prin modelul general: alertă critică; o persoană aproape de cameră apare
  ca „Persoană pe pistă” în locul lui „Obiect mare pe pistă”.
- **Detecția schimbărilor** față de pista curată prinde orice alt obiect, ca „obiect necunoscut”, deci nimic nu
  scapă, nici ce nu cunoaște niciun model.

**Severitatea pentru aeronavă** e în `backend/app/services/severity.json`: pentru fiecare tip de obiect, un nivel
de la 1 (neglijabil) la 5 (catastrofal), pericolele (ingestie în motor, tăierea anvelopei, lovirea structurii,
coliziune, atragerea păsărilor…), motivul, acțiunea recomandată și timpul maxim până la eliminare. Din nivel
vine riscul afișat: 4–5 **critic** (metal, unelte, pietre, baterii, animale, persoane, bucăți de anvelopă),
1–3 **mediu** (plastic, hârtie, lemn, bandă); un obiect necunoscut e „de verificat” (critic dacă e mare).
Fișierul se poate modifica fără cod; un test verifică că fiecare clasă are o intrare.

**Ce nu dă alarmă:** mâna care pune obiectul (alerta pornește după ce mâna a ieșit), umbrele și schimbările de
lumină, tremurul camerei, ce intră pe la marginea imaginii, marcajele pistei. Un vehicul sau un obiect mare apare
separat, ca „Obiect mare pe pistă”. **Alarmă falsă** învață zona, inclusiv după repornire.

## Testare

**Imagini pe care modelul nu le-a văzut** (`samples/fod-a-test/`, 93 de imagini, numele fișierului e clasa):

```powershell
venv\Scripts\python backend\scripts\test_samples.py   # tabel + planșă în data\samples_result\
```

Rezultatele modelului din repo (20 de epoci pe un GTX 1650 Ti, 31 octombrie 2026), pe setul de test FOD-A:

| Set de test (2.917 imagini, nevăzute la antrenare) | Valoare |
|---|---|
| mAP50 / mAP50-95 | 0,994 / 0,883 |
| Precizie / recall | 0,991 / 0,994 |
| Cele mai grele clase (mAP50-95) | cui 0,61 · cheie fixă 0,68 · șurub cu piuliță 0,71 |
| `test_samples.py` (93 de imagini) | 93/93 clasă exactă |

Detaliile sunt în `backend/models/metrics.json`, iar graficele pentru slide în `backend/models/report/`.
Imaginile de test vin din aceleași clipuri FOD-A (alte momente, fără cadre vecine), filmate de aproape. Pe alt
asfalt și de la altă distanță modelul greșește mai des: pe pista desenată din clipul demo recunoaște sigur doze,
cutere, baterii, clești, ciocane, șurubelnițe, dar confundă cheile reglabile, unele șuruburi și pietrele. Pentru
o cameră reală, modelul se reantrenează cu câteva sute de cadre filmate de ea (`backend/training/README.md`);
până atunci, detecția schimbărilor prinde oricum obiectul, ca „necunoscut”. Modelul general nu e antrenat de noi;
l-am verificat pe câteva fotografii (pasăre vie și moartă pe asfalt, persoane), nu pe un set de test.

**Clipuri video** (cifrele care contează): filmați cu camera fixă clipuri în care lăsați obiecte pe pistă (primele
2 secunde cu pista goală), puneți-le în `data\eval\`, apoi:

```powershell
venv\Scripts\python backend\scripts\annotate_clip.py data\eval\macheta_01.mp4   # momentul și zona fiecărui obiect
venv\Scripts\python backend\scripts\evaluate.py      # rata de detecție, alarme false pe oră, timp până la alertă
```

Fără clipuri filmate: `backend\scripts\make_eval_set.py` desenează 10 scenarii grele (umbre, lumină, vehicul,
zgomot, mână fără obiect), apoi `evaluate.py --dir data\eval_synthetic`. Sunt bune ca dovadă că sistemul nu dă
alarme la ele, nu pentru cifre.

**Înainte de prezentare:** `venv\Scripts\python backend\scripts\demo_check.py` verifică camera, modelele,
calibrarea, Supabase, serverul și firewall-ul, și spune ce de făcut pentru fiecare problemă.

**Demonstrația pentru juriu (~5 minute):**
1. **Live:** clipul demo (sau macheta cu camera) → la fiecare obiect lăsat, ecran roșu și sunet pe panou, alertă pe
   telefon cu severitatea; pe telefon **Am preluat** → **Obiect eliminat**; incidentul apare în **Istoric** și în Supabase.
2. **Ce nu dă alarmă:** treceți mâna peste pistă fără să lăsați nimic, faceți umbră, aprindeți o lampă.
3. **Modelul:** **Test** → 2–3 poze din `samples\fod-a-test\` și o fotografie cu o pasăre.
4. **Cifrele:** tabelul de mai sus, `data\samples_result\sheet.jpg` și, cu clipuri filmate, `data\eval\summary.md`.

## Supabase: baza de date centrală

Tabelele `cameras`, `incidents`, `incident_events` și bucket-ul `fod-images` (pozele). Fiecare laptop trimite
incidentele lui și primește, la ~10 s, incidentele celorlalte camere, deci **Istoric** arată tot aeroportul.
Alerta nu așteaptă internetul: incidentul se scrie întâi pe laptop, iar în 1–2 secunde ajunge în Supabase; fără
rețea, bara de jos arată „N în coadă” și totul pleacă la revenire.

1. Proiect nou: Supabase → SQL Editor → conținutul lui `backend/supabase/schema.sql` → Run.
   Proiectul existent: fișierele din `backend/supabase/migrations/`.
2. În `.env`: `SUPABASE_URL` și `SUPABASE_KEY` = cheia **secretă** (Project Settings → API Keys → Secret keys,
   `sb_secret_…`). Cheia publică (`sb_publishable_…`) nu poate scrie. Cheia rămâne pe laptop.

Fără URL și cheie, aplicația merge la fel, cu datele doar pe laptop.

## Setări (`.env`)

| Setare | Ce face |
|---|---|
| `VIDEO_SOURCE` | Sursa la prima pornire (apoi pagina **Sursă**): `data/videos/…mp4` clip · `0`, `1`… camera web · `rtsp://…` cameră IP · `http://<ip-telefon>:8080/video` telefon cu „IP Webcam” |
| `FALLBACK_VIDEO` | Clipul folosit când sursa nu răspunde (puneți aici un clip filmat cu macheta) |
| `RUNWAY_LENGTH_M`, `RUNWAY_WIDTH_M`, `MOCKUP_*_CM` | Dimensiunile pistei și ale machetei, pentru metri și mărimea obiectelor |
| `ENABLE_GENERAL_MODEL=false` | Fără modelul general (păsări, animale…): mai rapid pe laptopuri slabe |
| `ENABLE_YOLO=false` | Doar detecția schimbărilor |
| `ENABLE_SUPABASE=false` | Datele doar pe laptop |
| `ENABLE_CLAUDE=true` + `ANTHROPIC_API_KEY` | O propoziție scrisă de Claude la fiecare incident (opțional) |
| `DEMO_MODE=false` | Mod prezentare: fără panoul de reglaj și fără mesaje tehnice |
| `DEMO_FAKE_DETECTIONS=true` | Alerte de test la ~15 s, ca să arătați panoul fără cameră |

## Structura proiectului

```
backend/            serverul (Python, FastAPI)
  app/              vision/ (detecție, calibrare), services/ (incidente, risc + severity.json, Supabase, telefoane), routers/ (pagini, API)
  models/           best.pt (FOD-A), general.pt (păsări, animale…), metrics.json, report/ (grafice)
  scripts/          setup, demo_check, test_samples, evaluate, make_demo_clip…
  tests/            teste automate
  training/         pregătirea setului FOD-A și antrenarea (vezi training/README.md)
  supabase/         schema bazei de date
frontend/           interfața: templates/ (pagini), static/ (CSS, JS, fonturi, sunet)
samples/            imagini de test pentru demonstrație
data/               creat la rulare: incidente, poze, calibrare, clipul demo (nu intră în git)
```

## Dezvoltare

```powershell
venv\Scripts\python -m pytest backend               # testele automate
cd frontend; npm install; npm run css:watch          # doar dacă schimbați paginile (recompilează static\css\app.css)
```

Serverul pornit manual: `set PYTHONPATH=backend` și `venv\Scripts\python -m app` (din folderul proiectului).

## Probleme frecvente

- **Imagine neagră sau puține cadre pe secundă:** camera are nevoie de lumină; verificați `VIDEO_SOURCE`.
- **Alarme la umbre:** lumină constantă, apoi **Pista e curată acum**.
- **Telefonul nu se conectează:** vezi „Telefonul echipei” mai sus (hotspotul laptopului rezolvă aproape mereu).
- **Fără sunet pe panou:** clic oriunde în pagină sau pe **Activează sunetul**.
- **Un obiect ridicat lasă o alertă „necunoscut”:** obiectul era pe pistă când s-a apăsat **Pista e curată acum**:
  **Alarmă falsă** sau, cu pista goală, din nou **Pista e curată acum**.
