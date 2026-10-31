# RunwayEye

Supraveghere video a pistei care detectează obiectele străine (FOD) și alertează echipa de intervenție.
Craiova Hackathon V2.0 · track GROUND CONTROL.

**video → YOLO + detecția schimbărilor → persistență → alertă (ecran roșu, sunet) → sector și metri pe pistă →
telefonul echipei (`/team`) → „Am preluat” / „Obiect eliminat” → istoric**

Datele merg în **Supabase** (baza centrală, toate camerele). Detecția, panoul, alertele și pagina echipei rulează
pe laptop și merg și fără internet: incidentele intră într-o coadă și ajung în Supabase când revine conexiunea.

## Pornire rapidă (Windows, macOS, Linux)

Cerință: Python 3.11 sau mai nou (testat pe 3.14), de pe python.org (pe Windows bifați „Add to PATH”).
Nu e nevoie de Node.js, de placă video sau de cameră: CSS-ul, fonturile, HTMX/Alpine și modelul antrenat
(`models/best.pt`) vin cu repo-ul.

```powershell
git clone <repo> RunwayEye
cd RunwayEye
start_demo.bat          # Windows (merge și dublu clic)
./start_demo.sh         # macOS / Linux
```

La prima pornire scriptul rulează `scripts/setup.py` (~5 minute, o singură dată): creează `venv/`, instalează
dependențele (pe Linux varianta CPU a torch), copiază `.env.example` în `.env` și desenează clipul demo
`data/videos/synthetic_demo.mp4`. Apoi pornește serverul pe `0.0.0.0:8000`, afișează adresa pentru telefon
și deschide panoul în browser. Manual: `python scripts/setup.py`, apoi `venv\Scripts\python -m app`
(Windows) sau `venv/bin/python -m app`.

Fără cameră web, aplicația trece singură pe clipul demo (bara de stare arată **REZERVĂ**).
`venv/` nu se copiază de pe alt calculator; dacă a fost copiat, `setup.py` îl recunoaște și îl reface.

Înainte de fiecare prezentare: `python scripts\demo_check.py`. Pentru fiecare problemă spune ce setare din `.env` o ocolește.

## Paginile

| Adresă | Ce e |
|---|---|
| `/` | Command Center: video live, harta pistei cu sectoare, alertele, bara de stare |
| `/team` | Telefonul echipei de intervenție (alertă roșie, sunet, vibrație, „Am preluat”, „Obiect eliminat”) |
| `/incidents` | Istoricul, cu filtre și cronologia fiecărui incident; export CSV pentru Excel (`;`, virgulă zecimală) |
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

Fără cameră și fără clip real se folosește `data/videos/synthetic_demo.mp4` (desenat de `scripts/setup.py`):
o machetă în care o „mână” lasă un șurub și o agrafă. E doar pentru test, nu pentru cifre.

## Telefonul echipei

1. Laptopul și telefonul în aceeași rețea (cel mai sigur: hotspot de pe un telefon).
2. Portul permis în firewall (o singură dată, PowerShell ca administrator):
   `netsh advfirewall firewall add rule name="RunwayEye 8000" dir=in action=allow protocol=TCP localport=8000`
3. Pe panou: **Telefon echipă** → cod QR cu `http://<IP-laptop>:8000/team`.
   Codul conține și cheia de acces (`?key=…`): fără ea, alte dispozitive din rețea primesc „Acces doar cu codul QR”.
   Laptopul pe care rulează serverul nu are nevoie de cheie. Pentru panou pe alt laptop: `http://<IP-laptop>:8000/?key=<cheia>`
   (cheia e în `data/access_key.txt` sau `ACCESS_KEY` din `.env`).
4. Pe telefon: apăsați o dată **Activează alertele** (browserele blochează sunetul și vibrația până la prima atingere).
   Vibrația merge pe Android (Chrome); iPhone nu permite vibrația din pagini web.

## Calibrarea

`/calibrate` → clic pe colțuri în ordinea: prag 09 stânga, prag 09 dreapta, prag 27 dreapta, prag 27 stânga
(stânga/dreapta privit de la pragul 09). Se salvează în `data/calibration.json`. Dimensiunile pistei
(`RUNWAY_LENGTH_M`, `RUNWAY_WIDTH_M`) și ale machetei (`MOCKUP_*_CM`) sunt în `.env` și **trebuie verificate**.
Fără calibrare, toată imaginea e considerată pistă și alertele nu au sector.

## Detecția pe scurt

- **YOLO** (`models/best.pt`, YOLO11n antrenat de noi pe cele 31 de clase ale setului FOD-A: șurub, piuliță, cheie,
  clește, baterie, piatră, etichetă…; vezi `training/README.md`). Riscul: plasticul, etichetele, lemnul și banda
  sunt **mediu**, restul (metal, unelte, pietre) **critic**. Fără `best.pt` se folosește `yolo11n.pt` pre-antrenat,
  marcat **provizoriu** în bara de stare.
- **Modelul general** (`models/general.pt`, YOLOE cu vocabular deschis) găsește ce nu e în FOD-A: **păsări (și moarte)**,
  câini, pisici, vulpi, iepuri, bucăți de anvelopă, bagaje, pungi și sticle de plastic, cutii, mănuși. Lista e în
  `app/vision/types.py` (`GENERAL_PROMPTS`, în engleză); după ce o schimbați: `python scripts\build_general_model.py`.
  Păsările și animalele au risc **critic**.
- **Detecția schimbărilor** față de pista curată (butonul **Pista e curată acum**) prinde orice alt obiect, ca „obiect necunoscut”.
  Așa nimic nu scapă, nici ce nu știe niciun model.
- **Persistență:** alerta pornește doar după `PERSIST_FRAMES` cadre aproape în același loc: mâna care pune obiectul nu dă alarmă.
- **Obiect mare** (peste `MAX_AREA_PX`): eveniment separat „Vehicul / obiect mare pe pistă”, fără ecran roșu.
- **Ce nu dă alarmă:** umbrele și schimbările de lumină (se compară cât s-a schimbat zona față de asfaltul din jur),
  tremurul camerei de un pixel, ce intră pe la marginea imaginii. Cât timp un braț sau un vehicul intră în cadru,
  nu se confirmă nimic nou; obiectul lăsat apare la ~0,5 s după ce mâna a ieșit.
- **Alarmă falsă:** zona e învățată ca fundal și nu mai declanșează alerta. Dacă alerta venea doar de la YOLO
  (un marcaj sau o lumină luată drept obiect), zona se păstrează și după repornire (`data/ignored_zones.json`);
  un obiect nou căzut acolo dă alertă în continuare, iar o calibrare nouă golește lista.
- Reglaj live (doar cu `DEMO_MODE=true`): **Mai mult → Reglaj detecție**.

## Opriri din `.env`

| Setare | Efect |
|---|---|
| `ENABLE_YOLO=false` | Doar detecția schimbărilor (nu are nevoie de model) |
| `ENABLE_SUPABASE=false` sau `USE_LOCAL_DATA=true` | Fără Supabase: doar copia de pe laptop (`data/runwayeye.db` + poze în `data/images/`) |
| `ENABLE_GENERAL_MODEL=false` | Fără modelul general (păsări, animale…): mai rapid pe laptopuri slabe |
| `ENABLE_CLAUDE=false` | Fără descrierea AI (bonus) |
| `DEMO_MODE=false` | Mod prezentare: fără panou de reglaj și fără mesaje tehnice |
| `DEMO_FAKE_DETECTIONS=true` | Detecții false la ~15 s, ca să arătați panoul fără cameră |

## Supabase: baza de date centrală

Tabelele `cameras`, `incidents`, `incident_events` și bucket-ul `fod-images` (pozele). Fiecare laptop cu cameră
trimite incidentele lui și primește, la ~10 s, incidentele celorlalte camere: istoricul (`/incidents`) arată tot
aeroportul, cu pozele din Supabase Storage.

**De ce și o copie pe laptop:** alerta pentru un obiect pe pistă nu are voie să aștepte internetul. Incidentul se
scrie întâi pe disc (câteva milisecunde), ecranul roșu și telefonul echipei pornesc imediat, iar în 1–2 secunde
rândul și pozele sunt în Supabase. Dacă pică rețeaua, bara de stare arată „N în coadă” și totul pleacă la revenire.

1. Proiect nou: Supabase → SQL Editor → conținutul lui `supabase/schema.sql` → Run. Proiectul existent:
   fișierele din `supabase/migrations/`.
2. În `.env`: `SUPABASE_URL` și `SUPABASE_KEY` = cheia **secretă** (Project Settings → API Keys → Secret keys,
   `sb_secret_…`, sau cheia veche `service_role`). Cheia publică (`sb_publishable_…`) nu poate scrie: tabelele au
   RLS fără politici, deci doar serverul are acces. Cheia rămâne pe laptop; browserul nu vorbește cu Supabase.
3. `python scripts\demo_check.py` verifică cheia, citirea și scrierea.

Fără URL și cheie (de exemplu pe un laptop nou), aplicația merge la fel, doar cu datele de pe laptop.

## Descrierea AI (opțional)

Cu `ENABLE_CLAUDE=true` și `ANTHROPIC_API_KEY` în `.env`, fiecare incident nou primește în câteva secunde o propoziție
scrisă de Claude pe baza pozei decupate („Un șurub metalic pe axul pistei…”). Apare pe card, în detalii și în cronologie.
Rulează în fundal: fără internet alerta pleacă la fel, doar fără descriere.

## Cum demonstrăm că merge

Patru niveluri, de la cel mai rapid la cel care contează pentru juriu.

**1. Teste automate** (secunde): `venv\Scripts\python -m pytest`. Logica de detecție, ciclul incidentului, API-ul.

**2. Scenarii grele, generate** (~3 min): pistă desenată în care știm exact când și unde cade fiecare obiect.

```powershell
python scripts\make_eval_set.py                                     # 10 clipuri în data/eval_synthetic/
python scripts\evaluate.py --dir data\eval_synthetic --no-yolo
```

Testează umbre peste marcaje, lumină care crește lent sau brusc, un vehicul care oprește pe pistă, cameră cu
zgomot și tremur, obiecte mici sau închise la culoare și o mână care trece fără să lase nimic. Sunt bune ca să
arătați că sistemul nu dă alarme la astea, dar cifrele **nu** sunt pentru slide (raportul o și spune).

**3. Modelul YOLO pe imagini pe care nu le-a văzut** (secunde):

```powershell
python scripts\test_samples.py      # 93 de imagini din samples/fod-a-test → data/samples_result/
```

Imaginile sunt din setul de **test** FOD-A (3 pe clasă), separat de antrenare pe blocuri de cadre, deci fără
„aproape-copii” în train. Scriptul folosește exact detectorul aplicației și face `summary.md` (obiect găsit,
clasă exactă, risc corect) și `sheet.jpg`, o planșă cu toate imaginile (verde = clasă corectă). Cifrele pe tot
setul de test (2.917 imagini: mAP50, mAP50-95, precizie, recall, pe fiecare clasă) sunt în `models/metrics.json`,
iar graficele pentru slide (curbele de antrenare, matricea de confuzie, curba precizie-recall, exemple) în
`models/report/`. Refacerea modelului: `training/README.md`.

Rezultatele modelului din repo (YOLO11n, 20 de epoci pe un GTX 1650 Ti, 31 octombrie 2026):

| Set de test FOD-A (2.917 imagini, nevăzute la antrenare) | Valoare |
|---|---|
| mAP50 / mAP50-95 | 0,994 / 0,883 |
| Precizie / recall | 0,991 / 0,994 |
| Cele mai grele clase (mAP50-95) | cui 0,61 · cheie fixă 0,68 · șurub cu piuliță 0,71 |
| `scripts\test_samples.py` (93 de imagini, 3 pe clasă) | 93/93 clasă exactă |

**Ce înseamnă și ce nu:** imaginile de test sunt din aceleași ~55 de clipuri FOD-A (alte momente, fără cadre
vecine), deci pe același tip de asfalt și de aproape. Pe macheta voastră, cu camera sus și obiecte mici, cifrele
vor fi mai mici: pentru juriu contează nivelul 4 de mai jos, măsurat pe clipurile voastre. Modelul general
(păsări, animale) nu e antrenat de noi; l-am verificat pe câteva fotografii (pasăre vie și moartă pe asfalt),
nu pe un set de test, iar uneori confundă un șurub cu șaibă cu o pasăre (vezi `CombinedDetector`).

**4. Clipuri reale cu macheta: cifrele pentru juriu.** Cu camera fixă, în locul și lumina de la prezentare:

| Clip | Ce faceți |
|---|---|
| 3–4 clipuri de 1–2 min | lăsați câte 2–3 obiecte diferite, în sectoare diferite, unul ridicat după un timp |
| 1 clip | lumina se aprinde / se stinge, apoi cade un obiect |
| 1 clip | cineva trece prin fața lămpii (umbră) și o mână trece peste pistă fără să lase nimic |
| 1 clip de 5 min | pistă goală: din el iese rata de alarme false pe oră |

Primele 2 secunde din fiecare clip arată pista curată. Puneți clipurile în `data/eval/`, apoi:

```powershell
python scripts\annotate_clip.py data\eval\macheta_01.mp4   # momentul și zona fiecărui obiect
python scripts\evaluate.py                                 # rata de detecție, alarme false/oră, timp până la alertă
```

Rezultatul e în `data/eval/summary.md`, cu un tabel pe clipuri. Pe slide: rata de detecție, alarmele false pe oră,
timpul până la alertă și mAP-ul din `models/metrics.json`, cu numărul de clipuri și minute din care au ieșit.

## Demonstrația în fața juriului (~5 minute)

Înainte: `python scripts\demo_check.py`, camera fixată pe machetă, **Pista e curată acum**, telefonul echipei
deschis prin codul QR și **Activează alertele** apăsat.

1. **Live:** lăsați un obiect pe machetă → în ~1 s ecran roșu și sunet pe panou, alertă pe telefon cu sectorul
   și distanța pe pistă. Pe telefon: **Am preluat** → **Obiect eliminat**; incidentul apare în `/incidents`.
2. **Ce nu dă alarmă:** treceți mâna peste pistă fără să lăsați nimic, faceți umbră, aprindeți o lampă.
3. **Modelul pe imagini noi:** `/upload` → trageți 2–3 poze din `samples/fod-a-test/` (numele fișierului e
   clasa adevărată) → chenarul, clasa în română, încrederea și riscul.
4. **Cifrele:** `data/samples_result/sheet.jpg` și `summary.md` (de la `test_samples.py`), `models/metrics.json`
   (setul de test complet) și, dacă ați filmat clipurile, `data/eval/summary.md`.

Fără cameră: pasul 1 merge pe clipul demo (sau cu `DEMO_FAKE_DETECTIONS=true`), restul la fel.

## Dezvoltare

```powershell
venv\Scripts\python -m pytest        # teste pe cadre sintetice + API
npm run css:watch                    # recompilează CSS-ul la modificarea șabloanelor
python scripts\browser_check.py http://127.0.0.1:8000/ panou.png --wait 4   # captură headless
```

`npm install` e nevoie doar dacă modificați șabloanele (recompilează `app/static/css/app.css`, care e în repo).

Structura: `app/vision/` (captură, detecție, fuziune, calibrare, desenare), `app/services/` (incidente, risc,
date, SSE, telefonul echipei), `app/routers/`, `app/templates/`, `scripts/`, `tests/`, `training/` (pregătirea
FOD-A, antrenarea locală și în Colab), `samples/` (imagini de test pentru demonstrație).

## Probleme frecvente

- **Imagine neagră / puține cadre pe secundă:** camera are nevoie de lumină (lampă pe machetă); verificați indexul camerei.
- **Alarme la umbre:** lumină constantă, apoi **Pista e curată acum**; măriți pragul din **Reglaj**.
- **Telefonul nu se conectează:** aceeași rețea, regula de firewall, `HOST=0.0.0.0`; `demo_check.py` verifică toate trei.
- **Fără sunet pe panou:** clic oriunde în pagină sau pe **Activează sunetul**.
- **Un obiect ridicat lasă o alertă „necunoscut”:** se întâmplă dacă obiectul era pe pistă când s-a învățat referința.
  Apăsați **Alarmă falsă** (zona se învață) sau **Pista e curată acum** cu pista goală.
