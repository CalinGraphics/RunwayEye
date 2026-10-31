import asyncio
import logging
from contextlib import asynccontextmanager
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.access import load_key, require_key
from app.config import FRONTEND_DIR, get_settings
from app.routers import calibration, control, events, incidents, pages, source, upload, video
from app.services.claude_describe import ClaudeDescriber
from app.services.events import EventBus
from app.services.fake_detections import FakeDetectionGenerator
from app.services.incident_service import IncidentService, InvalidAction
from app.services.supabase_service import DataStore
from app.templating import templates
from app.vision.pipeline import VisionService, runway_geometry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# One line per Supabase request (a pull every 10 s) would bury the useful lines.
for noisy in ("httpx", "httpcore", "hpack"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("runwayeye")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    s.data_path.mkdir(parents=True, exist_ok=True)

    bus = EventBus()
    bus.bind_loop(asyncio.get_running_loop())
    camera = {"id": s.camera_id, "name": s.camera_name, "source": s.video_source, "calibration": None}
    store = DataStore(s, camera)
    incident_service = IncidentService(store, bus, s.camera_id, describer=ClaudeDescriber(s))
    vision = VisionService(s, on_event=incident_service.handle_engine_event)
    incident_service.on_close = vision.close_incident
    if not s.demo_fake_detections:
        incident_service.close_fake()
    vision.restore_incidents(incident_service.open_incidents())

    app.state.settings = s
    app.state.access_key = load_key(s)
    app.state.bus = bus
    app.state.store = store
    app.state.incidents = incident_service
    app.state.vision = vision

    store.start()
    vision.start()
    fake = None
    if s.demo_fake_detections:
        fake = FakeDetectionGenerator(incident_service, runway_geometry(s), s.unknown_critical_cm)
        fake.start()
    log.info("RunwayEye started: source=%s yolo=%s supabase=%s", s.video_source, s.enable_yolo, s.supabase_active)
    try:
        yield
    finally:
        if fake:
            fake.stop()
        vision.stop()
        store.stop()


app = FastAPI(title="RunwayEye", lifespan=lifespan, docs_url=None, redoc_url=None)
app.middleware("http")(require_key)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")

for router in (pages.router, video.router, events.router, incidents.router, control.router, upload.router,
               calibration.router, source.router):
    app.include_router(router)


@app.exception_handler(InvalidAction)
async def invalid_action(_: Request, exc: InvalidAction):
    return JSONResponse({"detail": str(exc)}, status_code=409)


@app.exception_handler(RequestValidationError)
async def bad_request(_: Request, exc: RequestValidationError):
    detail = exc.errors() if get_settings().demo_mode else "Date invalide"
    return JSONResponse({"detail": detail}, status_code=422)


def _wants_page(request: Request) -> bool:
    return request.method == "GET" and "text/html" in request.headers.get("accept", "") and not request.headers.get("HX-Request")


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    messages = {403: "Nu e disponibil în modul prezentare", 404: "Pagina nu există", 405: "Acțiune nepermisă"}
    detail = exc.detail if get_settings().demo_mode else messages.get(exc.status_code, "A apărut o problemă")
    if _wants_page(request):
        title = messages.get(exc.status_code, "A apărut o problemă")
        # Starlette's own details are English status phrases ("Not Found"); ours are Romanian sentences.
        own = isinstance(detail, str) and detail not in (title, HTTPStatus(exc.status_code).phrase)
        message = detail if own else "Verifică adresa sau întoarce-te la panou."
        return templates.TemplateResponse(
            request, "error.html", {"status_code": exc.status_code, "title": title, "message": message},
            status_code=exc.status_code,
        )
    return JSONResponse({"detail": detail}, status_code=exc.status_code)


@app.exception_handler(Exception)
async def server_error(_: Request, exc: Exception):
    log.exception("Unhandled error", exc_info=exc)
    detail = f"Eroare internă: {exc}" if get_settings().demo_mode else "A apărut o problemă. Încercați din nou."
    return JSONResponse({"detail": detail}, status_code=500)
