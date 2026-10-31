from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app.services.team_alerts import team_url
from app.templating import templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def command_center(request: Request):
    st = request.app.state
    return templates.TemplateResponse(
        request,
        "command_center.html",
        {
            "geometry": st.vision.geometry,
            "team_url": team_url(st.settings.port, st.access_key),
            "tuning": st.vision.status()["tuning"],
        },
    )


@router.get("/team", response_class=HTMLResponse)
def team(request: Request):
    return templates.TemplateResponse(request, "team.html", {})
