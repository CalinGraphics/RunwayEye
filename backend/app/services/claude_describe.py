"""Bonus: a one-sentence description of the detected object, written by Claude from the crop.

Runs in its own thread so a slow or missing internet connection never delays the alert.
"""

import base64
import logging
import threading
from typing import Callable

from app.config import Settings

log = logging.getLogger(__name__)

PROMPT = (
    "Imaginea e decupată dintr-o cameră care supraveghează o pistă de aeroport (sau o machetă a ei). "
    "Detectorul a semnalat aici un posibil obiect străin (FOD), clasificat ca „{label}”. "
    "Descrie într-o singură propoziție scurtă, în română, ce obiect se vede și de ce ar putea fi periculos pentru avioane. "
    "Dacă nu se distinge niciun obiect, spune asta. Fără introducere, doar propoziția."
)


class ClaudeDescriber:
    def __init__(self, s: Settings):
        self.model = s.claude_model
        self.enabled = s.enable_claude and bool(s.anthropic_api_key)
        self._client = None
        if self.enabled:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=s.anthropic_api_key, timeout=25.0, max_retries=1)
            except Exception:
                log.exception("Claude client could not be created")
                self.enabled = False

    def describe_later(self, crop_jpeg: bytes, label: str, done: Callable[[str], None]) -> None:
        if not self.enabled or not crop_jpeg:
            return
        threading.Thread(target=self._run, args=(crop_jpeg, label, done), name="claude", daemon=True).start()

    def _run(self, crop_jpeg: bytes, label: str, done: Callable[[str], None]) -> None:
        text = self.describe(crop_jpeg, label)
        if text:
            done(text)

    def describe(self, crop_jpeg: bytes, label: str) -> str | None:
        import anthropic

        try:
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=2000,
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                     "data": base64.standard_b64encode(crop_jpeg).decode("ascii")}},
                        {"type": "text", "text": PROMPT.format(label=label)},
                    ],
                }],
            )
        except anthropic.AuthenticationError:
            log.warning("Claude: ANTHROPIC_API_KEY is not valid")
            return None
        except anthropic.APIStatusError as exc:
            log.warning("Claude: API error %s", exc.status_code)
            return None
        except anthropic.APIConnectionError:
            log.info("Claude: no connection, skipping the description")
            return None
        if response.stop_reason == "refusal":
            return None
        text = " ".join(b.text for b in response.content if b.type == "text").strip()
        return text or None
