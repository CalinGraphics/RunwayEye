const RISK_ORDER = { critic: 0, mediu: 1, "de verificat": 2 };

// Operator flow: the next expected step is the highlighted button, the others stay available.
const NEXT_ACTION = { new: "confirm", confirmed: "dispatch", dispatched: "resolve" };
const ALLOWED_ACTIONS = {
  new: ["confirm", "dispatch", "resolve", "false-alarm"],
  confirmed: ["dispatch", "resolve", "false-alarm"],
  dispatched: ["resolve", "false-alarm"],
};
const ACTION_LABELS = {
  confirm: "Confirm",
  dispatch: "Echipă trimisă",
  resolve: "Obiect eliminat",
  "false-alarm": "Alarmă falsă",
};
const ACTION_BUTTONS = { confirm: "btn-primary", dispatch: "btn-primary", resolve: "btn-success" };

const RE = {
  // Each (re)connection receives a full "hello" state, so nothing is lost while the network is down.
  connect(url, handlers, onState) {
    let es = null;
    let retry = null;
    const open = () => {
      es = new EventSource(url);
      es.onopen = () => onState && onState(true);
      es.onerror = () => {
        onState && onState(false);
        if (es.readyState === EventSource.CLOSED) {
          es.close();
          clearTimeout(retry);
          retry = setTimeout(open, 2000);
        }
      };
      for (const [name, fn] of Object.entries(handlers)) {
        es.addEventListener(name, (e) => fn(JSON.parse(e.data)));
      }
    };
    open();
  },

  async post(url, body) {
    const opts = { method: "POST" };
    if (body !== undefined) {
      opts.headers = { "Content-Type": "application/json" };
      opts.body = JSON.stringify(body);
    }
    let res;
    try {
      res = await fetch(url, opts);
    } catch {
      throw new Error("Fără conexiune la server");
    }
    let data = null;
    try {
      data = await res.json();
    } catch {}
    if (!res.ok) {
      throw new Error(data && typeof data.detail === "string" ? data.detail : "Acțiunea nu a reușit");
    }
    return data;
  },

  // Most dangerous first, then newest.
  byPriority(a, b) {
    const r = (RISK_ORDER[a.risk] ?? 9) - (RISK_ORDER[b.risk] ?? 9);
    return r || Date.parse(b.detected_at) - Date.parse(a.detected_at);
  },

  // Insert, replace or drop `inc` in `list` (in place); returns true if it was dropped.
  merge(list, inc) {
    const idx = list.findIndex((i) => i.id === inc.id);
    if (!inc.is_open) {
      if (idx >= 0) list.splice(idx, 1);
      return idx >= 0;
    }
    if (idx >= 0) list.splice(idx, 1, inc);
    else list.unshift(inc);
    return false;
  },

  elapsed(iso, now) {
    const secs = Math.max(0, Math.floor((now - Date.parse(iso)) / 1000));
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    const s = secs % 60;
    const mm = String(m).padStart(2, "0");
    const ss = String(s).padStart(2, "0");
    return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
  },

  clock(iso) {
    return new Date(iso).toLocaleTimeString("ro-RO", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  },

  riskClass(risk) {
    return { critic: "badge-error", mediu: "badge-warning", "de verificat": "badge-info" }[risk] || "badge-ghost";
  },

  riskTone(risk) {
    return { critic: "error", mediu: "warning" }[risk] || "info";
  },

  statusClass(inc) {
    if (inc.team_ack_at) return "badge-success";
    return { new: "badge-error badge-outline", confirmed: "badge-primary badge-outline", dispatched: "badge-secondary" }[
      inc.status
    ] || "badge-ghost";
  },

  sourceLabel(source) {
    return { yolo: "model YOLO", change: "detecția schimbărilor", both: "YOLO + schimbări" }[source] || source;
  },

  position(inc) {
    if (inc.runway_distance_m == null) return "poziție necalibrată";
    const off = inc.centerline_offset_m;
    const side = Math.abs(off) < 0.5 ? "pe ax" : `${Math.abs(off).toFixed(0)} m ${off < 0 ? "stânga" : "dreapta"} axului`;
    return `${Math.round(inc.runway_distance_m)} m de capătul ${window.RUNWAY_END || "09"} · ${side}`;
  },

  alertText(inc) {
    const parts = ["FOD detectat", inc.object_class];
    if (inc.sector) parts.push(`Sector ${inc.sector}`);
    parts.push(`risc ${inc.risk}`);
    return parts.join(" · ");
  },
};

// Browsers only allow sound after a user gesture: unlock() must run inside a click/tap.
// muted is the operator's choice; unlocked is the browser's permission.
class Alarm {
  constructor(src) {
    this.audio = new Audio(src);
    this.audio.preload = "auto";
    this.unlocked = false;
    this.muted = false;
  }
  async unlock() {
    // Already allowed: don't touch the element, an alarm may be ringing.
    if (this.unlocked) return true;
    try {
      this.audio.muted = true;
      await this.audio.play();
      this.audio.pause();
      this.audio.currentTime = 0;
      this.unlocked = true;
    } catch {}
    this.audio.muted = false;
    return this.unlocked;
  }
  play(times = 2) {
    if (!this.unlocked || this.muted) return;
    clearTimeout(this._cut);
    let n = 0;
    const a = this.audio;
    a.onended = () => {
      if (++n < times) {
        a.currentTime = 0;
        a.play().catch(() => {});
      }
    };
    a.currentTime = 0;
    a.play().catch(() => {});
  }
  // A short sample, so the operator hears that sound works.
  preview() {
    this.play(1);
    this._cut = setTimeout(() => this.stop(), 600);
  }
  stop() {
    clearTimeout(this._cut);
    this.audio.onended = null;
    this.audio.pause();
    this.audio.currentTime = 0;
  }
}

document.addEventListener("alpine:init", () => {
  Alpine.data("commandCenter", (cfg) => ({
    cfg,
    incidents: [],
    status: null,
    connected: false,
    now: Date.now(),
    alarm: null,
    flash: [],
    large: null,
    toasts: [],
    zoomId: null,
    hl: null,
    mview: "alerts", // phone layout: the alerts, or the live runway (camera and map)
    soundOn: false,
    showQr: false,
    qrIp: null, // address picked in the phone window; null = the most likely one (first in the list)
    showTuning: false,
    tuning: cfg.tuning,
    busy: {},
    resetting: false,

    get teamUrls() {
      return (this.status && this.status.team_urls) || [];
    },
    get teamAddress() {
      const list = this.teamUrls;
      return list.find((a) => a.ip === this.qrIp) || list[0] || null;
    },
    get phones() {
      return this.status ? this.status.clients.team : 0;
    },

    init() {
      this.sound = new Alarm("/static/sounds/alarm.wav");
      RE.connect(
        "/events?client=panel",
        {
          hello: (d) => {
            this.incidents = d.incidents;
            this.large = d.large_object;
          },
          "incident.new": (inc) => this.onNew(inc),
          "incident.updated": (inc) => this.onUpdate(inc),
          large_object: (d) => (this.large = d.active ? d : null),
          reference: (d) =>
            this.toast(
              d.reason === "lumina"
                ? "Lumina s-a schimbat: referința pistei a fost refăcută"
                : "Referința se reface: pista e considerată curată"
            ),
          clients: (c) => {
            if (this.status) this.status.clients = c;
          },
        },
        (ok) => (this.connected = ok)
      );
      this.refreshStatus();
      setInterval(() => this.refreshStatus(), 2000);
      setInterval(() => (this.now = Date.now()), 500);
      // The first tap anywhere unlocks audio, so alarms ring even if nobody pressed the sound button.
      // Taps on the button itself are left to toggleSound(), otherwise one tap would unlock and mute.
      const autoUnlock = async (e) => {
        if (e.target.closest("[data-sound-toggle]")) return;
        document.removeEventListener("pointerdown", autoUnlock);
        if ((await this.sound.unlock()) && !this.sound.muted) this.soundOn = true;
      };
      document.addEventListener("pointerdown", autoUnlock);
    },

    async toggleSound() {
      if (this.soundOn) {
        this.sound.muted = true;
        this.sound.stop();
        this.soundOn = false;
        this.toast("Sunet oprit. Alarmele noi nu se vor auzi.");
        return;
      }
      this.sound.muted = false;
      this.soundOn = await this.sound.unlock();
      if (this.soundOn) this.sound.preview();
      else this.toast("Browserul a blocat sunetul. Mai apasă o dată.", "error");
    },

    async refreshStatus() {
      try {
        const res = await fetch("/api/status");
        if (res.ok) this.status = await res.json();
      } catch {}
    },

    onNew(inc) {
      if (!this.incidents.some((i) => i.id === inc.id)) this.incidents.unshift(inc);
      this.mview = "alerts"; // on a phone, a new alert brings its list to the front
      clearTimeout(this._alarmTimer);
      this.alarm = { inc, text: RE.alertText(inc) };
      this._alarmTimer = setTimeout(() => (this.alarm = null), 6000);
      this.sound.play(3);
      if (inc.sector) {
        this.flash = [...this.flash, inc.sector];
        setTimeout(() => (this.flash = this.flash.filter((s) => s !== inc.sector)), 10000);
      }
    },

    onUpdate(inc) {
      if (RE.merge(this.incidents, inc)) this.toast(`Incident închis · ${inc.object_class} · ${inc.status_label}`);
      if (!inc.is_open && this.alarm && this.alarm.inc.id === inc.id) this.dismissAlarm();
    },

    dismissAlarm() {
      clearTimeout(this._alarmTimer);
      this.alarm = null;
      this.sound.stop();
    },

    async act(inc, action) {
      if (this.busy[inc.id]) return;
      this.busy = { ...this.busy, [inc.id]: true };
      try {
        this.onUpdate(await RE.post(`/incidents/${inc.id}/${action}`));
      } catch (e) {
        this.toast(e.message, "error");
      } finally {
        this.busy = { ...this.busy, [inc.id]: false };
      }
    },

    async resetReference() {
      if (this.resetting) return;
      // Everything in view becomes the clean runway: an object still on it would never alert again.
      if (!confirm("Confirmi că pista a fost curățată? Tot ce se vede acum devine fundal, iar orice obiect apărut de acum încolo va fi semnalat.")) return;
      this.resetting = true;
      try {
        await RE.post("/reference");
      } catch (e) {
        this.toast(e.message, "error");
      } finally {
        setTimeout(() => (this.resetting = false), 800);
      }
    },

    async saveTuning() {
      try {
        this.tuning = await RE.post("/tuning", this.tuning);
      } catch (e) {
        this.toast(e.message, "error");
      }
    },

    toast(text, kind = "info") {
      const id = Math.random();
      this.toasts.push({ id, text, kind });
      setTimeout(() => (this.toasts = this.toasts.filter((t) => t.id !== id)), 4500);
    },

    elapsed(inc) {
      return RE.elapsed(inc.detected_at, this.now);
    },
    clock: RE.clock,
    riskClass: RE.riskClass,
    riskTone: RE.riskTone,
    statusClass: RE.statusClass,
    sourceLabel: RE.sourceLabel,
    position: RE.position,

    actions(inc) {
      const next = NEXT_ACTION[inc.status];
      return (ALLOWED_ACTIONS[inc.status] || []).map((a) => ({
        a,
        label: ACTION_LABELS[a],
        primary: a === next,
        cls: a === next ? ACTION_BUTTONS[a] : a === "false-alarm" ? "btn-ghost text-base-content/75 hover:text-error" : "border-line bg-base-100 hover:border-base-content/40",
      }));
    },

    // Detected → confirmed → team → removed; "now" is the step the operator or team should do next.
    steps(inc) {
      const done = { new: 1, confirmed: 2, dispatched: inc.team_ack_at ? 3 : 2 }[inc.status] ?? 4;
      const labels = ["Detectat", "Confirmat", inc.team_ack_at ? "Echipa în drum" : "Echipă", "Eliminat"];
      return labels.map((label, i) => ({ label, state: i < done ? "done" : i === done ? "now" : "todo" }));
    },

    // Both clocks with seconds, in the same HH:MM:SS shape, so they line up.
    get localTime() {
      return new Date(this.now).toLocaleTimeString("ro-RO", { hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23" });
    },
    get utcTime() {
      return new Date(this.now).toISOString().slice(11, 19);
    },

    markerStyle(inc) {
      const g = this.cfg.geometry;
      const u = inc.runway_distance_m / g.length;
      const v = (inc.centerline_offset_m + g.width / 2) / g.width;
      // Must match the runway rectangle inside the SVG viewBox (x 60..940, y 40..160 of 1000x200).
      return `left:${(60 + u * 880) / 10}%;top:${(40 + v * 120) / 2}%`;
    },

    get ordered() {
      return [...this.incidents].sort(RE.byPriority);
    },
    get positioned() {
      return this.incidents.filter((i) => i.runway_distance_m != null);
    },
    get criticalCount() {
      return this.incidents.filter((i) => i.risk === "critic").length;
    },
    get zoom() {
      return this.incidents.find((i) => i.id === this.zoomId) || null;
    },

    // One-glance answer to "is the runway usable?", shown in the top bar.
    get runwayState() {
      const n = this.incidents.length;
      if (n) {
        const top = this.ordered[0].risk;
        return {
          tone: RE.riskTone(top),
          text: `FOD pe pistă · ${n} ${n === 1 ? "alertă" : "alerte"}`,
        };
      }
      if (this.large) return { tone: "warning", text: this.large.label ? this.large.label.charAt(0).toUpperCase() + this.large.label.slice(1) : "Obiect mare pe pistă" };
      return { tone: "success", text: "Pistă liberă" };
    },

    // The footer used to list all of this; now the header shows one dot and a menu.
    get systemRows() {
      const names = { video: "Video", model: "Detecție", data: "Salvare", calib: "Calibrare" };
      return Object.keys(names).map((k) => ({ k, name: names[k], ...this.chip(k) }));
    },
    get systemIssues() {
      return this.status ? this.systemRows.filter((r) => r.dot === "bg-error" || r.dot === "bg-warning") : [];
    },
    get systemDot() {
      if (!this.status) return "bg-base-content/30";
      if (this.systemIssues.some((r) => r.dot === "bg-error")) return "bg-error";
      return this.systemIssues.length ? "bg-warning" : "bg-success";
    },
    get systemLabel() {
      if (!this.status) return "Se conectează…";
      const n = this.systemIssues.length;
      return n ? (n === 1 ? "1 problemă" : `${n} probleme`) : "Sistem OK";
    },

    chip(kind) {
      const s = this.status;
      if (!s) return { dot: "bg-base-content/30", text: "…" };
      if (kind === "video") {
        const v = s.video;
        if (!v.connected) return { dot: "bg-error", text: v.message || "fără semnal" };
        if (v.using_fallback) return { dot: "bg-warning", text: `rezervă: ${v.label}` };
        return { dot: "bg-success", text: v.label };
      }
      if (kind === "model") {
        const y = s.yolo;
        const map = {
          oprit: ["bg-base-content/30", "oprită"],
          "se încarcă": ["bg-warning", "se încarcă…"],
          încărcat: ["bg-success", y.model],
          provizoriu: ["bg-warning", `provizoriu: ${y.model}`],
          eroare: ["bg-error", "modelul nu pornește"],
        };
        const [dot, text] = map[y.state] || ["bg-base-content/30", y.state];
        return { dot, text };
      }
      if (kind === "data") {
        const d = s.data;
        const dot = { conectat: "bg-success", local: "bg-info", oprit: "bg-base-content/30" }[d.state] || "bg-warning";
        return { dot, text: d.label };
      }
      if (kind === "calib") {
        return s.calibrated
          ? { dot: "bg-success", text: "făcută" }
          : { dot: "bg-warning", text: "lipsește" };
      }
      return { dot: "", text: "" };
    },
  }));

  Alpine.data("teamPage", (cfg) => ({
    cfg,
    armed: false,
    connected: false,
    incidents: [],
    now: Date.now(),
    busy: false,
    error: "",

    init() {
      this.sound = new Alarm("/static/sounds/alarm.wav");
      RE.connect(
        "/events?client=team",
        {
          hello: (d) => (this.incidents = d.incidents),
          "incident.new": (inc) => {
            if (!this.incidents.some((i) => i.id === inc.id)) this.incidents.unshift(inc);
            this.ring();
          },
          "incident.updated": (inc) => {
            RE.merge(this.incidents, inc);
            if (this.mode !== "alarm") this.sound.stop();
          },
        },
        (ok) => (this.connected = ok)
      );
      setInterval(() => (this.now = Date.now()), 500);
      // Keep ringing every 10 s while an alert is not taken over.
      setInterval(() => {
        if (this.mode === "alarm") this.ring();
      }, 10000);
    },

    async arm() {
      await this.sound.unlock();
      if (navigator.vibrate) navigator.vibrate(80);
      try {
        if (navigator.wakeLock) await navigator.wakeLock.request("screen");
      } catch {}
      this.armed = true;
      if (this.mode === "alarm") this.ring();
    },

    ring() {
      if (!this.armed) return;
      this.sound.play(3);
      if (navigator.vibrate) navigator.vibrate([500, 200, 500, 200, 900]);
    },

    // Not yet taken over first, then the most dangerous, then the newest.
    get current() {
      const sorted = [...this.incidents].sort(RE.byPriority);
      return sorted.find((i) => !i.team_ack_at) || sorted[0] || null;
    },
    get others() {
      return this.incidents.length - (this.current ? 1 : 0);
    },
    get mode() {
      if (!this.current) return "calm";
      return this.current.team_ack_at ? "enroute" : "alarm";
    },

    async act(action) {
      if (!this.current || this.busy) return;
      this.busy = true;
      this.error = "";
      try {
        RE.merge(this.incidents, await RE.post(`/incidents/${this.current.id}/${action}`));
        this.sound.stop();
        if (navigator.vibrate) navigator.vibrate(0);
      } catch (e) {
        this.error = e.message;
      } finally {
        this.busy = false;
      }
    },

    elapsed(inc) {
      return RE.elapsed(inc.detected_at, this.now);
    },
    position: RE.position,
  }));
});
