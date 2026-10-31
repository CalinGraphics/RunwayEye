-- RunwayEye: full schema for a NEW Supabase project (SQL Editor → paste → Run). Safe to run again.
-- An existing project only needs the files in supabase/migrations/.
--
-- RLS is on and there are no policies on purpose: only the server-side secret key (sb_secret_… /
-- service_role, kept in .env on the laptop) can read or write. The browser never talks to Supabase.

create table if not exists public.cameras (
  id          text primary key,
  name        text,
  source      text,
  calibration jsonb,
  updated_at  timestamptz not null default now()
);

create table if not exists public.incidents (
  id                  uuid primary key,
  camera_id           text references public.cameras (id),
  detected_at         timestamptz not null,
  status              text not null check (status in ('new', 'confirmed', 'dispatched', 'resolved', 'false_alarm')),
  source              text not null check (source in ('yolo', 'change', 'both')),
  object_class        text not null,
  class_key           text,
  confidence          real,
  risk                text not null check (risk in ('critic', 'mediu', 'de verificat')),
  bbox                jsonb,
  center_px           jsonb,
  sector              text,
  runway_distance_m   real,
  centerline_offset_m real,
  size_estimate_cm    real,
  snapshot_url        text,
  crop_url            text,
  object_present      boolean not null default true,
  confirmed_at        timestamptz,
  dispatched_at       timestamptz,
  team_ack_at         timestamptz,
  resolved_at         timestamptz,
  false_alarm_at      timestamptz,
  updated_at          timestamptz,
  ai_description      text
);

create table if not exists public.incident_events (
  id          uuid primary key,
  incident_id uuid not null references public.incidents (id),
  at          timestamptz not null,
  action      text not null,
  note        text
);

create index if not exists incidents_detected_at on public.incidents (detected_at);
create index if not exists incidents_updated_at on public.incidents (updated_at);
create index if not exists incident_events_incident on public.incident_events (incident_id);

alter table public.cameras enable row level security;
alter table public.incidents enable row level security;
alter table public.incident_events enable row level security;

-- Snapshots and crops. Public read, so the history can show pictures from every camera.
insert into storage.buckets (id, name, public)
values ('fod-images', 'fod-images', true)
on conflict (id) do nothing;
