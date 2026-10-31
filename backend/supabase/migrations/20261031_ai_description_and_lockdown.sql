-- For the existing RunwayEye project (created before 2026-10-31). Supabase → SQL Editor → Run.

-- Claude's one-line description of each incident, so it reaches the central database too.
alter table public.incidents add column if not exists ai_description text;

-- Pulling incidents from the other cameras asks for "changed since"; events are read per incident.
create index if not exists incidents_updated_at on public.incidents (updated_at);
create index if not exists incident_events_incident on public.incident_events (incident_id);

-- Security advisor: this SECURITY DEFINER helper was callable through the public REST API.
revoke execute on function public.rls_auto_enable() from anon, authenticated, public;
