-- Urban Traffic Twin: core tables (spec section 9.1), on Supabase instead of DuckDB.
-- Large raw state logs go to the private Storage bucket `sim-logs` as Parquet;
-- `state_log` holds only the rows the app replays.
-- RLS is on with no policies: only the service role (backend/API) reads and writes.

create table if not exists public.sensor_hourly (
    iu_ac          text        not null,
    libelle        text,
    zone           text,
    ts             timestamptz not null,
    flow_veh_h     real,
    occupancy_pct  real,
    traffic_state  text,
    lat            double precision,
    lon            double precision,
    primary key (iu_ac, ts)
);
create index if not exists sensor_hourly_zone_ts_idx on public.sensor_hourly (zone, ts);

create table if not exists public.demand_profile (
    approach   text    not null,           -- branch of the intersection
    hour       smallint not null check (hour between 0 and 23),
    day_type   text    not null,           -- e.g. tue_wed_thu
    flow_veh_h real    not null check (flow_veh_h >= 0),
    source     text    not null,           -- sensor | field_count | assumed
    primary key (approach, hour, day_type, source)
);

create table if not exists public.run (
    run_id       uuid primary key default gen_random_uuid(),
    scenario     text        not null,
    seed         integer     not null,
    code_version text,
    config       jsonb       not null default '{}'::jsonb,
    started_at   timestamptz not null default now(),
    duration_s   integer,
    state_log_path text                     -- Storage object in sim-logs
);
create index if not exists run_scenario_idx on public.run (scenario, seed);

create table if not exists public.state_log (
    run_id   uuid    not null references public.run (run_id) on delete cascade,
    t        real    not null,              -- simulation time (s)
    entity   text    not null check (entity in ('car', 'bus', 'ambulance', 'pedestrian', 'intersection')),
    id       text    not null,
    x        real,
    y        real,
    speed    real,
    phase    smallint,
    queue    smallint,
    primary key (run_id, t, entity, id)
);

create table if not exists public.trip (
    run_id      uuid not null references public.run (run_id) on delete cascade,
    id          text not null,
    type        text not null,
    depart_s    real,
    arrival_s   real,
    duration_s  real,
    time_loss_s real,
    waiting_s   real,
    stops       smallint,
    primary key (run_id, id)
);

create table if not exists public.kpi_run (
    run_id   uuid not null references public.run (run_id) on delete cascade,
    scenario text not null,
    seed     integer not null,
    user_type text not null,               -- usager: car | bus | ambulance | pedestrian | all
    kpi      text not null,                -- time_loss_mean, wait_p95, headway_std, ...
    value    double precision not null,
    primary key (run_id, user_type, kpi)
);
create index if not exists kpi_run_scenario_idx on public.kpi_run (scenario, kpi);

create table if not exists public.eta_prediction (
    run_id        uuid not null references public.run (run_id) on delete cascade,
    route         text not null,           -- trajet
    t             real not null,
    eta_median_s  real not null,
    eta_low_s     real not null,
    eta_high_s    real not null,
    model_version text not null,
    primary key (run_id, route, t, model_version),
    check (eta_low_s <= eta_median_s and eta_median_s <= eta_high_s)
);

alter table public.sensor_hourly  enable row level security;
alter table public.demand_profile enable row level security;
alter table public.run            enable row level security;
alter table public.state_log      enable row level security;
alter table public.trip           enable row level security;
alter table public.kpi_run        enable row level security;
alter table public.eta_prediction enable row level security;

insert into storage.buckets (id, name, public)
values ('sim-logs', 'sim-logs', false)
on conflict (id) do nothing;
