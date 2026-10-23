"""`tasks.py demo-pack`: docs/demo/demo_pack.zip + docs/demo/MANIFEST.json (D116, Q38).

What a fresh clone needs for the demo and cannot get from git (everything under data/ is
gitignored, the ETA model is 244 MB):

- the demo runs at seed 4000 (configs/demo.yaml `pack.runs`), without their SUMO XML (the
  API never reads it): run.json, kpi_run.csv and the Parquet tables, ETA files included;
- every evaluation output (`data/results/evaluation/**`: Part 4 tables, pairs, and after
  Part 8 final_table.csv, a06.json, series.csv, digest.json);
- the ETA bundle files the API reads: LATEST, freeze.json, <version>/{reference.parquet,
  meta.json, eval.json} (never model.ubj).

The zip is deterministic (sorted entries, fixed timestamps), so repacking identical inputs
gives identical bytes. MANIFEST.json (also stored inside the zip) lists every file with its
sha256 and size, the code version and the source commit. No run on an evaluation seed is
packed: only the evaluation's aggregate outputs.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from traffic_twin.demo.common import DemoError, git_info
from traffic_twin.paths import DEMO_PACK

MANIFEST = "MANIFEST.json"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
SCHEMA = 1
ALLOWED_PREFIXES = ("data/demo/", "data/results/evaluation/", "data/models/eta/")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_files(
    cfg: dict, root: Path, *, allow_missing: bool = False
) -> tuple[list[str], list[str]]:
    """(repo-relative files to pack, problems). Problems are missing inputs; with
    `allow_missing` the missing demo runs are skipped instead."""
    p = cfg.get("pack") or {}
    root = Path(root)
    files: list[str] = []
    problems: list[str] = []
    runs_dir = Path(p.get("runs_dir", "data/demo"))
    suffixes = tuple(p.get("run_suffixes") or (".parquet", ".csv", ".json"))
    seed = int(p.get("seed", 4000))
    for scenario in p.get("runs") or []:
        d = runs_dir / f"{scenario}_seed{seed}"
        if not (root / d / "run.json").is_file():
            if not allow_missing:
                problems.append(f"demo run {d.as_posix()} not found")
            continue
        files += sorted(
            f.relative_to(root).as_posix()
            for f in (root / d).iterdir()
            if f.is_file() and f.suffix in suffixes
        )
    ev = Path(p.get("evaluation_dir", "data/results/evaluation"))
    if not (root / ev / "main" / "kpis.csv").is_file():
        problems.append(f"evaluation outputs {ev.as_posix()}/main/kpis.csv not found")
    else:
        files += sorted(
            f.relative_to(root).as_posix() for f in (root / ev).rglob("*") if f.is_file()
        )
    eta = Path(p.get("eta_dir", "data/models/eta"))
    latest = root / eta / "LATEST"
    if not latest.is_file():
        problems.append(f"ETA bundle {eta.as_posix()}/LATEST not found")
    else:
        version = latest.read_text(encoding="utf-8").strip()
        for name in p.get("eta_files") or ["LATEST"]:
            if (root / eta / name).is_file():
                files.append((eta / name).as_posix())
        for name in p.get("eta_bundle_files") or ["reference.parquet"]:
            f = eta / version / name
            if (root / f).is_file():
                files.append(f.as_posix())
            elif name == "reference.parquet":
                problems.append(f"ETA reference table {f.as_posix()} not found")
    bad = [f for f in files if not f.startswith(ALLOWED_PREFIXES) or f.endswith(".xml")]
    problems += [f"refusing to pack {f}" for f in bad]
    return sorted(set(files)), problems


def pack(
    cfg: dict, out: Path = DEMO_PACK, *, root: Path | None = None, allow_missing: bool = False
) -> Path:
    """Write the pack `out` and `<out dir>/MANIFEST.json`; returns `out`. DemoError when an
    input is missing (unless `allow_missing`, for the demo runs only)."""
    from traffic_twin.api.settings import find_repo_root

    root = Path(root or find_repo_root()).resolve()
    out = Path(out) if Path(out).is_absolute() else root / out
    files, problems = pack_files(cfg, root, allow_missing=allow_missing)
    if problems:
        raise DemoError("missing inputs: " + "; ".join(problems))
    info = git_info(root)
    entries = [
        {"path": f, "sha256": sha256(root / f), "bytes": (root / f).stat().st_size} for f in files
    ]
    manifest = {
        "schema": SCHEMA,
        "what": "demo pack (D116): demo runs at seed 4000 without XML, evaluation outputs, "
        "ETA reference bundle. Simulated data only.",
        "code_version": info["code_version"],
        "commit": info["full"],
        "files": entries,
        "total_bytes": sum(e["bytes"] for e in entries),
    }
    body = (json.dumps(manifest, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for e in entries:
            zi = zipfile.ZipInfo(e["path"], date_time=ZIP_DATE)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, (root / e["path"]).read_bytes())
        zi = zipfile.ZipInfo(MANIFEST, date_time=ZIP_DATE)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = 0o644 << 16
        z.writestr(zi, body)
    tmp.replace(out)
    (out.parent / MANIFEST).write_bytes(body)
    return out


def read_manifest(pack_path: Path) -> dict:
    with zipfile.ZipFile(pack_path) as z:
        try:
            return json.loads(z.read(MANIFEST).decode("utf-8"))
        except KeyError as e:
            raise DemoError(f"{pack_path}: no {MANIFEST} inside the pack") from e


def verify(pack_path: Path, manifest_path: Path | None = None) -> dict:
    """The pack's manifest after checking every entry (listed, safe path, sha256, size)
    and, when given, that `manifest_path` equals the one inside. DemoError otherwise."""
    pack_path = Path(pack_path)
    if not pack_path.is_file():
        raise DemoError(f"demo pack {pack_path} not found (python tasks.py demo-pack)")
    try:
        z = zipfile.ZipFile(pack_path)
    except zipfile.BadZipFile as e:
        raise DemoError(f"{pack_path} is not a zip file") from e
    with z:
        try:
            manifest = json.loads(z.read(MANIFEST).decode("utf-8"))
        except KeyError as e:
            raise DemoError(f"{pack_path}: no {MANIFEST} inside the pack") from e
        listed = {e["path"]: e for e in manifest.get("files") or []}
        names = set(z.namelist()) - {MANIFEST}
        problems = []
        if names != set(listed):
            problems.append(
                f"entries differ from the manifest: extra {sorted(names - set(listed))[:5]},"
                f" missing {sorted(set(listed) - names)[:5]}"
            )
        for name in sorted(names & set(listed)):
            if not safe_member(name):
                problems.append(f"unsafe path {name!r}")
                continue
            data = z.read(name)
            e = listed[name]
            if len(data) != e["bytes"] or hashlib.sha256(data).hexdigest() != e["sha256"]:
                problems.append(f"{name}: sha256 or size differs from the manifest")
    if manifest_path is not None and Path(manifest_path).is_file():
        outside = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if outside != manifest:
            problems.append(f"{manifest_path} differs from the manifest inside the pack")
    if problems:
        raise DemoError("demo pack: " + "; ".join(problems))
    return manifest


def safe_member(name: str) -> bool:
    """A relative path under one of the packable folders, without `..` or a drive."""
    p = Path(name)
    return (
        not p.is_absolute()
        and ":" not in name
        and "\\" not in name
        and ".." not in p.parts
        and name.startswith(ALLOWED_PREFIXES)
    )
