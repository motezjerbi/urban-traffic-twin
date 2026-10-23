"""`tasks.py demo`: start the demo (D116) and block until Ctrl+C.

1. prerequisites (configs/demo.yaml), each with a fix hint; a `missing` one stops here
   (`--allow-missing` goes on), a `pending` one (a Part 8 output) is only reported;
2. impostor and port checks: 8000 must be free (a server whose /openapi.json title is not
   "Urban Traffic Twin API", such as run_mock.py, is named as an impostor), 8501 too;
3. uvicorn on 127.0.0.1:8000 and Streamlit (`app.py`) on 127.0.0.1:8501, each in its own
   process group, pids in data/demo_check/.pids.json;
4. a quick API check of every step;
5. opens /mock?view=home in the browser (unless `--no-browser`);
6. blocks until Ctrl+C (or `--exit-after S`), then stops both trees (Windows: CTRL_BREAK,
   then `taskkill /T /F`) and checks that the ports are free.

No simulation runs during the demo (spec 7.1).
"""

from __future__ import annotations

import shutil
import tempfile
import time
import webbrowser
from pathlib import Path

from traffic_twin.demo.common import (
    PIDS_FILE,
    DemoError,
    ProcessGroup,
    api_cmd,
    check_dir,
    child_env,
    operator_cmd,
    probe_api,
    probe_streamlit,
    wait_http,
)
from traffic_twin.demo.config import api_base, operator_base
from traffic_twin.demo.prereq import format_rows, prerequisites
from traffic_twin.paths import DEMO_CHECK_DIR


def port_checks(cfg: dict, api_port: int, op_port: int, operator: bool) -> list[str]:
    """Problems with the demo ports (busy, or an impostor on the API port)."""
    problems = []
    base = api_base(cfg, api_port)
    kind, detail = probe_api(base, cfg["api"]["title"])
    if kind == "impostor":
        problems.append(
            f"port {api_port} is held by another server ({detail}). It is not the "
            "real API: stop it (run_mock.py serves invented data) and retry."
        )
    elif kind == "api":
        problems.append(
            f"the real API already runs on port {api_port}: stop it first, or "
            f"check it with `python -m traffic_twin.demo check --url {base}`"
        )
    if operator:
        k, d = probe_streamlit(operator_base(cfg, op_port))
        if k != "free":
            problems.append(
                f"port {op_port} is busy ({k}: {d}); stop that server or "
                "`python -m traffic_twin.demo clean`"
            )
    return problems


def serve(
    cfg: dict,
    *,
    open_browser: bool = True,
    root: Path | None = None,
    api_config: Path | None = None,
    api_port: int | None = None,
    operator_port: int | None = None,
    operator: bool = True,
    allow_missing: bool = False,
    fixtures: bool = False,
    exit_after: float | None = None,
) -> int:
    """Exit status: 0 after a clean stop, 1 when a check refused to start or a child
    process died, 2 when a port was not freed."""
    from traffic_twin.api.settings import find_repo_root, load_settings
    from traffic_twin.demo.check import check
    from traffic_twin.demo.rehearsal import api_config_path, build_fixtures

    root = Path(root or find_repo_root()).resolve()
    api_port = api_port or int(cfg["api"]["port"])
    op_port = operator_port or int(cfg["operator"]["port"])
    work = Path(tempfile.mkdtemp(prefix="tt_demo_serve_")) if fixtures else None
    cfg_path = build_fixtures(work) if work else api_config_path(root, api_config)
    settings = load_settings(cfg_path, repo_root=root)
    print("Prerequisites" + (" (FAKE fixtures)" if fixtures else "") + ":")
    rows = prerequisites(cfg, settings)
    print(format_rows(rows))
    if any(r["status"] == "missing" for r in rows) and not allow_missing:
        print("\nA prerequisite is missing: fix it (hints above) or pass --allow-missing.")
        return 1
    problems = port_checks(cfg, api_port, op_port, operator)
    if problems:
        print("\n".join(["", "Cannot start:", *(f"- {p}" for p in problems)]))
        return 1
    out = check_dir(root, "serve")
    group = ProcessGroup(root / DEMO_CHECK_DIR / PIDS_FILE, stop_s=float(cfg["budgets"]["stop_s"]))
    base, op_url = api_base(cfg, api_port), operator_base(cfg, op_port)
    status = 0
    try:
        env = child_env(TT_API_CONFIG=str(cfg_path), TT_RESULTS_DIRS=None, TT_API_INPROCESS=None)
        api = group.start(
            "api",
            api_cmd(api_port, cfg["api"]["host"]),
            cwd=root,
            env=env,
            port=api_port,
            log=out / "api.log",
        )
        up = wait_http(base + "/health", timeout_s=cfg["budgets"]["api_start_s"], proc=api.proc)
        print(f"\nAPI up in {up:.1f} s: {base} (log {out / 'api.log'})")
        if operator:
            op_env = child_env(TT_API_URL=base, TT_API_INPROCESS=None, TT_API_CONFIG=None)
            st = group.start(
                "operator",
                operator_cmd(cfg["operator"]["script"], op_port, cfg["operator"]["host"]),
                cwd=root,
                env=op_env,
                port=op_port,
                log=out / "operator.log",
            )
            up = wait_http(
                op_url + "/_stcore/health",
                timeout_s=cfg["budgets"]["operator_start_s"],
                proc=st.proc,
                expect="ok",
            )
            print(f"Operator (Streamlit) up in {up:.1f} s: {op_url}")
        quick = check(
            cfg,
            base,
            browser=False,
            operator_url=op_url if operator else None,
            root=root,
            quiet=True,
        )
        s = quick["summary"]
        print(
            f"Quick check: {quick['status']} ({s['pass']} PASS, {s['fail']} FAIL, "
            f"{s['pending']} PENDING)"
        )
        for r in quick["rows"]:
            if r["result"] in ("FAIL", "PENDING"):
                print(f"  [{r['result']}] {r['group']}: {r['check']} {r['detail']}")
        home = f"{base}/mock?view=home"
        if operator and op_port != int(cfg["operator"]["port"]):
            home += f"&operator={op_url}"
        if operator:
            print(f"\nDemo: {home}\nOperator: {op_url}\nCtrl+C stops both.")
        else:
            print(f"\nDemo: {home}\n(Operator not started: --no-operator)\nCtrl+C stops the API.")
        if open_browser:
            webbrowser.open(home)
        t0 = time.monotonic()
        while True:
            dead = group.dead()
            if dead:
                print(f"\n{dead[0].name} exited with code {dead[0].proc.returncode}; stopping.")
                status = 1
                break
            if exit_after is not None and time.monotonic() - t0 >= exit_after:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping ...")
    except DemoError as e:
        print(f"\nCannot start: {e}")
        status = 1
    finally:
        freed = group.stop()
        busy = [n for n, ok in freed.items() if not ok]
        if busy:
            print(f"Ports still held after the stop: {busy} (python -m traffic_twin.demo clean)")
            status = status or 2
        else:
            print("Stopped; ports freed.")
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)
    return status
