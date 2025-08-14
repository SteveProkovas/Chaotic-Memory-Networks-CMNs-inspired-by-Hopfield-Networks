Is that a 10/10 script:
"""
scripts/setup_environment.py
Auto-detect system resources and produce safe runtime environment settings for
chaotic-memory-nets. This script *does not* permanently modify your system by
default. It writes a per-project config file (env_config.json) and can optionally
apply settings to the current process or persist them to the user's shell.

Top-priority hardening applied in this version:
 - logging instead of print, with --verbose flag to enable DEBUG
 - atomic writes + backup when persisting to shell profiles
 - shell-escaping for exported values and refusal to persist suspicious values
 - replace broad excepts with logged exceptions (logging.exception)
 - write env_config.json atomically and fsync for durability
 - a lightweight internal self-test available with --self-test

Usage:
    python scripts/setup_environment.py           # prints recommendations + writes env_config.json
    python scripts/setup_environment.py --dry-run  # only prints, doesn't write
    python scripts/setup_environment.py --apply    # set variables in current os.environ
    python scripts/setup_environment.py --persist  # attempt to persist to shell (requires consent)
    python scripts/setup_environment.py --self-test # run built-in assertions/tests

Note: installing `psutil` is recommended for better detection accuracy; the
script gracefully falls back when it's not available.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from math import floor
from pathlib import Path
from typing import Dict, Optional


# ---------- Constants ----------
GB = 1024 ** 3
MB = 1024 ** 2


# ---------- Utilities ----------


def human_bytes(n: int) -> str:
    if n >= GB:
        return f"{n / GB:.2f} GB"
    if n >= MB:
        return f"{n / MB:.2f} MB"
    return f"{n} bytes"


# ---------- System detection ----------

@dataclass
class SystemInfo:
    platform: str
    os_name: str
    logical_cpus: int
    physical_cpus: Optional[int]
    total_ram: int  # bytes


def detect_system() -> SystemInfo:
    """Detects CPU and RAM. Uses psutil if available, otherwise falls back to
    heuristics and /proc where possible."""
    plat = platform.system()

    # Try psutil for better accuracy
    try:
        import psutil  # type: ignore

        logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
        physical = psutil.cpu_count(logical=False)
        mem = psutil.virtual_memory().total
        return SystemInfo(platform.platform(), plat, logical, physical, mem)
    except Exception as e:
        logging.getLogger(__name__).debug("psutil not available or failed: %s", e)

    logical = os.cpu_count() or 1
    physical = None
    total_ram = None

    # Linux: try /proc/cpuinfo and /proc/meminfo
    if sys.platform.startswith("linux"):
        try:
            # physical cpu detection
            phys_ids = set()
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip().startswith("physical id"):
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            phys_ids.add(parts[1].strip())
            if phys_ids:
                physical = len(phys_ids)
        except Exception:
            logging.getLogger(__name__).debug("Failed to parse /proc/cpuinfo")

        try:
            with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        # MemTotal is in kB
                        kb = int(parts[1])
                        total_ram = kb * 1024
                        break
        except Exception:
            logging.getLogger(__name__).debug("Failed to parse /proc/meminfo")

    # macOS fallback
    if total_ram is None and sys.platform == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()
            total_ram = int(out)
        except Exception:
            logging.getLogger(__name__).debug("Failed to read hw.memsize via sysctl")

    # Windows fallback
    if total_ram is None and plat == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_ram = int(stat.ullTotalPhys)
        except Exception:
            logging.getLogger(__name__).debug("Failed to query memory via ctypes on Windows")

    # final fallback: assume 8GB if unknown
    if total_ram is None:
        total_ram = 8 * GB

    if physical is None:
        # heuristic: assume half of logical if >=2, else 1
        physical = logical // 2 or 1

    return SystemInfo(platform.platform(), plat, logical, physical, total_ram)


# ---------- Recommendation logic ----------

@dataclass
class Recommendation:
    num_threads: int
    num_threads_physical: Optional[int]
    memory_limit_bytes: int
    reserved_for_system_bytes: int
    env_vars: Dict[str, str]
    notes: str


def compute_recommendation(sysinfo: SystemInfo) -> Recommendation:
    logical = max(1, sysinfo.logical_cpus or 1)
    physical = sysinfo.physical_cpus or (logical // 2 if logical >= 2 else 1)

    # Reserve at least 1 CPU for the OS; conservative policy
    threads = max(1, physical - 1)

    total_ram = sysinfo.total_ram
    if total_ram >= 16 * GB:
        reserve = 2 * GB
    elif total_ram >= 8 * GB:
        reserve = int(1.5 * GB)
    elif total_ram >= 4 * GB:
        reserve = int(1 * GB)
    else:
        reserve = int(512 * MB)

    usable = max(int(256 * MB), total_ram - reserve)
    memory_limit = int(usable * 0.9)

    env_vars = {
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "NUMBA_NUM_THREADS": str(threads),
        "VECLIB_MAXIMUM_THREADS": str(max(1, threads)),
        "CMN_NUM_WORKER_THREADS": str(max(1, threads)),
        "CMN_MEMORY_LIMIT_BYTES": str(memory_limit),
    }

    notes = (
        "Recommended conservative defaults. Use --apply to set these for the current "
        "process, or --persist to try to persist them for future shells. When persisting, "
        "the script backs up the target profile and writes atomically."
    )

    return Recommendation(
        num_threads=threads,
        num_threads_physical=physical,
        memory_limit_bytes=memory_limit,
        reserved_for_system_bytes=reserve,
        env_vars=env_vars,
        notes=notes,
    )


# ---------- Safe persistence helpers ----------


def _shell_escape_val(val: str) -> str:
    # Safely escape single quotes for use inside single-quoted shell export
    return "'" + val.replace("'", "'\"'\"'") + "'"


def _is_suspicious_value(val: str) -> bool:
    # heuristics to avoid accidentally persisting secrets
    if not isinstance(val, str):
        return True
    if len(val) > 2000:
        return True
    if "
" in val:
        return True
    low = val.lower()
    if "private key" in low or "begin rsa private" in low or "-----begin" in low:
        return True
    return False


def _atomic_write(path: Path, content: str) -> None:
    # write to a temp file in the same directory and atomically replace
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    finally:
        # ensure temp removed if replacement failed
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                logging.getLogger(__name__).debug("Failed to remove temp file %s", tmp_path)


# ---------- Apply / persist ----------

def apply_to_current_env(env: Dict[str, str]) -> None:
    for k, v in env.items():
        os.environ[k] = v


def persist_to_shell(env: Dict[str, str]) -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    system = platform.system()

    # pre-filter suspicious values
    for k, v in env.items():
        if _is_suspicious_value(v):
            logging.getLogger(__name__).warning("Refusing to persist suspicious value for %s", k)
            results[k] = False
    
    safe_env = {k: v for k, v in env.items() if not _is_suspicious_value(v)}

    if system in ("Linux", "Darwin"):
        home = Path.home()
        rc = home / ".bashrc"
        if not rc.exists():
            rc = home / ".profile"

        marker_start = "# >>> chaotic-memory-nets environment (start) >>>"
        marker_end = "# <<< chaotic-memory-nets environment (end) <<<"

        exports = [f"export {k}={_shell_escape_val(v)}" for k, v in safe_env.items()]
        block = "
" + marker_start + "
" + "
".join(exports) + "
" + marker_end + "
"

        try:
            # backup
            try:
                if rc.exists():
                    backup = rc.with_suffix(rc.suffix + ".cmn.bak")
                    shutil.copy2(rc, backup)
                    logging.getLogger(__name__).info("Backed up %s to %s", rc, backup)
            except Exception:
                logging.getLogger(__name__).warning("Backup of %s failed", rc)

            text = rc.read_text(encoding="utf-8") if rc.exists() else ""
            if marker_start in text:
                import re

                new_text = re.sub(r"# >>> chaotic-memory-nets environment \(start\) >>>.*?# <<< chaotic-memory-nets environment \(end\) <<<
",
                                  block, text, flags=re.S)
            else:
                new_text = text + block

            _atomic_write(rc, new_text)
            for k in safe_env.keys():
                results[k] = True
        except Exception:
            logging.getLogger(__name__).exception("Failed to persist to shell profile %s", rc)
            for k in safe_env.keys():
                results[k] = False

    elif system == "Windows":
        for k, v in safe_env.items():
            try:
                # setx persists into user environment for future processes. Note: limited length.
                subprocess.check_call(["setx", k, v], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                results[k] = True
            except Exception:
                logging.getLogger(__name__).exception("Failed to persist %s via setx", k)
                results[k] = False
    else:
        for k in safe_env.keys():
            results[k] = False

    return results


# ---------- Config file ----------

def write_config_file(reco: Recommendation, path: Path) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_commit_if_any(),
        "user": os.getenv("USER") or os.getenv("USERNAME") or "unknown",
        "hostname": platform.node(),
        "num_threads": reco.num_threads,
        "num_threads_physical": reco.num_threads_physical,
        "memory_limit_bytes": reco.memory_limit_bytes,
        "reserved_for_system_bytes": reco.reserved_for_system_bytes,
        "env_vars": reco.env_vars,
        "notes": reco.notes,
    }
    content = json.dumps(payload, indent=2)
    _atomic_write(path, content)


def _get_git_commit_if_any() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


# ---------- CLI / main ----------

def _configure_logging(verbose: bool, logfile: Optional[Path] = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    if logfile:
        handlers.append(logging.FileHandler(str(logfile), encoding="utf-8"))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s", handlers=handlers)


def _self_test() -> int:
    """Run lightweight assertions to catch regressions without requiring pytest.
    These are intentionally simple smoke tests.
    """
    log = logging.getLogger(__name__)
    try:
        si = detect_system()
        assert isinstance(si.total_ram, int) and si.total_ram > 0
        reco = compute_recommendation(si)
        assert reco.memory_limit_bytes < si.total_ram
        assert int(reco.env_vars.get("OMP_NUM_THREADS", "1")) >= 1
        log.info("Self-test passed: system detection and recommendation basic checks OK")
        return 0
    except AssertionError as e:
        log.exception("Self-test assertion failed")
        return 2
    except Exception:
        logging.getLogger(__name__).exception("Self-test encountered an unexpected error")
        return 3


def main(argv=None):
    p = argparse.ArgumentParser(description="Detect system resources and produce safe environment settings for chaotic-memory-nets.")
    p.add_argument("--dry-run", action="store_true", help="Only print recommendations; don't write files or apply env vars")
    p.add_argument("--apply", action="store_true", help="Apply recommended environment variables to current process")
    p.add_argument("--persist", action="store_true", help="Attempt to persist environment variables for future shells (will modify shell profile or use setx on Windows)")
    p.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]), help="Project root to write env_config.json into")
    p.add_argument("--verbose", action="store_true", help="Enable verbose (debug) logging")
    p.add_argument("--self-test", action="store_true", help="Run built-in smoke tests and exit")
    args = p.parse_args(argv)

    log_path = Path(args.project_root) / "setup_environment.log"
    _configure_logging(args.verbose, logfile=log_path)
    log = logging.getLogger(__name__)

    if args.self_test:
        return _self_test()

    log.info("Detecting system...")
    try:
        sysinfo = detect_system()
        reco = compute_recommendation(sysinfo)
    except Exception:
        log.exception("Failed during system detection or recommendation computation")
        return 1

    log.info("System summary: %s (%s)", sysinfo.platform, sysinfo.os_name)
    log.info("Logical CPUs: %d", sysinfo.logical_cpus)
    log.info("Physical CPUs: %s", str(sysinfo.physical_cpus))
    log.info("Total RAM: %s", human_bytes(sysinfo.total_ram))

    log.info("Recommended settings: threads=%d reserved=%s memory_limit=%s", reco.num_threads, human_bytes(reco.reserved_for_system_bytes), human_bytes(reco.memory_limit_bytes))

    for k, v in reco.env_vars.items():
        log.info("ENV: %s=%s", k, (human_bytes(int(v)) if k.endswith("BYTES") else v if len(v) < 200 else v[:200] + "..."))

    project_root = Path(args.project_root)
    cfg_path = project_root / "env_config.json"

    if args.dry_run:
        log.info("Dry-run requested: not writing files or changing your environment.")
        return 0

    try:
        project_root.mkdir(parents=True, exist_ok=True)
        write_config_file(reco, cfg_path)
        log.info("Wrote recommended config to: %s", cfg_path)
    except Exception:
        log.exception("Failed to write config file")

    if args.apply:
        try:
            apply_to_current_env(reco.env_vars)
            log.info("Applied environment variables to current process (os.environ).")
        except Exception:
            log.exception("Failed to apply environment variables to current process")

    if args.persist:
        log.info("Attempting to persist environment variables for future shells...")
        try:
            results = persist_to_shell(reco.env_vars)
            for k, ok in results.items():
                log.info("Persist %s: %s", k, "OK" if ok else "FAILED")
            log.info("If any persist operations failed, consider setting variables manually or run the script with elevated permissions.")
        except Exception:
            log.exception("Persist operation failed unexpectedly")

    log.info("Done. Next steps: run experiments with the config at env_config.json. Do not rely on persisted shell vars in automated CI runs.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
