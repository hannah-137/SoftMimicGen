"""State of a dataset folder (made by make_dataset.py): one row per demo number, a summary and the next commands.

  python experiments/policy_data/status.py <dataset_dir>

It reads every list file of the dataset and the files on disk, and writes two files, both made new on every run:
  dataset.csv   one row per demo number: source demo, place type, reference image, its spec version and its check,
                video and its review, and the state (what the number needs next)
  videos.csv    one row per Cosmos video, current and rejected, with the spec version it was made with
Then it prints the count per state, per group of 50, per place type, per spec version, the problems and the next
commands.
make_references.py, checks/check_references.py, make_dataset.py and run_cosmos.py call it at the end, so dataset.csv
stays current.

States, in the order they are handled:
  conflict        more than one reference image or video for the number, its image is in the wrong folder, or its
                  frame-0 image or geoedge video is missing: fix by hand (the problem list says what)
  needs_replace   the simulator demo looks wrong, or its reference image or video failed max_attempts times:
                  make_dataset.py --replace puts another demo at this number. Also a replace that stopped halfway
                  (the column note says so): the same --replace command finishes it
  needs_ref       no reference image yet
  needs_check     the reference image has no check result (the check stores the sha1 of the file it checked)
  ref_failed      the reference image failed the check, attempts are left
  needs_video     no Cosmos video yet
  needs_review    the video has no review yet, or it is marked weak or rejected without a reason (note "no reason")
  video_rejected  the video was rejected with a reason, attempts are left
  approved        the video was approved, or marked weak (usable but weak) with a reason
The column note says "replace running", "replace stopped halfway" or "cosmos running (<run>)" when a job works on
the number; the next commands leave such numbers out.

Every number keeps its place in the ratios (place type, strong light slot), because they come from the number, not
from the image. A number is done only when its video is approved (or weak, with a reason); nothing is dropped. The
column review keeps "weak", so weak videos can be counted and left out later. The variation seed, the tag of the
reference images and max_attempts are fixed per dataset in its run_config.json.

It is safe to run at any time, also while other scripts run. Every script that writes a list file or moves a
tracked file holds the dataset lock (<dataset>/.lists.lock, flock) for a short time; this script takes it too and
waits while another script holds it. The lock needs a local file system (flock).

This file also holds the helpers that the other scripts import: lock(), lock_fd(), read_csv(), write_csv(),
append_csv(), write_json(), parse_demos(), ranges(), current_sources(), dataset_config(), file_sha1(),
check_is_current(), open_hdf5(), stop_signals(), running_cosmos(), cosmos_claims(), reason_ids() and has_reason().
The review rules are here too: VIDEO_VERDICTS, NEEDS_REASON and REASONS (the reason list of the review page).
SPEC is the spec version (specs/<version>.json) of new reference images when --spec is not given; row_spec() gives
the version of a row of refs/references.csv.
"""

import argparse
import collections
import contextlib
import csv
import fcntl
import glob
import hashlib
import json
import os
import re
import signal
import sys
import threading
import time

import layout
import variations

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = "v2"  # spec version (specs/<version>.json) of new reference images when --spec is not given
UNRECORDED_SPEC = "v2"  # images made before refs/references.csv had the column spec (2026-09-29) follow v2
MAX_ATTEMPTS = 15  # reference images per demo, and videos per demo, before the demo is replaced
TAG = "01"  # a dataset has one reference image per demo: demo_NNN_01.png
VARIATION_SEED = 0  # seed of variations.py for the reference images of a dataset
LISTS_LOCK = ".lists.lock"  # short: held while a list file is written or a tracked file is moved
REFS_LOCK = ".make_references.lock"  # long: held by make_references.py for the whole run
REPLACE_LOCK = ".replace.lock"  # long: held by make_dataset.py --replace for the whole replace
RUNNING_LOCK = ".running.lock"  # in a Cosmos run folder: held by its run_cosmos.py until it ends
H5_WAIT = 600  # seconds to wait while another process has an hdf5 file open

SOURCE_COLUMNS = ["demo", "source_run", "source_demo", "seed", "camera_noise_pos_m", "camera_noise_rot_deg", "steps"]
REPLACEMENT_COLUMNS = ["demo", "replacement", "old_source_run", "old_source_demo", "new_source_run", "new_source_demo",
                       "reason", "folder", "replaced_at"]
REF_REJECTED_COLUMNS = ["demo", "name", "attempt", "source", "file", "room_score", "wrist_score", "reason",
                        "rejected_at"]
REVIEW_COLUMNS = ["video", "demo", "name", "source", "verdict", "ref_problem", "reasons", "review", "reviewed_at"]
SIM_REVIEW_COLUMNS = ["demo", "source", "verdict", "review", "reviewed_at"]
DATASET_COLUMNS = ["demo", "group", "source", "place_type", "strong_light", "sim_review", "reference", "spec",
                   "ref_attempts", "ref_check", "room_score", "wrist_score", "refs_rejected", "video",
                   "videos_rejected", "review", "ref_problem", "reasons", "review_text", "replaced", "state", "note"]
VIDEO_COLUMNS = ["demo", "name", "cosmos_run", "file", "where", "source", "spec", "verdict", "ref_problem",
                 "reasons", "review_text"]
REVIEW_HISTORY_COLUMNS = ["kind", "video", "demo", "name", "source", "verdict", "ref_problem", "reasons", "review",
                          "reviewed_at"]
VIDEO_VERDICTS = ["approved", "weak", "rejected"]  # review.csv; weak = usable but weak, it counts as done
NEEDS_REASON = ("weak", "rejected")  # these verdicts count only with a reason (a reason id, the tick or a line)
# Reasons for weak or rejected: (id in review.csv, text on the review page). "Reference image problem" is the column
# ref_problem (a rejected video is then made again with a new reference image), and "other" is the line of text.
REASONS = [
    ("towel_look", "Towel color, texture or pattern changed"),
    ("towel_shape", "Towel shape differs from the simulator"),
    ("towel_doubled", "Towel looks doubled, or more than one towel"),
    ("towel_wrinkle", "Towel wrinkle problem"),
    ("extra_object", "Extra object visible (hand, cable, gripper, text or logo, ...)"),
    ("robot", "Robot problem (ghosting, doubling, melting, pose or motion differs from the simulator, ...)"),
    ("background", "Background problem"),
    ("wrist_background", "Wrist view background problem"),
    ("lighting", "Lighting problem"),
    ("table", "Table problem"),
    ("image_quality", "Image quality problem (blur, smeared, overexposed)"),
    ("views_differ", "Room and wrist views do not match"),
]
REASON_IDS = [r[0] for r in REASONS]
STATES = ["conflict", "needs_replace", "needs_ref", "needs_check", "ref_failed", "needs_video", "needs_review",
          "video_rejected", "approved"]

REF_RE = re.compile(r"demo_(\d{3,})(?:_([^/]+))?\.png")
VIDEO_RE = re.compile(r"demo_(\d{3,})_([^/]+)\.mp4")
GROUP_RE = re.compile(r"\d{3,}-\d{3,}")  # a group folder: 000-049, 050-099, ...


# ---------------------------------------------------------------------------------------------------------------
# helpers (imported by the other scripts)

_held = {}  # lock path -> [threading.RLock, file descriptor, depth]
_held_guard = threading.Lock()


@contextlib.contextmanager
def lock(folder: str, name: str = LISTS_LOCK):
    """Exclusive lock on <folder>/<name>. Other processes and other threads wait; the same thread may enter again."""
    if not os.path.isdir(folder):
        sys.exit(f"no such folder: {folder}")
    path = os.path.abspath(os.path.join(folder, name))
    with _held_guard:
        entry = _held.setdefault(path, [threading.RLock(), None, 0])
    entry[0].acquire()
    try:
        if entry[2] == 0:
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
            t0, pause, said = time.time(), 0.01, False
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if not said and time.time() - t0 > 2:  # say it only for a real wait, not for a short write
                        print(f"waiting: another script holds {path}", flush=True)
                        said = True
                    time.sleep(pause)
                    pause = min(pause * 2, 1.0)
            entry[1] = fd
        entry[2] += 1
        try:
            yield
        finally:
            entry[2] -= 1
            if entry[2] == 0:
                os.close(entry[1])  # closing the file releases the lock
                entry[1] = None
    finally:
        entry[0].release()


def lock_fd(folder: str, name: str) -> int | None:
    """File descriptor of a lock this process holds (to hand it to a child process), or None."""
    entry = _held.get(os.path.abspath(os.path.join(folder, name)))
    return entry[1] if entry else None


def read_csv(path: str) -> list:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _header(path: str) -> list:
    with open(path, newline="") as f:
        return next(csv.reader(f), [])


def _tmp(path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(path)), f".{os.path.basename(path)}.tmp{os.getpid()}")


def write_csv(path: str, rows: list, columns: list) -> None:
    """Write the whole file: a temporary file first, then one rename, so a reader never sees half a file."""
    extra = [k for r in rows for k in r if k not in columns]
    fields = columns + list(dict.fromkeys(extra))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = _tmp(path)
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def append_csv(path: str, row: dict, columns: list) -> None:
    """Add one row. A file with an older header gets the new columns first. Call it while holding lock()."""
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        header = _header(path)
        if all(c in header for c in columns) and all(k in header for k in row):
            with open(path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=header, restval="").writerow(row)
            return
        write_csv(path, read_csv(path) + [row], columns + [c for c in header if c not in columns])
        return
    write_csv(path, [row], columns)


def write_json(path: str, obj) -> None:
    tmp = _tmp(path)
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def parse_demos(values) -> list:
    """["0-49", "60", "070-099"] -> sorted demo indices. None stays None."""
    if values is None:
        return None
    out = set()
    for v in values:
        for part in str(v).replace(",", " ").split():
            a, sep, b = part.partition("-")
            try:
                lo, hi = int(a), int(b) if sep else int(a)
            except ValueError:
                sys.exit(f"--demos {part}: use numbers and ranges, for example 0-49 60 70-99")
            if hi < lo:
                sys.exit(f"--demos {part}: the range goes down")
            out.update(range(lo, hi + 1))
    return sorted(out)


def ranges(idxs) -> str:
    """[0, 1, 2, 5, 7, 8] -> "0-2 5 7-8" (the form --demos takes)."""
    idxs = sorted(set(idxs))
    parts, i = [], 0
    while i < len(idxs):
        j = i
        while j + 1 < len(idxs) and idxs[j + 1] == idxs[j] + 1:
            j += 1
        parts.append(str(idxs[i]) if i == j else f"{idxs[i]}-{idxs[j]}")
        i = j + 1
    return " ".join(parts)


def source_key(run: str, demo) -> str:
    return f"{run}:{int(demo)}"


def current_sources(folder: str) -> dict:
    """demo number -> "<source run>:<source demo>" from sources.csv. Empty for a run folder (not a dataset)."""
    return {int(r["demo"]): source_key(r["source_run"], r["source_demo"]) for r in read_csv(f"{folder}/sources.csv")}


def is_dataset(folder: str) -> bool:
    return os.path.isfile(f"{folder}/sources.csv")


def dataset_config(ds: str) -> dict:
    """run_config.json of a dataset, with the defaults for variation_seed, tag and max_attempts."""
    path = f"{ds}/run_config.json"
    cfg = json.load(open(path)) if os.path.isfile(path) else {}
    cfg.setdefault("variation_seed", VARIATION_SEED)
    cfg.setdefault("tag", TAG)
    cfg.setdefault("max_attempts", MAX_ATTEMPTS)
    return cfg


def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_is_current(refs: str, row: dict) -> bool:
    """True when a row of check_references.csv belongs to the image that is in refs/ now (same sha1)."""
    path = f"{layout.ref_dir(refs, int(row['demo']))}/{row['name']}.png"
    return bool(row.get("sha1")) and os.path.isfile(path) and file_sha1(path) == row["sha1"]


@contextlib.contextmanager
def open_hdf5(path: str, mode: str = "r"):
    """h5py.File that waits (up to H5_WAIT s) while another process has the file open for writing."""
    import h5py

    t0, said = time.time(), False
    while True:
        try:
            f = h5py.File(path, mode)
            break
        except OSError as e:
            if "lock" not in str(e).lower():
                raise
            if time.time() - t0 > H5_WAIT:
                sys.exit(f"{path} is open in another process for more than {H5_WAIT} s: run again later")
            if not said:
                print(f"waiting: another process has {os.path.basename(path)} open", flush=True)
                said = True
            time.sleep(5)
    try:
        yield f
    finally:
        f.close()


@contextlib.contextmanager
def stop_signals():
    """Turn SIGTERM and SIGHUP (a closed terminal) into KeyboardInterrupt, so cleanup code runs."""
    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    old = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def lock_holder(folder: str, name: str) -> str | None:
    """None when no process holds <folder>/<name> right now, else the text in the file (a look only)."""
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return None
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return None
    except BlockingIOError:
        with open(path) as f:
            return f.read().strip()
    finally:
        os.close(fd)


def running_cosmos(ds: str) -> dict:
    """Cosmos run folder name -> its run_config.json, for every run_cosmos.py that runs now (.running.lock held)."""
    out = {}
    for cfg_path in sorted(glob.glob(f"{ds}/cosmos/*/run_config.json")):
        folder = os.path.dirname(cfg_path)
        if lock_holder(folder, RUNNING_LOCK) is None:
            continue
        try:
            out[os.path.basename(folder)] = json.load(open(cfg_path))
        except (OSError, ValueError):
            continue
    return out


def cosmos_claims(ds: str) -> dict:
    """demo number -> Cosmos run folder, for the demos of every run_cosmos.py that runs now."""
    return {int(d): name for name, cfg in running_cosmos(ds).items() for d in cfg.get("demos", [])}


def pending_replacements(ds: str) -> set:
    """Demo numbers with a --replace that is running or stopped halfway. Other scripts leave them alone."""
    return {int(j["demo"]) for _, j in open_replacements(ds) if "demo" in j}


def open_replacements(ds: str) -> list:
    """(path, plan) of every --replace that started and did not finish (sim_rejected/.../replaced.json with state
    started or moved). Unreadable plans are left out here; broken_replacements() lists them."""
    out = []
    for path in sorted(glob.glob(f"{ds}/sim_rejected/*/*/replaced.json")):
        try:
            j = json.load(open(path))
        except (OSError, ValueError):
            continue
        if isinstance(j, dict) and j.get("state") in ("started", "moved") and isinstance(j.get("demo"), int):
            out.append((path, j))
    return out


def broken_replacements(ds: str) -> list:
    """Paths of replaced.json files that cannot be read or have no demo number or state."""
    out = []
    for path in sorted(glob.glob(f"{ds}/sim_rejected/*/*/replaced.json")):
        try:
            j = json.load(open(path))
            ok = isinstance(j, dict) and isinstance(j.get("demo"), int) and "state" in j
        except (OSError, ValueError):
            ok = False
        if not ok:
            out.append(path)
    return out


# ---------------------------------------------------------------------------------------------------------------
# state of a dataset

def _rel(ds: str, path: str) -> str:
    return os.path.relpath(path, ds)


def video_id(path: str) -> str:
    """"<cosmos run>/<name>" of a video, the key of review.csv. It stays the same when the video moves from cosmos/
    to cosmos_rejected/ (both keep <run>/<group>/<name>.mp4)."""
    parts = os.path.normpath(path).split(os.sep)
    return f"{parts[-3]}/{parts[-1][: -len('.mp4')]}"


def reason_ids(text: str) -> list:
    """The known reason ids in the reasons column of review.csv ("a;b"), in the order of REASONS."""
    ids = str(text or "").split(";")
    return [x for x in REASON_IDS if x in ids]


def has_reason(rev: dict) -> bool:
    """True when a review.csv row gives a reason: a reason id, the reference image tick, or a line of text."""
    return bool(reason_ids(rev.get("reasons")) or rev.get("ref_problem") == "1"
                or str(rev.get("review") or "").strip())


def row_spec(row: dict) -> str:
    """The spec version of a row of refs/references.csv."""
    return row.get("spec") or UNRECORDED_SPEC


def latest_refs(rows: list) -> dict:
    """(name, source) -> the row of refs/references.csv with the highest attempt: the image that is in refs/ now
    (source is "" in a run folder)."""
    out = {}
    for r in rows:
        k = (r["name"], r.get("source") or "")
        if k not in out or int(r.get("attempt") or 0) >= int(out[k].get("attempt") or 0):
            out[k] = r
    return out


def _video_meta(mp4: str) -> dict:
    """The key policy_data of the video json (source demo, spec, ...), {} when the json does not have it."""
    try:
        with open(mp4[: -len(".mp4")] + ".json") as f:
            meta = json.load(f).get("policy_data", {})
        return meta if isinstance(meta, dict) else {}
    except (OSError, ValueError, AttributeError):
        return {}


def _run_spec(ds: str, run: str, cache: dict) -> str:
    """The spec of a Cosmos run folder (key spec of its run_config.json), for videos whose json has none."""
    if run not in cache:
        try:
            cache[run] = str(json.load(open(f"{ds}/{run}/run_config.json")).get("spec") or "")
        except (OSError, ValueError, AttributeError):
            cache[run] = ""
    return cache[run]


def _scan_refs(ds: str, n: int, problems: list) -> tuple[dict, set]:
    """(demo -> [reference images in its group folder], demos with an image in a wrong folder)."""
    refs, stray = collections.defaultdict(list), set()
    for path in sorted(glob.glob(f"{ds}/refs/**/*.png", recursive=True)):
        m = REF_RE.fullmatch(os.path.basename(path))
        if not m:
            continue  # check_<name>.png drawings of check_references.py
        idx = int(m.group(1))
        if idx >= n:
            problems.append(f"{_rel(ds, path)}: no demo {idx} in the dataset")
        elif os.path.dirname(path) != layout.ref_dir(f"{ds}/refs", idx):
            problems.append(f"{_rel(ds, path)}: wrong folder, move it to {_rel(ds, layout.ref_dir(ds + '/refs', idx))}")
            stray.add(idx)
        else:
            refs[idx].append(path)
    return refs, stray


def _scan_videos(ds: str, n: int, problems: list) -> tuple[dict, list, set]:
    """(demo -> [current videos], [rejected videos], demos with a video in a wrong group folder). Only group
    folders count: the other folders of a Cosmos run (refs/, specs/) hold temporary files."""
    current, stray = collections.defaultdict(list), set()
    for path in sorted(glob.glob(f"{ds}/cosmos/*/*/*.mp4")):
        m = VIDEO_RE.fullmatch(os.path.basename(path))
        group = os.path.basename(os.path.dirname(path))
        if not m or not GROUP_RE.fullmatch(group):
            continue
        idx = int(m.group(1))
        if idx >= n:
            problems.append(f"{_rel(ds, path)}: no demo {idx} in the dataset")
        elif group != layout.chunk(idx):
            problems.append(f"{_rel(ds, path)}: wrong group folder (demo {idx} belongs in {layout.chunk(idx)})")
            stray.add(idx)
        else:
            current[idx].append(path)
    rejected = sorted(p for p in glob.glob(f"{ds}/cosmos_rejected/**/*.mp4", recursive=True)
                      if VIDEO_RE.fullmatch(os.path.basename(p)))
    return current, rejected, stray


def _check_hdf5(ds: str, cfg: dict, sources: dict, problems: list) -> None:
    """The dataset hdf5 must hold data/demo_0 ... data/demo_<n-1>, each from the source demo in sources.csv."""
    path = f"{ds}/{cfg.get('hdf5', '')}"
    if not os.path.isfile(path):
        problems.append(f"hdf5 not found: {cfg.get('hdf5')}")
        return
    try:
        import h5py

        with h5py.File(path, "r") as f:
            data = f["data"]
            keys = set(data.keys())
            found = {int(k.split("_")[1]): source_key(data[k].attrs.get("source_run", ""),
                                                      data[k].attrs.get("source_demo", -1))
                     for k in keys if re.fullmatch(r"demo_\d+", k)}
    except ImportError:
        return
    except OSError as e:
        if "lock" in str(e).lower():
            return  # another script writes the file right now; the next run checks it
        problems.append(f"hdf5 not readable: {e}")
        return
    want = {f"demo_{i}" for i in range(len(sources))}
    if keys != want:
        problems.append(f"hdf5: missing {sorted(want - keys)[:5]}, extra {sorted(keys - want)[:5]}")
    wrong = [i for i, s in sources.items() if i in found and found[i] != s]
    if wrong:
        problems.append(f"hdf5 data does not match sources.csv for demos {ranges(wrong)}")


def collect(ds: str) -> tuple[list, list, list]:
    """(rows of dataset.csv, rows of videos.csv, problems). Call it while holding lock(ds)."""
    cfg = dataset_config(ds)
    seed, tag, max_attempts = int(cfg["variation_seed"]), cfg["tag"], int(cfg["max_attempts"])
    source_rows = read_csv(f"{ds}/sources.csv")
    n = len(source_rows)
    problems = []
    if [int(r["demo"]) for r in source_rows] != list(range(n)):
        problems.append("sources.csv: the demo numbers are not 0, 1, 2, ... in order")
    if cfg.get("n_demos") not in (None, n):
        problems.append(f"sources.csv has {n} demos, run_config.json says {cfg['n_demos']}")
    current = current_sources(ds)
    _check_hdf5(ds, cfg, current, problems)
    running = lock_holder(ds, REPLACE_LOCK)  # the demo number of the --replace that runs now, or None
    pending = {}  # demo -> note
    for path, j in open_replacements(ds):
        if running == str(j["demo"]):
            pending[j["demo"]] = "replace running"
        else:
            pending[j["demo"]] = "replace stopped halfway"
            problems.append(f"demo {j['demo']}: --replace stopped halfway ({_rel(ds, path)}): run make_dataset.py "
                            f"<dataset> --replace {j['demo']} to finish it, or add --cancel to drop it")
    for path in broken_replacements(ds):
        problems.append(f"{_rel(ds, path)}: unreadable or incomplete, fix by hand")
    claims = cosmos_claims(ds)

    ref_rows = read_csv(f"{ds}/refs/references.csv")
    tries = collections.Counter((r["name"], r.get("source") or "") for r in ref_rows)
    latest = latest_refs(ref_rows)
    check_rows = {r["name"]: r for r in read_csv(f"{ds}/refs/check_references.csv")}
    ref_rejected = collections.Counter(r["name"] for r in read_csv(f"{ds}/refs_rejected/rejected.csv"))
    reviews = {r["video"]: r for r in read_csv(f"{ds}/review.csv")}
    sim_reviews = {(int(r["demo"]), r["source"]): r for r in read_csv(f"{ds}/sim_review.csv")}
    replaced = collections.Counter(int(r["demo"]) for r in read_csv(f"{ds}/replacements.csv"))

    refs, stray_refs = _scan_refs(ds, n, problems)
    videos, rejected_videos, stray_videos = _scan_videos(ds, n, problems)
    rejected_by_demo = collections.defaultdict(list)
    video_rows, run_specs = [], {}

    def video_spec(path: str, meta: dict) -> str:
        return meta.get("spec") or _run_spec(ds, os.path.dirname(os.path.dirname(_rel(ds, path))), run_specs)

    for path in rejected_videos:
        m = VIDEO_RE.fullmatch(os.path.basename(path))
        meta = _video_meta(path)
        idx, src = int(m.group(1)), meta.get("source")
        rejected_by_demo[idx].append(src)
        rev = reviews.get(video_id(path), {})
        video_rows.append({"demo": idx, "name": os.path.basename(path)[: -len(".mp4")],
                           "cosmos_run": _rel(ds, path).split(os.sep)[1], "file": _rel(ds, path), "where": "rejected",
                           "source": src or "", "spec": video_spec(path, meta), "verdict": rev.get("verdict", ""),
                           "ref_problem": rev.get("ref_problem", ""),
                           "reasons": ";".join(reason_ids(rev.get("reasons"))), "review_text": rev.get("review", "")})

    rows = []
    for idx in range(n):
        src = current.get(idx, "")
        name = f"{layout.demo_name(idx)}_{tag}"
        row = {c: "" for c in DATASET_COLUMNS}
        row.update(demo=idx, group=layout.chunk(idx), source=src, replaced=replaced.get(idx, 0),
                   refs_rejected=ref_rejected.get(name, 0), place_type=variations.place_type(idx, seed, tag),
                   strong_light=int(variations.strong_light(idx, seed, tag)))
        if idx in pending:  # a replace is running or stopped halfway: nothing else for this number now
            row.update(state="needs_replace", note=pending[idx])
            rows.append(row)
            continue
        if idx in claims:
            row["note"] = f"cosmos running ({claims[idx]})"
        missing = False
        for kind, path in (("geoedge video", f"{layout.demo_dir(ds, idx)}/{layout.demo_name(idx)}_geoedge.mp4"),
                           ("frame-0 image", f"{layout.ref_sim_dir(ds, idx)}/{layout.demo_name(idx)}_ref_sim.png")):
            if not os.path.isfile(path):
                problems.append(f"{layout.demo_name(idx)}: {kind} missing ({_rel(ds, path)})")
                missing = True
        sim = sim_reviews.get((idx, src))
        row["sim_review"] = sim["verdict"] if sim else ""
        n_tries = tries.get((name, src), 0)
        row["ref_attempts"] = n_tries
        vids = videos.get(idx, [])
        # A rejected video without a source in its json counts only while the number was never replaced.
        n_redo = sum(1 for s in rejected_by_demo.get(idx, []) if s == src or (s is None and not replaced.get(idx)))
        row["videos_rejected"] = n_redo
        for path in vids:
            rev, meta = reviews.get(video_id(path), {}), _video_meta(path)
            video_rows.append({"demo": idx, "name": os.path.basename(path)[: -len(".mp4")],
                               "cosmos_run": _rel(ds, path).split(os.sep)[1], "file": _rel(ds, path),
                               "where": "current", "source": meta.get("source") or "", "spec": video_spec(path, meta),
                               "verdict": rev.get("verdict", ""), "ref_problem": rev.get("ref_problem", ""),
                               "reasons": ";".join(reason_ids(rev.get("reasons"))),
                               "review_text": rev.get("review", "")})

        images = refs.get(idx, [])
        row["reference"] = " ".join(os.path.basename(p)[: -len(".png")] for p in images)
        made = latest.get((row["reference"], src)) if len(images) == 1 else None
        row["spec"] = row_spec(made) if made else ""  # "" for an image that make_references.py did not make
        row["video"] = " ".join(_rel(ds, p) for p in vids)
        if len(images) > 1 or len(vids) > 1 or idx in stray_refs or idx in stray_videos or missing:
            if len(images) > 1 or len(vids) > 1:
                what = [f"{len(images)} reference images"] if len(images) > 1 else []
                what += [f"{len(vids)} videos"] if len(vids) > 1 else []
                problems.append(f"{layout.demo_name(idx)}: {' and '.join(what)}, there must be one")
            row["state"] = "conflict"
            rows.append(row)
            continue
        if images:
            chk = check_rows.get(row["reference"])
            if chk and check_is_current(f"{ds}/refs", chk):
                row["ref_check"] = "passed" if chk.get("passed") == "True" else "failed"
                row["room_score"], row["wrist_score"] = chk.get("room_score", ""), chk.get("wrist_score", "")
            else:
                row["ref_check"] = "unchecked"
        reason = False
        if vids:
            rev = reviews.get(video_id(vids[0]), {})
            row["review"] = rev.get("verdict", "")
            row["ref_problem"], row["review_text"] = rev.get("ref_problem", ""), rev.get("review", "")
            row["reasons"], reason = ";".join(reason_ids(rev.get("reasons"))), has_reason(rev)

        if row["sim_review"] == "rejected":
            state = "needs_replace"
        elif not images:
            state = "needs_replace" if n_tries >= max_attempts else "needs_ref"
        elif row["ref_check"] == "unchecked":
            state = "needs_check"
        elif row["ref_check"] == "failed":
            state = "needs_replace" if n_tries >= max_attempts else "ref_failed"
        elif not vids:
            state = "needs_video"
        elif row["review"] == "approved" or (row["review"] == "weak" and reason):
            state = "approved"  # weak counts as done; the column review still says weak
        elif row["review"] == "rejected" and reason:
            state = "needs_replace" if n_redo + 1 >= max_attempts else "video_rejected"
        else:
            state = "needs_review"
            if row["review"] in NEEDS_REASON:  # saved without a reason: not done, not made again
                row["note"] = "; ".join(x for x in (row["note"], "no reason") if x)
        row["state"] = state
        rows.append(row)
    return rows, video_rows, problems


def _why_replace(row: dict, max_attempts: int) -> str:
    if row["sim_review"] == "rejected":
        return "the simulator demo looks wrong"
    if row["review"] == "rejected":
        return f"the video was rejected {max_attempts} times"
    return f"the reference image failed {max_attempts} times"


def _n(count: int, noun: str) -> str:
    """'1 video', '2 videos'."""
    return f"{count} {noun}{'' if count == 1 else 's'}"


GPU_PAIRS = ["0,1", "2,3"]  # the example GPU pairs of the run_cosmos.py hint (a 4-GPU machine)


def next_commands(ds_arg: str, rows: list, max_attempts: int, running: dict | None = None) -> list:
    """The commands to run next, in the order the states are handled. running: the Cosmos jobs that run now."""
    s = os.path.relpath(HERE)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["state"]].append(r)
    cmds = []
    for r in [r for r in by["needs_replace"] if r["note"] == "replace stopped halfway"]:
        cmds.append(f"python {s}/make_dataset.py {ds_arg} --replace {r['demo']}    # finishes the stopped replace")
    new = [r for r in by["needs_replace"] if not r["note"]]
    for r in new[:10]:
        cmds.append(f'python {s}/make_dataset.py {ds_arg} --replace {r["demo"]} '
                    f'--reason "{_why_replace(r, max_attempts)}"')
    if len(new) > 10:
        cmds.append(f"... and {len(new) - 10} more to replace (see dataset.csv)")
    if by["ref_failed"]:
        cmds.append(f"python {s}/make_references.py {ds_arg} --retry --demos "
                    f"{ranges(r['demo'] for r in by['ref_failed'])}")
    if by["needs_check"]:
        cmds.append(f"python {s}/checks/check_references.py {ds_arg} --demos "
                    f"{ranges(r['demo'] for r in by['needs_check'])}")
    if by["needs_ref"]:
        group = by["needs_ref"][0]["group"]
        cmds.append(f"python {s}/make_references.py {ds_arg} --demos "
                    f"{ranges(r['demo'] for r in by['needs_ref'] if r['group'] == group)}")
    ready = [r for r in by["needs_video"] if not r["note"]]
    busy = {c.get("gpus") for c in (running or {}).values()}
    free = [g for g in GPU_PAIRS if g not in busy]
    if ready and not free:
        cmds.append(f"{len(ready)} demos are ready for Cosmos; both GPU pairs are busy: wait for a running job")
    elif ready:
        group = ready[0]["group"]
        demos = [r["demo"] for r in ready if r["group"] == group]
        pairs = free[:2] if len(demos) > 1 else free[:1]
        extra = f" --max {(len(demos) + 1) // 2}" if len(pairs) == 2 else ""
        for gpus in pairs:
            cmds.append(f"python {s}/run_cosmos.py {ds_arg} --demos {ranges(demos)}{extra} --gpus {gpus} "
                        "--cp 2 --framework <dir> --checkpoint <dir> --hf_home <dir>")
        if len(pairs) == 2:
            cmds.append("  (the two run_cosmos.py commands can run at the same time: each takes other demos)")
    if by["needs_review"]:
        no_reason = sum(1 for r in by["needs_review"] if r["review"] in NEEDS_REASON)
        cmds.append(f"python {s}/review.py {ds_arg} --group {by['needs_review'][0]['group']}    "
                    f"# {_n(len(by['needs_review']), 'video')} to review"
                    + (f" ({no_reason} marked weak or rejected, without a reason)" if no_reason else ""))
    if by["video_rejected"]:
        cmds.append(f"{_n(len(by['video_rejected']), 'video')} rejected: run_cosmos.py --redo_bad (makes them "
                    "again) comes in the next version")
    if rows and len(by["approved"]) == len(rows):
        weak = sum(1 for r in rows if r["review"] == "weak")
        cmds.append(f"done: all {len(rows)} demos approved" + (f" ({weak} of them weak)" if weak else ""))
    return cmds


def update(ds: str, ds_arg: str | None = None, quiet: bool = False) -> dict:
    """Write dataset.csv and videos.csv, print the summary and the next commands. Returns the count per state."""
    ds = os.path.abspath(ds)
    if not is_dataset(ds):
        sys.exit(f"{ds} is not a dataset folder (no sources.csv): make one with make_dataset.py --take")
    with lock(ds):
        rows, video_rows, problems = collect(ds)
        write_csv(f"{ds}/dataset.csv", rows, DATASET_COLUMNS)
        write_csv(f"{ds}/videos.csv", video_rows, VIDEO_COLUMNS)
    counts = collections.Counter(r["state"] for r in rows)
    if quiet:
        return counts
    ds_arg = ds_arg or os.path.relpath(ds)
    weak = sum(1 for r in rows if r["state"] == "approved" and r["review"] == "weak")
    print(f"\n== status of {ds_arg}: {len(rows)} demos, {counts.get('approved', 0)} approved"
          + (f" ({weak} weak)" if weak else "") + f" ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    print("  " + ", ".join(f"{s} {counts[s]}" for s in STATES if counts.get(s)))
    no_reason = sum(1 for r in rows if r["state"] == "needs_review" and r["review"] in NEEDS_REASON)
    if no_reason:
        print(f"  no reason: {_n(no_reason, 'video')} marked weak or rejected without a reason. "
              f"{'It counts' if no_reason == 1 else 'They count'} as needs_review. Add a reason on the review page.")
    groups = collections.defaultdict(collections.Counter)
    for r in rows:
        groups[r["group"]][r["state"] == "approved"] += 1
    print("  approved per group: " + ", ".join(f"{g} {groups[g][True]}/{sum(groups[g].values())}"
                                               for g in sorted(groups)))
    types = collections.defaultdict(collections.Counter)
    for r in rows:
        types[r["place_type"]][r["state"] == "approved"] += 1
        if str(r["strong_light"]) == "1":
            types["strong light"][r["state"] == "approved"] += 1
    print("  approved per place type: " + ", ".join(f"{t} {types[t][True]}/{sum(types[t].values())}"
                                                   for t in [*variations.PLACE_TYPES, "strong light"] if t in types))
    specs = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["spec"]:
            specs[r["spec"]][r["state"] == "approved"] += 1
    if specs:
        print("  approved per spec (of the demos with a reference image): "
              + ", ".join(f"{v} {specs[v][True]}/{sum(specs[v].values())}" for v in sorted(specs)))
    sims = collections.Counter(r["sim_review"] or "unreviewed" for r in rows)
    print("  simulator review: " + ", ".join(f"{k} {v}" for k, v in sorted(sims.items())))
    running = running_cosmos(ds)
    for name, c in running.items():
        print(f"  cosmos running: {name} on GPUs {c.get('gpus', '?')}, demos {ranges(c.get('demos', []))}")
    if problems:
        print(f"  problems ({len(problems)}), fix by hand:")
        for p in problems[:20]:
            print(f"    {p}")
        if len(problems) > 20:
            print(f"    ... and {len(problems) - 20} more")
    cmds = next_commands(ds_arg, rows, int(dataset_config(ds)["max_attempts"]), running)
    if cmds:
        print("  next:")
        for c in cmds:
            print(f"    {c}")
    print(f"  lists: {ds_arg}/dataset.csv, {ds_arg}/videos.csv", flush=True)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", help="dataset folder made by make_dataset.py --take")
    args = ap.parse_args()
    update(args.dataset_dir, ds_arg=args.dataset_dir)


if __name__ == "__main__":
    main()
