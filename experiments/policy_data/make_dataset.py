"""Make one dataset folder from the demos of several runs, replace a demo in it, and add demos to it.

  python experiments/policy_data/make_dataset.py --take <run>:<N> [<run>:<N> ...] [--out <folder>] [--name <name>]
      [--skip_raw <kind> ...]
  python experiments/policy_data/make_dataset.py <dataset_dir> --replace <demo> --reason "<text>" [--from <run>]
      [--runs_dir <folder>]
  python experiments/policy_data/make_dataset.py <dataset_dir> --add <run>:<N> [<run>:<N> ...] [--runs_dir <folder>]

--take takes the first N demos of each run whose towel stays inside the room image in every frame, in the order
given, and numbers them 0, 1, 2, ... Each run needs <run>/check_towel_in_view.csv first:
    python experiments/policy_data/checks/check_towel_in_view.py <run>/<name>.hdf5 --out <run>/check_towel_in_view.csv
It writes a new folder <out>/<task>_n<total>_seeds<seeds>_<YYYYMMDD>_<HHMM>/ (default <out>: the folder of the
first run; <seeds> are the seeds of the runs written one after the other, for example seeds123). --name <name> sets
the name instead: the folder is <out>/<name>/ and the hdf5 is <name>.hdf5. Use it for a dataset that grows with
--add, so the name does not state a demo count. The folder has:
  <task>_n<total>_seeds<seeds>.hdf5   the demos, copied. The runs keep theirs: --replace takes spare demos there.
  demos/, ref_sim/                    videos and frame-0 images, as hard links with the new numbers (no extra space;
                                      a copy when a hard link is not possible, for example on another disk)
  sources.csv                         demo number -> source run and demo, seed, room camera noise, steps
  run_config.json                     the takes, every run (relative path and its settings), and the fixed values
                                      for the reference images: variation_seed 0, tag 01, max_attempts 5
All runs must have the same task, cameras, image size, video frames and simulation settings, and different seeds
(the same seed repeats the start poses). The runs can number the instance ids differently, so every demo keeps the
table of its run in the attribute instance_ids of data/demo_N (json: camera -> id -> prim path), next to source_run
and source_demo. There is no <name>_instance_ids.json in a dataset. The hdf5 copy needs about 350 MB per demo; the
script stops before it starts when the disk has less free space. The folder is built under a temporary name
(.<name>.partial) and renamed at the end, so a stopped run leaves no half dataset.
--skip_raw <kind> ... leaves the raw renderer data of these kinds (depth, normals, instance) out of the dataset
hdf5: the observations <camera>_<kind>_raw are not copied. The runs keep them. Depth and normals are about three
quarters of a demo (about 250 of 350 MB), and policy training does not read them. run_config.json records the
choice as "skip_raw", and --add and --replace leave out the same kinds. What reads the raw data of a dataset:
run_cosmos.py reads depth for the depth control and instance for the color guide, and checks/check_references.py
reads instance. Nothing reads normals.

--replace puts the next unused demo of the same run (or of --from <run>) at number <demo>: for a demo that looks
wrong in the simulator, or whose reference image or video failed too often. The number keeps its place type and
strong light slot, so the ratios stay the same. Everything of the old demo (videos, frame-0 image, reference images,
Cosmos videos) moves to sim_rejected/<group>/demo_NNN_r<k>/, and replacements.csv gets one row. Then make the
reference image and the video of the number again (status.py prints the commands).
All checks (settings, seed, frames, env_args, free space) run first. Then the plan is written to
sim_rejected/<group>/demo_NNN_r<k>/replaced.json, before anything moves. When the replace stops halfway (an error,
Ctrl-C, a closed terminal), status.py says so: run make_dataset.py <dataset> --replace <demo> again (no --reason
needed) and it finishes the plan, or add --cancel to drop it while the new demo is not linked yet (the old files
move back). Until then the other scripts leave the number alone. Run --replace again only when status.py says it
stopped halfway: otherwise it replaces the number once more. Only one --replace runs on a dataset at a time
(.replace.lock); a second --replace of the same number that waited for it stops.
A --from run must have the same settings, the same room camera noise as the run of the old demo, and a seed that no
other run of the dataset has. The run is then added to run_config.json.
It waits while make_references.py runs on the dataset, or while another process has the hdf5 open. It refuses a
number that a running run_cosmos.py makes a video for (replace it after that job); other numbers go on. The old hdf5
data stays in its source run. The hdf5 file does not shrink when a demo is replaced (h5repack makes it small again).
--runs_dir <folder>: where the source runs are, when the dataset or the runs were moved (default: the relative paths
in run_config.json).

--add puts the next N unused demos of each run after the last demo of the dataset, with new numbers (a dataset with
demos 0-749 gets 750, 751, ...). <run> is a run folder, or the name of a run of the dataset. A new run needs the same
settings as the dataset and a seed that no other run of the dataset has; its room camera noise can differ (the
script prints it). The folder and the hdf5 keep their names; n_demos in run_config.json and sources.csv grow, and
run_config.json lists each --add under "added". An added demo is no spare any more: --replace finds no unused demo
in a run when --add took them all. The place type shares are exact in every full group of 50 numbers, so add up to
the end of a group when the shares matter. Like --replace, --add waits for make_references.py and for the hdf5, and
only one of them runs on a dataset at a time. When it stops before it prints "added demos", run the same command
again: it goes on with the demos that are not copied yet. Then make the reference images and the videos of the new
numbers as for the others.
"""

import argparse
import errno
import filecmp
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import cv2
import h5py
import numpy as np

import layout
import status

HERE = os.path.dirname(os.path.abspath(__file__))
SAME = ["task", "image_size", "wrist_focal_mm", "room_camera", "wrist_camera", "raw"]  # must match across runs
NOISE = ["camera_noise_pos_m", "camera_noise_rot_deg"]
NEEDED = ["_source.mp4", "_geoedge.mp4"]  # per-demo videos a dataset demo must have
RAW_KINDS = ("depth", "normals", "instance")  # --skip_raw: the observations <camera>_<kind>_raw
GB = 1e9


class Run:
    """A finished run folder of make_demos.sh with its towel-in-view check."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.name = os.path.basename(os.path.normpath(self.path))
        if not os.path.isfile(f"{self.path}/run_config.json"):
            sys.exit(f"{path}: no run_config.json (not a run folder of make_demos.sh)")
        self.cfg = json.load(open(f"{self.path}/run_config.json"))
        if not os.path.isfile(f"{self.path}/summary.txt"):
            sys.exit(f"{self.name}: no summary.txt, the run is not finished")
        h5s = [f for f in sorted(os.listdir(self.path)) if f.endswith(".hdf5") and "_failed" not in f]
        if len(h5s) != 1:
            sys.exit(f"{self.name}: need one demo hdf5 (without _failed), found {h5s}")
        self.hdf5 = f"{self.path}/{h5s[0]}"
        table = self.hdf5[: -len(".hdf5")] + "_instance_ids.json"
        if not os.path.isfile(table):
            sys.exit(f"{self.name}: {os.path.basename(table)} not found")
        self.tables = json.load(open(table))
        check = f"{self.path}/check_towel_in_view.csv"
        if not os.path.isfile(check):
            sys.exit(f"{self.name}: check_towel_in_view.csv not found. Run first:\n  python "
                     f"{os.path.relpath(HERE)}/checks/check_towel_in_view.py {os.path.relpath(self.hdf5)} "
                     f"--out {os.path.relpath(check)}")
        self.check = status.read_csv(check)
        with h5py.File(self.hdf5, "r") as f:
            self.keys = set(f["data"].keys())
            self.env_args = json.loads(f["data"].attrs["env_args"])
        if len(self.check) != len(self.keys):
            sys.exit(f"{self.name}: check_towel_in_view.csv has {len(self.check)} demos, the hdf5 has "
                     f"{len(self.keys)}: run the check again")
        self._frames = None

    def in_view(self) -> list:
        """Source demo indices whose towel stays in view, in order."""
        return [layout.demo_index(r["demo"]) for r in self.check if str(r["cut"]) == "0"]

    def files(self, s: int) -> list:
        """(source file, name after the demo number) of every video and the frame-0 image of source demo s."""
        old, d = layout.demo_name(s), layout.demo_dir(self.path, s)
        out = [(f"{d}/{f}", f[len(old):]) for f in sorted(os.listdir(d)) if f.startswith(old + "_")] \
            if os.path.isdir(d) else []
        return out + [(f"{layout.ref_sim_dir(self.path, s)}/{old}_ref_sim.png", "_ref_sim.png")]

    def usable(self, s: int) -> bool:
        have = {suffix for path, suffix in self.files(s) if os.path.isfile(path)}
        return f"demo_{s}" in self.keys and all(x in have for x in NEEDED + ["_ref_sim.png"])

    def frames(self) -> int:
        """Frames of the geoedge video of the first usable demo (81 by default, more with --all_frames)."""
        if self._frames is None:
            s = next((s for s in self.in_view() if self.usable(s)), None)
            if s is None:
                sys.exit(f"{self.name}: no usable demo")
            cap = cv2.VideoCapture(f"{layout.demo_dir(self.path, s)}/{layout.demo_name(s)}_geoedge.mp4")
            self._frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        return self._frames

    def config(self) -> dict:
        """run_config.json without host paths (input and output keep only the file name)."""
        return {k: os.path.basename(v) if k in ("input", "output") and isinstance(v, str) else v
                for k, v in self.cfg.items()}


def check_same(a: Run, b: Run) -> None:
    diff = [k for k in SAME if a.cfg.get(k) != b.cfg.get(k)]
    if diff:
        sys.exit(f"{a.name} and {b.name} differ in {diff}: they cannot be in one dataset")
    if a.env_args != b.env_args:
        sys.exit(f"{a.name} and {b.name} have different env_args in the hdf5: they cannot be in one dataset")
    if a.frames() != b.frames():
        sys.exit(f"{a.name} has {a.frames()} video frames, {b.name} has {b.frames()} (--all_frames): they cannot be "
                 "in one dataset")


def noise(cfg: dict) -> tuple:
    return tuple(float(cfg.get(k, 0.0)) for k in NOISE)


def storage(group: h5py.Group) -> int:
    """Bytes the datasets of an hdf5 group take on disk (compressed)."""
    total = 0

    def visit(name, obj):
        nonlocal total
        if isinstance(obj, h5py.Dataset):
            total += obj.id.get_storage_size()

    group.visititems(visit)
    return total


def total_steps(data: h5py.Group) -> np.int64:
    """Sum of num_samples over data/demo_N (the value of the attribute total)."""
    return np.int64(sum(int(data[k].attrs.get("num_samples", 0)) for k in data if re.fullmatch(r"demo_\d+", k)))


def can_link(run: Run, folder: str) -> bool:
    """True when a hard link from the run into folder works (same disk, allowed)."""
    src = run.files(run.in_view()[0])[-1][0] if run.in_view() else None
    if not src or not os.path.isfile(src):
        return False
    test = f"{folder}/.link_test{os.getpid()}"
    try:
        os.link(src, test)
        os.remove(test)
        return True
    except OSError:
        return False


def check_space(folder: str, need: int) -> None:
    free = shutil.disk_usage(folder).free
    print(f"hdf5 copy: {need / GB:.1f} GB, free on the disk: {free / GB:.1f} GB", flush=True)
    if need * 1.05 + GB > free:
        sys.exit(f"not enough free space in {folder}: need {need / GB:.1f} GB plus a margin")


def link_files(run: Run, s: int, ds: str, i: int, overwrite: bool = False) -> tuple[int, int]:
    """Hard-link the videos and the frame-0 image of source demo s as demo i of the dataset. -> (linked, copied)
    A copy (when a link is not possible) goes to a temporary name first, so a stop never leaves half a file. A file
    that is already there and equal to the source is kept; another file there is an error, or with overwrite (a
    replace that runs again after the old files moved away) a broken copy that is made again."""
    linked = copied = 0
    new = layout.demo_name(i)
    for src, suffix in run.files(s):
        folder = layout.ref_sim_dir(ds, i) if suffix == "_ref_sim.png" else layout.demo_dir(ds, i)
        dst = f"{folder}/{new}{suffix}"
        os.makedirs(folder, exist_ok=True)
        if os.path.exists(dst):
            if os.path.samefile(src, dst) or filecmp.cmp(src, dst, shallow=False):
                continue
            if not overwrite:
                raise FileExistsError(f"{dst} exists and is not the file of {run.name} demo {s}")
            os.remove(dst)
        try:
            os.link(src, dst)
            linked += 1
        except OSError as e:
            if e.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK, errno.EOPNOTSUPP):
                raise
            for old_tmp in glob.glob(f"{folder}/.{new}{suffix}.tmp*"):  # left from a stopped copy
                os.remove(old_tmp)
            tmp = f"{folder}/.{new}{suffix}.tmp{os.getpid()}"
            try:
                shutil.copy2(src, tmp)
                os.replace(tmp, dst)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
            copied += 1
    return linked, copied


def copy_demo(src: h5py.File, run: Run, s: int, data: h5py.Group, key: str, skip_raw=()) -> int:
    """Copy data/demo_<s> of a run into the dataset group under key (compressed chunks as they are). With skip_raw
    (kinds of RAW_KINDS) the observations <camera>_<kind>_raw are left out; everything else is copied. -> steps"""
    demo = src["data"][f"demo_{s}"]
    if not skip_raw:
        src.copy(demo, data, name=key)
    else:
        ends = tuple(f"_{kind}_raw" for kind in skip_raw)
        g = data.create_group(key)
        for k, v in demo.attrs.items():
            g.attrs[k] = v
        for name, item in demo.items():
            if name != "obs":
                src.copy(item, g, name=name)
                continue
            obs = g.create_group("obs")
            for k, v in item.attrs.items():
                obs.attrs[k] = v
            for term, value in item.items():
                if not term.endswith(ends):
                    src.copy(value, obs, name=term)
    g = data[key]
    g.attrs["instance_ids"] = json.dumps(run.tables, sort_keys=True)
    g.attrs["source_run"] = run.name
    g.attrs["source_demo"] = s
    return int(g.attrs.get("num_samples", g["actions"].shape[0]))


def source_row(i: int, run: Run, s: int, steps: int) -> dict:
    return {"demo": i, "source_run": run.name, "source_demo": s, "seed": run.cfg.get("seed", ""),
            "camera_noise_pos_m": run.cfg.get("camera_noise_pos_m", 0.0),
            "camera_noise_rot_deg": run.cfg.get("camera_noise_rot_deg", 0.0), "steps": steps}


def git_commit() -> str:
    r = subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "--short", "HEAD"], cwd=HERE,
                       capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def take(args) -> None:
    specs = []
    for t in args.take:
        path, sep, n = t.rpartition(":")
        if not sep or not n.isdigit() or int(n) < 1:
            sys.exit(f"--take {t}: use <run folder>:<number of demos>")
        specs.append((Run(path), int(n)))
    if len({r.name for r, _ in specs}) != len(specs):
        sys.exit("--take: a run is listed twice")
    seeds = [int(r.cfg["seed"]) for r, _ in specs]
    if len(set(seeds)) != len(seeds):
        sys.exit(f"--take: two runs have the same seed ({seeds}): the same seed repeats the start poses")
    first = specs[0][0]
    for run, _ in specs[1:]:
        check_same(first, run)

    selection = []  # (run, source demo), in dataset order
    for run, n in specs:
        good = [s for s in run.in_view() if run.usable(s)]
        if len(good) < n:
            sys.exit(f"{run.name}: {len(good)} demos with the towel in view and all videos, --take asks for {n}")
        selection += [(run, s) for s in good[:n]]
        spare = f"demo {good[n]}" if len(good) > n else "none"
        print(f"{run.name}: {n} of {len(good)} usable demos (first spare: {spare})")
    total = len(selection)
    out = os.path.abspath(args.out or os.path.dirname(first.path))
    if args.name:  # a fixed name, for a dataset that grows with --add
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name):
            sys.exit(f"--name {args.name}: use letters, digits, '_', '-' and '.' only")
        stem = name = args.name
    else:
        stem = f"{first.cfg['task']}_n{total}_seeds{''.join(map(str, seeds))}"
        name = f"{stem}_{time.strftime('%Y%m%d_%H%M')}"
    final, tmp = f"{out}/{name}", f"{out}/.{name}.partial"
    if os.path.exists(final) or os.path.exists(tmp):
        sys.exit(f"{final} (or {tmp}) exists already: use it, or remove it and run again")
    os.makedirs(out, exist_ok=True)
    left = glob.glob(f"{out}/.*.partial")
    if left:
        print(f"note: left from a stopped run, remove by hand: {', '.join(left)}")
    need = 0
    for run, _ in specs:
        # An estimate from the file size. The exact sum reads every dataset of every demo, which takes hours on a
        # busy disk. The demos of one run have nearly the same size, and check_space adds a margin.
        need += os.path.getsize(run.hdf5) * sum(1 for r, _ in selection if r is run) // max(len(run.keys), 1)
        if not can_link(run, out):  # the videos are copied, not linked
            need += sum(os.path.getsize(f) for r, s in selection if r is run for f, _ in r.files(s))
    skip_raw = sorted(set(args.skip_raw or []))
    if skip_raw:
        print(f"left out of the hdf5: the raw {', '.join(skip_raw)} data (the copy is smaller than the estimate below)")
    check_space(out, need)

    os.makedirs(tmp)
    try:
        with status.stop_signals():
            rows, steps_total, linked, copied, t0 = [], 0, 0, 0, time.time()
            with h5py.File(f"{tmp}/{stem}.hdf5", "w") as dst:
                data = dst.create_group("data")
                for run, _ in specs:
                    picked = [(i, s) for i, (r, s) in enumerate(selection) if r is run]
                    with h5py.File(run.hdf5, "r") as src:
                        if run is first:
                            for k, v in src["data"].attrs.items():
                                if k != "total":
                                    data.attrs[k] = v
                        for i, s in picked:
                            steps = copy_demo(src, run, s, data, f"demo_{i}", skip_raw)
                            steps_total += steps
                            a, b = link_files(run, s, tmp, i)
                            linked, copied = linked + a, copied + b
                            rows.append(source_row(i, run, s, steps))
                            print(f"{layout.demo_name(i)} <- {run.name} demo {s} ({i + 1}/{total}, "
                                  f"{time.time() - t0:.0f} s)", flush=True)
                data.attrs["total"] = total_steps(data)
            rows.sort(key=lambda r: r["demo"])
            status.write_csv(f"{tmp}/sources.csv", rows, status.SOURCE_COLUMNS)
            status.write_json(f"{tmp}/run_config.json", {
                "task": first.cfg["task"], "n_demos": total, "hdf5": f"{stem}.hdf5",
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_commit": git_commit(),
                "variation_seed": status.VARIATION_SEED, "tag": status.TAG, "max_attempts": status.MAX_ATTEMPTS,
                "skip_raw": skip_raw, "takes": [{"run": r.name, "n": n} for r, n in specs],
                "runs": {r.name: {"path": os.path.relpath(r.path, final), "config": r.config()} for r, _ in specs}})
            os.rename(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)  # only this run's copies and links; the runs are not touched
        print(f"stopped: removed {tmp}")
        raise
    print(f"dataset: {final}\n  {total} demos, {steps_total} steps, {linked} files linked, {copied} copied, "
          f"{time.time() - t0:.0f} s")
    status.update(final)


def resolve_run(ds: str, cfg: dict, run: str, runs_dir: str | None) -> str:
    """Folder of a source run: a path, a name in --runs_dir, or the relative path in run_config.json."""
    if os.path.isfile(f"{run}/run_config.json"):
        return run
    if runs_dir and os.path.isdir(f"{runs_dir}/{run}"):
        return f"{runs_dir}/{run}"
    known = cfg.get("runs", {}).get(run)
    if known and os.path.isdir(os.path.normpath(f"{ds}/{known['path']}")):
        return os.path.normpath(f"{ds}/{known['path']}")
    sys.exit(f"source run {run} not found: give --runs_dir <folder with the runs>")


def plan_run(ds: str, cfg: dict, plan: dict, runs_dir: str | None) -> str:
    """Folder of the new source run of a plan: --runs_dir first, then the path in the plan, then run_config.json."""
    if runs_dir and os.path.isdir(f"{runs_dir}/{plan['new_source_run']}"):
        return f"{runs_dir}/{plan['new_source_run']}"
    path = os.path.normpath(f"{ds}/{plan['new_source_path']}")
    if os.path.isfile(f"{path}/run_config.json"):
        return path
    return resolve_run(ds, cfg, plan["new_source_run"], runs_dir)


def plan_moves(ds: str, n: int, rej: str) -> list:
    """[from, to] (relative to the dataset) for every file of demo n: videos, frame-0 image, reference images and
    their check drawings from every folder under refs/, Cosmos videos and their json from every group folder. The
    target keeps the folder of the file, so two files with the same name never meet."""
    name, moves = layout.demo_name(n), []
    d = layout.demo_dir(ds, n)
    if os.path.isdir(d):
        moves += [(f"{d}/{f}", f"{rej}/{f}") for f in sorted(os.listdir(d))]
    ref_sim = f"{layout.ref_sim_dir(ds, n)}/{name}_ref_sim.png"
    if os.path.isfile(ref_sim):
        moves.append((ref_sim, f"{rej}/{name}_ref_sim.png"))
    for f in sorted(glob.glob(f"{ds}/refs/**/*.png", recursive=True)):
        if re.fullmatch(rf"(check_)?{name}(_[^/]+)?\.png", os.path.basename(f)):
            moves.append((f, f"{rej}/refs/{os.path.relpath(f, ds + '/refs')}"))
    for f in sorted(glob.glob(f"{ds}/cosmos/*/*/*")):
        if status.GROUP_RE.fullmatch(f.split(os.sep)[-2]) and re.fullmatch(rf"{name}_[^/.]+\.(mp4|json)",
                                                                         os.path.basename(f)):
            moves.append((f, f"{rej}/cosmos/{os.path.relpath(f, ds + '/cosmos')}"))
    return [[os.path.relpath(a, ds), os.path.relpath(b, ds)] for a, b in moves]


def do_moves(ds: str, n: int, moves: list) -> None:
    """Move the planned files. A file already at its target is skipped, so a stopped replace can run again."""
    for a, b in moves:
        a, b = f"{ds}/{a}", f"{ds}/{b}"
        if os.path.exists(a) and os.path.exists(b):
            raise FileExistsError(f"{os.path.relpath(a, ds)} and its target {os.path.relpath(b, ds)} both exist")
        if os.path.exists(b):
            continue
        if not os.path.exists(a):
            print(f"note: {os.path.relpath(a, ds)} is gone, not moved", flush=True)
            continue
        os.makedirs(os.path.dirname(b), exist_ok=True)
        os.replace(a, b)
    d = layout.demo_dir(ds, n)
    if os.path.isdir(d) and not os.listdir(d):
        os.rmdir(d)


def rej_folders(ds: str, n: int) -> set:
    return set(glob.glob(f"{ds}/sim_rejected/{layout.chunk(n)}/{layout.demo_name(n)}_r*"))


def used_sources(ds: str, sources: list) -> set:
    """(source run, source demo) of every demo that the dataset has now or had before a --replace."""
    used = {(r["source_run"], int(r["source_demo"])) for r in sources}
    used |= {(r["old_source_run"], int(r["old_source_demo"])) for r in status.read_csv(f"{ds}/replacements.csv")}
    for path in glob.glob(f"{ds}/sim_rejected/*/*/replaced.json"):
        try:
            j = json.load(open(path))
            if j.get("state") != "cancelled":
                used |= {(j["old_source_run"], int(j["old_source_demo"])),
                         (j["new_source_run"], int(j["new_source_demo"]))}
        except (OSError, ValueError, KeyError, TypeError):
            print(f"note: {os.path.relpath(path, ds)} is unreadable or incomplete, not used", flush=True)
    return used


def new_plan(args, ds: str, cfg: dict, n: int, data: h5py.Group) -> tuple[str, dict]:
    """Check everything, choose the spare demo and write the plan (replaced.json, state started). Call it while
    holding lock(ds). Nothing is written when a check fails."""
    sources = status.read_csv(f"{ds}/sources.csv")
    if not 0 <= n < len(sources):
        sys.exit(f"--replace {n}: the dataset has demos 0-{len(sources) - 1}")
    old = sources[n]
    claims = status.cosmos_claims(ds)
    if n in claims:
        sys.exit(f"demo {n} is in a running Cosmos job (cosmos/{claims[n]}): replace it after that job ends")
    used = used_sources(ds, sources)
    if args.from_run and not os.path.isfile(f"{args.from_run}/run_config.json"):
        sys.exit(f"--from {args.from_run}: no run_config.json there (not a run folder of make_demos.sh)")
    run = Run(resolve_run(ds, cfg, args.from_run or old["source_run"], args.runs_dir))
    runs = cfg.get("runs", {})
    old_cfg = runs[old["source_run"]]["config"]
    diff = [k for k in SAME if run.cfg.get(k) != old_cfg.get(k)]
    if diff:
        sys.exit(f"{run.name} differs from {old['source_run']} in {diff}")
    if noise(run.cfg) != noise(old_cfg):
        sys.exit(f"{run.name} has room camera noise {noise(run.cfg)}, the demo it replaces has {noise(old_cfg)}")
    if run.name not in runs and int(run.cfg["seed"]) in {int(v["config"]["seed"]) for v in runs.values()}:
        sys.exit(f"{run.name} has seed {run.cfg['seed']}, which another run of the dataset has: the same seed repeats "
                 "the start poses. Make the spare demos with a new --seed.")
    if run.name not in runs:
        old_run = Run(resolve_run(ds, cfg, old["source_run"], args.runs_dir))
        if run.frames() != old_run.frames():
            sys.exit(f"{run.name} has {run.frames()} video frames, the dataset has {old_run.frames()}")
    if json.loads(data.attrs["env_args"]) != run.env_args:
        sys.exit(f"{run.name} has different env_args than the dataset")
    spare = [s for s in run.in_view() if (run.name, s) not in used and run.usable(s)]
    if not spare:
        sys.exit(f"no unused demo left in {run.name}. Make more demos with make_demos.sh (a new --seed, the other "
                 "settings the same), run checks/check_towel_in_view.py on them, and give that run with --from <run>.")
    with h5py.File(run.hdf5, "r") as src:
        need = storage(src["data"][f"demo_{spare[0]}"])
    if not can_link(run, ds):
        need += sum(os.path.getsize(f) for f, _ in run.files(spare[0]))
    check_space(ds, need)
    k = 1 + len(rej_folders(ds, n))
    rej = f"{ds}/sim_rejected/{layout.chunk(n)}/{layout.demo_name(n)}_r{k}"
    plan = {"demo": n, "replacement": k, "old_source_run": old["source_run"], "old_source_demo": old["source_demo"],
            "new_source_run": run.name, "new_source_demo": spare[0], "new_source_path": os.path.relpath(run.path, ds),
            "reason": args.reason, "folder": os.path.relpath(rej, ds), "state": "started",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "moves": plan_moves(ds, n, rej)}
    os.makedirs(rej)
    status.write_json(f"{rej}/replaced.json", plan)
    return f"{rej}/replaced.json", plan


def mark_replace(ds: str, n) -> None:
    """Write the demo number into .replace.lock (held by this process), so status.py knows which replace runs.
    --add writes the word add."""
    with open(f"{ds}/{status.REPLACE_LOCK}", "w") as f:
        f.write(f"{n}\n")


def cancel(ds: str, cfg: dict, n: int) -> None:
    """Drop a stopped replace that has not linked the new demo yet: move the old files back, remove the copy."""
    with status.stop_signals(), status.lock(ds, status.REPLACE_LOCK):
        mark_replace(ds, n)
        with status.lock(ds, status.REFS_LOCK), status.open_hdf5(f"{ds}/{cfg['hdf5']}", "r+") as dst:
            stopped = [(p, j) for p, j in status.open_replacements(ds) if j.get("demo") == n]
            if not stopped:
                sys.exit(f"demo {n} has no replace that stopped halfway")
            jpath, plan = stopped[0]
            if plan["state"] != "started":
                sys.exit(f"the replace of demo {n} has moved all old files already: finish it with --replace {n}")
            if f"_new_demo_{n}" in dst["data"]:
                del dst["data"][f"_new_demo_{n}"]
            with status.lock(ds):
                for a, b in plan["moves"]:
                    a, b = f"{ds}/{a}", f"{ds}/{b}"
                    if os.path.exists(b) and not os.path.exists(a):
                        os.makedirs(os.path.dirname(a), exist_ok=True)
                        os.replace(b, a)
                plan.update(state="cancelled", cancelled_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
                status.write_json(jpath, plan)
    print(f"{layout.demo_name(n)}: the stopped replace is cancelled ({plan['folder']}); the demo is as before")
    status.update(ds)


def replace(args) -> None:
    ds = os.path.abspath(args.dataset_dir)
    if not status.is_dataset(ds):
        sys.exit(f"{ds} is not a dataset folder (no sources.csv)")
    cfg = status.dataset_config(ds)
    n = args.replace
    if args.cancel:
        cancel(ds, cfg, n)
        return
    key, new_key = f"demo_{n}", f"_new_demo_{n}"
    # What demo n looks like before the wait: another --replace of n may finish while this one waits.
    snapshot = (status.current_sources(ds).get(n), rej_folders(ds, n))
    # One replace at a time; then wait for make_references.py and for the hdf5 (without the list lock).
    with status.stop_signals(), status.lock(ds, status.REPLACE_LOCK):
        mark_replace(ds, n)
        with status.lock(ds, status.REFS_LOCK), status.open_hdf5(f"{ds}/{cfg['hdf5']}", "r+") as dst:
            data = dst["data"]
            stopped = [(p, j) for p, j in status.open_replacements(ds) if j.get("demo") == n]
            if stopped:
                jpath, plan = stopped[0]
                print(f"finishing the replace of demo {n} that stopped halfway ({os.path.relpath(jpath)})", flush=True)
                if args.reason or args.from_run:
                    print("note: --reason and --from are not used: the stopped plan is finished as it is", flush=True)
            else:
                if (status.current_sources(ds).get(n), rej_folders(ds, n)) != snapshot:
                    sys.exit(f"demo {n} was replaced by another make_dataset.py --replace while this one waited: "
                             "see status.py")
                if not args.reason:
                    sys.exit('--replace needs --reason "<why>"')
                with status.lock(ds):
                    jpath, plan = new_plan(args, ds, cfg, n, data)
            run = Run(plan_run(ds, cfg, plan, args.runs_dir))
            s = int(plan["new_source_demo"])
            # 1. Copy the new demo into the hdf5 under a temporary key. No list lock here: this can take a while.
            swapped = key in data and data[key].attrs.get("source_run") == run.name \
                and int(data[key].attrs.get("source_demo", -1)) == s
            if not swapped:
                if new_key in data:  # left from a stopped replace
                    del data[new_key]
                with h5py.File(run.hdf5, "r") as src:
                    copy_demo(src, run, s, data, new_key, cfg.get("skip_raw") or [])
            # 2. Under the list lock: move the old files, link the new ones, swap the hdf5 demo, update the lists.
            with status.lock(ds):
                if plan["state"] == "started":  # nothing linked yet: move every file of the number, also new ones
                    known = {a for a, _ in plan["moves"]}
                    plan["moves"] += [m for m in plan_moves(ds, n, f"{ds}/{plan['folder']}") if m[0] not in known]
                    status.write_json(jpath, plan)
                    do_moves(ds, n, plan["moves"])
                    plan["state"] = "moved"
                    status.write_json(jpath, plan)
                link_files(run, s, ds, n, overwrite=True)
                if not swapped:
                    if key in data:
                        del data[key]
                    data.move(new_key, key)
                data.attrs["total"] = total_steps(data)
                steps = int(data[key].attrs.get("num_samples", 0))
                sources = status.read_csv(f"{ds}/sources.csv")
                sources[n] = source_row(n, run, s, steps)
                status.write_csv(f"{ds}/sources.csv", sources, status.SOURCE_COLUMNS)
                cfg_now = json.load(open(f"{ds}/run_config.json"))
                if run.name not in cfg_now.setdefault("runs", {}):
                    cfg_now["runs"][run.name] = {"path": os.path.relpath(run.path, ds), "config": run.config()}
                    status.write_json(f"{ds}/run_config.json", cfg_now)
                if not any(r["folder"] == plan["folder"] for r in status.read_csv(f"{ds}/replacements.csv")):
                    status.append_csv(f"{ds}/replacements.csv", {
                        **{c: plan[c] for c in status.REPLACEMENT_COLUMNS if c in plan},
                        "replaced_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, status.REPLACEMENT_COLUMNS)
                plan.update(state="done", finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
                status.write_json(jpath, plan)
    print(f"{layout.demo_name(n)}: {plan['old_source_run']} demo {plan['old_source_demo']} -> {run.name} demo {s}; "
          f"{len(plan['moves'])} old files in {plan['folder']}")
    status.update(ds)


def video_frames(mp4: str) -> int:
    cap = cv2.VideoCapture(mp4)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def add_plan(args, ds: str, cfg: dict, data: h5py.Group) -> list:
    """Check everything and choose the demos of --add. -> [(new number, run, source demo)]. Call it while holding
    lock(ds). Nothing is written."""
    if status.open_replacements(ds):
        sys.exit("a --replace of this dataset stopped halfway: finish it first (see status.py)")
    sources = status.read_csv(f"{ds}/sources.csv")
    used, runs = used_sources(ds, sources), cfg.get("runs", {})
    base = next(iter(runs.values()))["config"]
    seeds = {int(v["config"]["seed"]) for v in runs.values()}
    frames = video_frames(f"{layout.demo_dir(ds, 0)}/{layout.demo_name(0)}_geoedge.mp4")
    plan, need, names = [], 0, set()
    for t in args.add:
        path, sep, n = t.rpartition(":")
        if not sep or not n.isdigit() or int(n) < 1:
            sys.exit(f"--add {t}: use <run folder or run name>:<number of demos>")
        run = Run(resolve_run(ds, cfg, path, args.runs_dir))
        if run.name in names:
            sys.exit(f"--add: {run.name} is listed twice")
        names.add(run.name)
        diff = [k for k in SAME if run.cfg.get(k) != base.get(k)]
        if diff:
            sys.exit(f"{run.name} differs from the dataset in {diff}")
        if json.loads(data.attrs["env_args"]) != run.env_args:
            sys.exit(f"{run.name} has different env_args than the dataset")
        if run.name not in runs:
            if int(run.cfg["seed"]) in seeds:
                sys.exit(f"{run.name} has seed {run.cfg['seed']}, which another run of the dataset has: the same "
                         "seed repeats the start poses")
            if run.frames() != frames:
                sys.exit(f"{run.name} has {run.frames()} video frames, the dataset has {frames}")
            seeds.add(int(run.cfg["seed"]))
        spare = [s for s in run.in_view() if (run.name, s) not in used and run.usable(s)]
        if len(spare) < int(n):
            sys.exit(f"{run.name}: {len(spare)} unused demos with the towel in view and all videos, --add asks for {n}")
        first = len(sources) + len(plan)
        plan += [(first + k, run, s) for k, s in enumerate(spare[:int(n)])]
        with h5py.File(run.hdf5, "r") as src:
            need += sum(storage(src["data"][f"demo_{s}"]) for s in spare[:int(n)])
        if not can_link(run, ds):
            need += sum(os.path.getsize(f) for s in spare[:int(n)] for f, _ in run.files(s))
        print(f"{run.name}: {n} of {len(spare)} unused demos -> numbers {first}-{first + int(n) - 1} (room camera "
              f"noise {noise(run.cfg)[0]} m, {noise(run.cfg)[1]} deg)", flush=True)
    check_space(ds, need)
    return plan


def add(args) -> None:
    ds = os.path.abspath(args.dataset_dir)
    if not status.is_dataset(ds):
        sys.exit(f"{ds} is not a dataset folder (no sources.csv)")
    cfg = status.dataset_config(ds)
    tmp_key = "_add_demo_"  # a demo of this --add in the hdf5, before it gets its number
    # As --replace: one at a time; wait for make_references.py and for the hdf5 (without the list lock).
    with status.stop_signals(), status.lock(ds, status.REPLACE_LOCK):
        mark_replace(ds, "add")
        with status.lock(ds, status.REFS_LOCK), status.open_hdf5(f"{ds}/{cfg['hdf5']}", "r+") as dst:
            data = dst["data"]
            with status.lock(ds):
                plan = add_plan(args, ds, cfg, data)

            def holds(key: str, run: Run, s: int) -> bool:
                return key in data and data[key].attrs.get("source_run") == run.name \
                    and int(data[key].attrs.get("source_demo", -1)) == s

            # 1. Copy the new demos into the hdf5 under temporary keys. No list lock here: this takes a while.
            for key in [k for k in data if k.startswith(tmp_key)]:
                i = int(key[len(tmp_key):])
                if not any(i == j and holds(key, run, s) for j, run, s in plan):  # a half copy, or of another --add
                    del data[key]
            t0 = time.time()
            for run in dict.fromkeys(r for _, r, _ in plan):
                with h5py.File(run.hdf5, "r") as src:
                    for i, r, s in plan:
                        if r is not run:
                            continue
                        if holds(f"demo_{i}", run, s) or holds(f"{tmp_key}{i}", run, s):
                            continue  # copied by this command before it stopped
                        if f"demo_{i}" in data:
                            sys.exit(f"the hdf5 has demo_{i} already, from another demo: see status.py")
                        copy_demo(src, run, s, data, f"{tmp_key}{i}", cfg.get("skip_raw") or [])
                        dst.flush()
                        print(f"{layout.demo_name(i)} <- {run.name} demo {s} ({time.time() - t0:.0f} s)", flush=True)
            # 2. Under the list lock: link the files, give the demos their numbers, update the lists.
            with status.lock(ds):
                rows = status.read_csv(f"{ds}/sources.csv")
                for i, run, s in plan:
                    link_files(run, s, ds, i)
                    if f"{tmp_key}{i}" in data:
                        data.move(f"{tmp_key}{i}", f"demo_{i}")
                    rows.append(source_row(i, run, s, int(data[f"demo_{i}"].attrs.get("num_samples", 0))))
                data.attrs["total"] = total_steps(data)
                dst.flush()
                cfg_now = json.load(open(f"{ds}/run_config.json"))
                for run in dict.fromkeys(r for _, r, _ in plan):
                    numbers = status.ranges(i for i, r, _ in plan if r is run)
                    if run.name not in cfg_now.setdefault("runs", {}):
                        cfg_now["runs"][run.name] = {"path": os.path.relpath(run.path, ds), "config": run.config()}
                    added = cfg_now.setdefault("added", [])
                    if not any(a["run"] == run.name and a["demos"] == numbers for a in added):  # not written yet
                        added.append({"run": run.name, "demos": numbers, "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                      "git_commit": git_commit()})
                cfg_now["n_demos"] = len(rows)
                status.write_json(f"{ds}/run_config.json", cfg_now)
                status.write_csv(f"{ds}/sources.csv", rows, status.SOURCE_COLUMNS)  # last: the demos count from here
    print(f"added demos {status.ranges(i for i, _, _ in plan)}: the dataset has {len(rows)} demos now")
    status.update(ds)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", nargs="?", help="dataset folder (for --replace and --add)")
    ap.add_argument("--take", nargs="+", metavar="RUN:N", help="make a new dataset from these runs")
    ap.add_argument("--out", default=None, help="folder for the new dataset (default: the folder of the first run)")
    ap.add_argument("--name", default=None, help="with --take: name of the dataset folder and of its hdf5 "
                    "(default: <task>_n<total>_seeds<seeds>, and the folder gets the date and time)")
    ap.add_argument("--skip_raw", nargs="+", choices=RAW_KINDS, default=None, metavar="KIND",
                    help="with --take: leave the raw data of these kinds (depth, normals, instance) out of the hdf5")
    ap.add_argument("--replace", type=int, default=None, metavar="DEMO", help="demo number to replace")
    ap.add_argument("--reason", default=None, help="why the demo is replaced (goes into replacements.csv)")
    ap.add_argument("--from", dest="from_run", default=None, help="take the spare demo from this run folder")
    ap.add_argument("--runs_dir", default=None, help="folder with the source runs (when they were moved)")
    ap.add_argument("--cancel", action="store_true",
                    help="with --replace: drop a replace that stopped before the new demo was linked")
    ap.add_argument("--add", nargs="+", metavar="RUN:N", help="put the next N unused demos of these runs after the "
                    "last demo of the dataset")
    args = ap.parse_args()
    if args.name and not args.take:
        ap.error("--name is for --take only")
    if args.skip_raw and not args.take:
        ap.error("--skip_raw is for --take only (--add and --replace follow run_config.json of the dataset)")
    if args.take and args.replace is None and not args.add and not args.dataset_dir:
        take(args)
    elif args.dataset_dir and args.replace is not None and not args.take and not args.add:
        replace(args)
    elif args.dataset_dir and args.add and args.replace is None and not args.take:
        add(args)
    else:
        ap.error("use --take RUN:N ..., or <dataset_dir> --replace DEMO --reason TEXT, or <dataset_dir> --add "
                 "RUN:N ...")


if __name__ == "__main__":
    main()
