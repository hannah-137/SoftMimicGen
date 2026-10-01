"""Make the reference images and the Cosmos videos of a dataset group by group, so the GPUs do not wait.

  python experiments/policy_data/run_queue.py <dataset_dir> [--demos 150-749] [--count 100] [--spec v2] \
      -- <run_cosmos.py options>

  for example:
  export OPENAI_API_KEY=...
  python experiments/policy_data/run_queue.py <dataset> --demos 150-749 -- --gpus 2,3 --cp 2 \
      --framework <cosmos-framework> --checkpoint <Cosmos3-Super-fp8> --hf_home <hf cache> --tools <tools>

For each group of 50 demos (000-049, 050-099, ...) in order, it does the same steps as the README:
  1. make_references.py --demos <group>, then checks/check_references.py, then make_references.py --retry and the
     check again while images fail, until they pass or max_attempts images are used,
  2. it waits until no other Cosmos job of the dataset runs on the same GPUs,
  3. run_cosmos.py --demos <group> with the options after "--".
It makes the images of the next group while Cosmos makes the videos of this group (the image API needs no GPU), so
the next Cosmos job starts right after the last one. A group with nothing left to make is skipped, and so are the
demos that another job works on now. The queue does not wait for the review: review the videos on the review page
while it runs.

A demo whose reference image failed max_attempts times gets no video; status.py prints the make_dataset.py
--replace command for it. A demo whose image the API did not make (network, no credit) is tried again in the next
step; if it still has no image, the log says so and the group goes on without it. The queue does not replace
demos, and it does not make rejected videos again. When run_cosmos.py fails (a video is missing, or the framework
stopped), the queue runs it once more for the missing videos. If videos are still missing, the queue goes on with
the next group when this group got at least one video, and stops when it got none (then the framework itself
fails: fix the cause and start the queue again). A new start goes on where the last one stopped, because every
step skips what is done.

--count N takes only the first N demos (in --demos) that still need a video, for example --count 100 for the
next two groups; then the queue stops. Start it again with a new --count to go on.
--spec v3 makes the new reference images and the videos of these demos with spec version v3 (specs/v3.json, see
SPEC.md). A demo whose image is of another version and has no video yet gets a new v3 image first (make_references.py
--redo; the old image goes to refs_rejected/ and counts as an attempt). Without --spec, new images get the default
version (SPEC in status.py) and every video follows the version of its image. Change the version at the start of a
group of 50, so each group has one version.
Stop: create the file <dataset>/.queue_stop to stop after the Cosmos job that runs now (the queue removes the file),
or press Ctrl-C or send SIGTERM to stop now: run_cosmos.py keeps the videos that are done.
Needs OPENAI_API_KEY when a group still needs reference images. All output goes to this script's output: send it
to a log file.
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time

import layout
import status
import variations

HERE = os.path.dirname(os.path.abspath(__file__))
STOP_FILE = ".queue_stop"
WAIT = 10  # seconds between looks at the running Cosmos jobs and at the stop file
NEEDS_IMAGE = ("needs_ref", "needs_check", "ref_failed")
TO_MAKE = NEEDS_IMAGE + ("needs_video",)


def log(msg: str) -> None:
    print(f"[run_queue {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


class Steps:
    """Runs the scripts as child processes and stops the running ones on Ctrl-C or SIGTERM."""

    def __init__(self):
        self.children = set()
        self.guard = threading.Lock()
        self.closed = False  # after stop(): start nothing new

    def run(self, script: str, *args: str) -> int:
        with self.guard:
            if self.closed:
                return 1
            p = subprocess.Popen([sys.executable, os.path.join(HERE, script), *args])
            self.children.add(p)
        code = p.wait()  # on Ctrl-C or SIGTERM the child stays in the set, so stop() can reach it
        with self.guard:
            self.children.discard(p)
        return code

    def stop(self, signalled: bool) -> None:
        """Ctrl-C reaches the children too, so wait for them first. A signal sent to the queue alone does not:
        pass SIGTERM on (run_cosmos.py then stops its job and keeps the videos that are done)."""
        with self.guard:
            self.closed = True
            kids = list(self.children)
        if not signalled:
            end = time.time() + 60
            while time.time() < end and any(p.poll() is None for p in kids):
                time.sleep(1)
        for p in kids:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for p in kids:
            try:
                p.wait(timeout=600)
            except subprocess.TimeoutExpired:
                p.kill()


def rows_of(ds: str, demos: list) -> dict:
    """demo number -> row of dataset.csv, for the given demos that no job works on now (column note empty)."""
    with status.lock(ds):
        rows, _, _ = status.collect(ds)
    wanted = set(demos)
    return {r["demo"]: r for r in rows if r["demo"] in wanted and not r["note"]}


def states(ds: str, demos: list) -> dict:
    """demo number -> state, for the given demos that no job works on now."""
    return {d: r["state"] for d, r in rows_of(ds, demos).items()}


def other_spec(row: dict, spec: str) -> bool:
    """True when --spec is given and the demo has an image of another version but no video yet."""
    return bool(spec) and row["state"] in ("needs_check", "needs_video") and row["spec"] not in ("", spec)


def make_images(steps: Steps, ds: str, demos: list, max_attempts: int, spec: str) -> None:
    """Reference images of one group: make them (with --spec: also the images of another version that have no
    video yet), check them, and make the failed ones again. A demo without a passed image at the end gets no video
    now (the log says which)."""
    group = layout.chunk(demos[0])
    spec_args = ["--spec", spec] if spec else []
    start = rows_of(ds, demos)
    redo = [d for d, r in start.items() if other_spec(r, spec)]
    need = sorted({d for d, r in start.items() if r["state"] in NEEDS_IMAGE} | set(redo))
    if not need:
        return
    log(f"{group}: reference images for demos {status.ranges(need)}")
    if redo:
        log(f"{group}: images of another spec version, made again as {spec}: demos {status.ranges(redo)}")
        if steps.run("make_references.py", ds, "--redo", *spec_args, "--reason", f"made again as spec {spec}",
                     "--demos", *status.ranges(redo).split()):
            log(f"{group}: make_references.py --redo failed for some images (see above)")

    def check():
        unchecked = [d for d, s in states(ds, need).items() if s == "needs_check"]
        if unchecked and steps.run("checks/check_references.py", ds, "--demos",
                                   *status.ranges(unchecked).split()) not in (0, 1):  # 1: some images failed
            log(f"{group}: check_references.py failed (see above)")

    for step in range(max_attempts + 1):  # each step makes at most one image per demo; make_references.py keeps
        check()                           # the limit of max_attempts images per demo
        now = states(ds, need)
        new = [d for d, s in now.items() if s == "needs_ref"]  # no image yet, also when the API did not answer
        failed = [d for d, s in now.items() if s == "ref_failed"] if step < max_attempts else []
        if not new and not failed:
            break
        codes = []
        if new:
            codes.append(steps.run("make_references.py", ds, *spec_args, "--demos", *status.ranges(new).split()))
        if failed:
            log(f"{group}: make the failed images again: demos {status.ranges(failed)}")
            codes.append(steps.run("make_references.py", ds, "--retry", *spec_args, "--demos",
                                   *status.ranges(failed).split()))
        if any(codes):
            log(f"{group}: make_references.py exit {max(codes)} (some images were not made; the next step tries "
                "again)")
    check()
    end = rows_of(ds, need)
    left = [d for d, r in end.items() if r["state"] in NEEDS_IMAGE]
    if left:
        log(f"{group}: no passed reference image for demos {status.ranges(left)}: they get no video now")
    old = [d for d, r in end.items() if other_spec(r, spec)]
    if old:
        log(f"{group}: the image of demos {status.ranges(old)} is still not spec {spec}: they get no video now")


def gpus_of(cosmos_args: list) -> set:
    """The GPU indices in the run_cosmos.py options ("--gpus 2,3" or "--gpus=2,3"; run_cosmos.py default: 0)."""
    value = "0"
    for i, a in enumerate(cosmos_args):
        if a == "--gpus" and i + 1 < len(cosmos_args):
            value = cosmos_args[i + 1]
        elif a.startswith("--gpus="):
            value = a.split("=", 1)[1]
    return {g.strip() for g in value.split(",") if g.strip()}


def busy(ds: str, gpus: set) -> str:
    """The name of a running Cosmos job of the dataset that uses one of these GPUs, or ""."""
    for name, cfg in status.running_cosmos(ds).items():
        if {g.strip() for g in str(cfg.get("gpus", "")).split(",")} & gpus:
            return name
    return ""


def main():
    argv = sys.argv[1:]
    cosmos_args = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", help="dataset folder made by make_dataset.py --take")
    ap.add_argument("--demos", nargs="+", default=None, help="demo numbers and ranges, e.g. 150-749 (default: all)")
    ap.add_argument("--count", type=int, default=None,
                    help="take only the first N demos that still need a video (in --demos), then stop")
    ap.add_argument("--spec", default=None,
                    help=f"spec version of new images and videos, e.g. v3 (specs/<version>.json; default: new images "
                         f"{status.SPEC}, videos the version of their image)")
    args = ap.parse_args(argv[: argv.index("--")] if "--" in argv else argv)
    ds = os.path.abspath(args.dataset_dir)
    if not status.is_dataset(ds):
        sys.exit(f"{ds} is not a dataset folder (no sources.csv)")
    if not cosmos_args:
        sys.exit('give the run_cosmos.py options after "--" (at least --framework and --checkpoint)')
    if any(a == "--demos" or a.startswith("--demos=") or a == "--max" for a in cosmos_args):
        sys.exit("do not give --demos or --max after \"--\": the queue gives each job one group")
    if any(a == "--spec" or a.startswith("--spec=") for a in cosmos_args):
        sys.exit("give --spec before \"--\": the queue passes it to make_references.py and run_cosmos.py")
    if args.spec:
        if args.spec not in variations.versions():
            sys.exit(f"--spec {args.spec}: no specs/{args.spec}.json (versions: {', '.join(variations.versions())})")
        try:
            variations.load(args.spec)
        except ValueError as e:
            sys.exit(str(e))
        cosmos_args += ["--spec", args.spec]
    stop_file = os.path.join(ds, STOP_FILE)
    if os.path.exists(stop_file):
        os.remove(stop_file)
        log(f"removed an old {STOP_FILE}")
    max_attempts = int(status.dataset_config(ds)["max_attempts"])
    gpus = gpus_of(cosmos_args)

    wanted = rows_of(ds, status.parse_demos(args.demos) or range(len(status.current_sources(ds))))
    if args.count is not None and args.count < 1:
        sys.exit("--count must be 1 or more")
    todo = [d for d, r in sorted(wanted.items()) if r["state"] in TO_MAKE][: args.count]
    wanted = {d: r for d, r in wanted.items() if d in todo}
    groups = {}
    for d in todo:
        groups.setdefault(layout.chunk(d), []).append(d)
    if not groups:
        log("nothing to make")
        status.update(ds)
        return
    if (any(r["state"] in NEEDS_IMAGE or other_spec(r, args.spec) for r in wanted.values())
            and not os.environ.get("OPENAI_API_KEY")):
        sys.exit("OPENAI_API_KEY is not set: some groups still need reference images")
    order = sorted(groups)
    log(f"groups {', '.join(order)} on GPUs {','.join(sorted(gpus))}, spec {args.spec or 'of each image'}; stop "
        f"after the running job: touch {stop_file}")

    steps = Steps()

    def images(group):
        make_images(steps, ds, groups[group], max_attempts, args.spec)

    def stopped() -> bool:
        return os.path.exists(stop_file)

    def cosmos(group) -> bool:
        """The videos of the demos of the group that are ready. A failed job (a video is missing, or the framework
        stopped) is run once more for the videos that are still missing. When videos are still missing after that,
        the queue goes on if the group got at least one video, and stops (False) if it got none."""
        first = None
        for attempt in (1, 2):
            ready = [d for d, r in rows_of(ds, groups[group]).items()
                     if r["state"] == "needs_video" and not other_spec(r, args.spec)]
            first = ready if first is None else first
            if not ready:
                if attempt == 1:
                    log(f"{group}: no demo is ready for a video: skipped")
                return True
            log(f"{group}: Cosmos videos for demos {status.ranges(ready)}" + (" (again)" if attempt == 2 else ""))
            code = steps.run("run_cosmos.py", ds, "--demos", *status.ranges(ready).split(), *cosmos_args)
            log(f"{group}: run_cosmos.py exit {code}")
            if code == 0:
                return True
        left = [d for d, r in rows_of(ds, groups[group]).items()
                if r["state"] == "needs_video" and not other_spec(r, args.spec)]
        if len(left) < len(first):
            log(f"{group}: still no video for demos {status.ranges(left)} after two jobs: the queue goes on "
                "without them (run the queue again later to make them)")
            return True
        return False

    try:
        with status.stop_signals():
            prep = threading.Thread(target=images, args=(order[0],), daemon=True)
            prep.start()
            for i, group in enumerate(order):
                while prep.is_alive():  # the images of this group (short join: Ctrl-C still works)
                    prep.join(1)
                if stopped():
                    log(f"{STOP_FILE} found: queue stopped")
                    break
                if i + 1 < len(order):  # the images of the next group, while Cosmos runs
                    prep = threading.Thread(target=images, args=(order[i + 1],), daemon=True)
                    prep.start()
                job = busy(ds, gpus)
                if job:
                    log(f"{group}: waiting for the Cosmos job {job} on the same GPUs")
                while job and not stopped():
                    time.sleep(WAIT)
                    job = busy(ds, gpus)
                if stopped():
                    log(f"{STOP_FILE} found: queue stopped")
                    break
                if not cosmos(group):
                    log(f"{group}: run_cosmos.py failed twice and made no video: queue stopped (see its log above)")
                    break
            while prep.is_alive():  # a group whose images are being made when the queue stops: let it finish
                prep.join(1)
    except KeyboardInterrupt as e:
        log("stopping")
        steps.stop(signalled=str(e).startswith("signal"))
    finally:
        if os.path.exists(stop_file):
            os.remove(stop_file)
    status.update(ds)


if __name__ == "__main__":
    main()
