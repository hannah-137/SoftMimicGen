"""Make the reference images and the Cosmos videos of a dataset group by group, so the GPUs do not wait.

  python experiments/policy_data/run_queue.py <dataset_dir> [--demos 150-749] -- <run_cosmos.py options>

  for example:
  export OPENAI_API_KEY=...
  python experiments/policy_data/run_queue.py <dataset> --demos 150-749 -- --gpus 2,3 --cp 2 --towel_prompt \
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


def states(ds: str, demos: list) -> dict:
    """demo number -> state, for the given demos that no job works on now (column note empty)."""
    with status.lock(ds):
        rows, _, _ = status.collect(ds)
    wanted = set(demos)
    return {r["demo"]: r["state"] for r in rows if r["demo"] in wanted and not r["note"]}


def make_images(steps: Steps, ds: str, demos: list, max_attempts: int) -> None:
    """Reference images of one group: make them, check them, and make the failed ones again. A demo without a
    passed image at the end gets no video now (the log says which)."""
    group = layout.chunk(demos[0])
    need = [d for d, s in states(ds, demos).items() if s in NEEDS_IMAGE]
    if not need:
        return
    log(f"{group}: reference images for demos {status.ranges(need)}")

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
            codes.append(steps.run("make_references.py", ds, "--demos", *status.ranges(new).split()))
        if failed:
            log(f"{group}: make the failed images again: demos {status.ranges(failed)}")
            codes.append(steps.run("make_references.py", ds, "--retry", "--demos", *status.ranges(failed).split()))
        if any(codes):
            log(f"{group}: make_references.py exit {max(codes)} (some images were not made; the next step tries "
                "again)")
    check()
    left = [d for d, s in states(ds, need).items() if s in NEEDS_IMAGE]
    if left:
        log(f"{group}: no passed reference image for demos {status.ranges(left)}: they get no video now")


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
    args = ap.parse_args(argv[: argv.index("--")] if "--" in argv else argv)
    ds = os.path.abspath(args.dataset_dir)
    if not status.is_dataset(ds):
        sys.exit(f"{ds} is not a dataset folder (no sources.csv)")
    if not cosmos_args:
        sys.exit('give the run_cosmos.py options after "--" (at least --framework and --checkpoint)')
    if any(a == "--demos" or a.startswith("--demos=") or a == "--max" for a in cosmos_args):
        sys.exit("do not give --demos or --max after \"--\": the queue gives each job one group")
    stop_file = os.path.join(ds, STOP_FILE)
    if os.path.exists(stop_file):
        os.remove(stop_file)
        log(f"removed an old {STOP_FILE}")
    max_attempts = int(status.dataset_config(ds)["max_attempts"])
    gpus = gpus_of(cosmos_args)

    wanted = states(ds, status.parse_demos(args.demos) or range(len(status.current_sources(ds))))
    groups = {}
    for d, s in sorted(wanted.items()):
        if s in TO_MAKE:
            groups.setdefault(layout.chunk(d), []).append(d)
    if not groups:
        log("nothing to make")
        status.update(ds)
        return
    if any(s in NEEDS_IMAGE for s in wanted.values()) and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set: some groups still need reference images")
    order = sorted(groups)
    log(f"groups {', '.join(order)} on GPUs {','.join(sorted(gpus))}; stop after the running job: "
        f"touch {stop_file}")

    steps = Steps()

    def images(group):
        make_images(steps, ds, groups[group], max_attempts)

    def stopped() -> bool:
        return os.path.exists(stop_file)

    def cosmos(group) -> bool:
        """The videos of the demos of the group that are ready. A failed job (a video is missing, or the framework
        stopped) is run once more for the videos that are still missing. When videos are still missing after that,
        the queue goes on if the group got at least one video, and stops (False) if it got none."""
        first = None
        for attempt in (1, 2):
            ready = [d for d, s in states(ds, groups[group]).items() if s == "needs_video"]
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
        left = [d for d, s in states(ds, groups[group]).items() if s == "needs_video"]
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
