"""Make reference images from the frame-0 images with the OpenAI image edit API (gpt-image-2.5-sunburst).

  export OPENAI_API_KEY=...
  python experiments/policy_data/make_references.py <run_dir> [--demos 0-49] [--seed 0]
      [--model gpt-image-2.5-sunburst] [--quality medium] [--size 2048x1024] [--tag 01] [--workers 3]
      [--retry] [--redo --reason "<why>"] [--max_attempts 3] [--dry_run] [--prompt_file <file>] [--refs <folder>]

<run_dir> is a run folder (make_demos.sh) or a dataset folder (make_dataset.py).
Input: <run_dir>/ref_sim/000-049/demo_NNN_ref_sim.png (frame 0, room | wrist, 1024x512, made by make_videos.py).
Output: <run_dir>/refs/000-049/demo_NNN_<tag>.png (1024x512) and <run_dir>/refs/references.csv with one row per
image: the 7 axes and the place type, the prompt, the tokens and the cost. --demos takes numbers and ranges
(0-49 60 70-99); make the images in batches of any size, the ratios do not depend on the batch.

Every image gets its own variation from variations.py (demo index, seed, tag, attempt): all 7 axes change, the
place types are exact in every group of 50 demos, and no two images in the csv share a combination. The prompt is
static on purpose: no action words (they make the model fold the towel), and it lists what must stay the same. The
API has no seed, so the same call twice gives two different images.

Size: the API needs at least 655,360 pixels, so the image is made at 2048x1024 and resized to 1024x512.
Flow: make_references.py -> checks/check_references.py -> make_references.py --retry -> check again -> run_cosmos.py
Images that already exist are skipped, so the same command again makes only the missing ones (for example after
an API error or a stop).
--retry reads refs/failed_references.txt (written by checks/check_references.py). It makes the images that failed
the check again, with this --tag, as a new attempt: same place type, new combination. Only an image whose failed
check row has the sha1 of the file on disk is made again; an image made after its check is skipped (check it
first). It does not make images that were never made: run without --retry for those. The replaced image goes to
<run_dir>/refs_rejected/000-049/<name>_attempt<N>.png, and refs_rejected/rejected.csv gets a row with its scores
and the reason.
--redo --demos ... --reason "<why>" makes the images of these demos again even when they passed the check (for
example after a change of the variation lists), as a new attempt. The old image goes to refs_rejected/ with the
reason. A demo that has a Cosmos video (or that a running run_cosmos.py works on) is not made again: reject its
video on the review page with "reference image problem" instead, so the image and the video are made again together.
--max_attempts (default 3): a demo gets at most this many images. In a dataset the count starts again when
make_dataset.py --replace puts another demo at the number. When a demo reaches the limit, the script prints the
make_dataset.py --replace command for it.
In a dataset folder (with the default --refs), --seed, --tag and --max_attempts come from its run_config.json
(variation_seed, tag, max_attempts), so every batch uses the same draw and the ratios stay exact. Other values are
refused.
--prompt_file uses another prompt: a text file with {variation} where the axis sentences go. --refs writes to another
folder (default <run_dir>/refs, the replaced images then go to <folder>_rejected). Two folders with the same --tag get
the same axis sentences, so two prompts can be compared on the same variations.
An image in the wrong folder is not made again: move it into the group folder of its demo.
--dry_run prints the prompt and the variations and writes nothing. It needs no API key.
Only one make_references.py runs on a folder at a time: a second one waits (.make_references.lock in the run folder,
or in the --refs folder when it is outside the run folder), then makes only what is still missing. In a dataset
folder, status.py runs at the end.
The run stops when the account has no credit left (error type insufficient_quota). Rate limits and server errors
are retried.
Needs: bash experiments/policy_data/setup.sh (openai), opencv-python, numpy. Cost per image is printed from the
usage of every call.
"""

import argparse
import base64
import collections
import contextlib
import glob
import importlib.util
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import layout
import status
import variations

OUT_W, OUT_H = 1024, 512
MIN_PIXELS = 655_360  # smallest image the API makes
PROMPT = (
    "Turn this simulator image into a realistic photo. It is a split screen: left the room camera, right the camera on "
    "the robot gripper. Keep the composition exactly the same in both halves: the same camera viewpoint and framing, "
    "and the same size and position of the robot, the towel, the table and the background. Keep the same aspect "
    "ratio. Do not crop, zoom or pad. The robot keeps the same pose. The towel keeps exactly the same position, shape, "
    "size and orientation. The table keeps the same position, size and shape; only its top surface changes. Do not "
    "add, remove, move, fold, bend or reshape anything. There is exactly one towel, flat on the table. {variation} "
    "The towel has the same color and texture on both sides. Change only the materials, textures, lighting and the "
    "look of the background. No text, no logos, no watermark."
)
# USD per 1M tokens, standard rate of gpt-image-2 and gpt-image-2.5-sunburst (2026-09). Only for the cost column;
# the bill is what OpenAI charges.
PRICE = {"text": 5.0, "image_in": 8.0, "image_out": 30.0}
COLUMNS = ["demo", "name", "attempt", "source", "seed", "model", "quality", "size", "place_type", *variations.FIELDS,
           "prompt", "input_tokens", "output_tokens", "cost_usd", "seconds", "finished_at"]
RETRY_WAIT = [15, 30, 60, 120, 240]  # seconds between tries after a rate limit or a server error
# Error codes of a 429 that means "no credit left" (the error type is insufficient_quota). No retry for these.
NO_CREDIT_CODES = {"insufficient_quota", "credit_balance_exhausted"}


class NoCredit(Exception):
    """The account has no credit left. Every other call would fail too."""


def check_size(size: str) -> None:
    try:
        w, h = (int(x) for x in size.lower().split("x"))
    except ValueError:
        sys.exit(f"--size {size}: use WIDTHxHEIGHT, for example 2048x1024")
    if w != 2 * h or w % 16 or h % 16 or w * h < MIN_PIXELS:
        sys.exit(f"--size {size}: needs aspect 2:1, both sides a multiple of 16, at least {MIN_PIXELS:,} pixels "
                 "(for example 2048x1024)")


def retry_list(refs: str, tag: str) -> list:
    """Demo indices to make again, from refs/failed_references.txt (written by checks/check_references.py)."""
    path = f"{refs}/failed_references.txt"
    if not os.path.isfile(path):
        if os.path.isfile(f"{refs}/check_references.csv"):
            print("no failed image in refs/check_references.csv: nothing to make again")
            return []
        sys.exit(f"{path} not found: run checks/check_references.py first")
    lines = [tuple(x.strip() for x in line.partition(":")[::2]) for line in open(path)]
    # A demo whose image is only in the wrong folder is also "missing": moving the file fixes both, no new image.
    moved = {layout.demo_index(n) for n, r in lines if r.startswith("wrong folder")}
    idxs, never = [], []
    for name, reason in lines:
        if not name:
            continue
        if reason.startswith("wrong folder"):
            print(f"{name}: {reason} (not made again)", flush=True)
        elif reason == "missing":  # the name is demo_NNN, no image yet
            if layout.demo_index(name) not in moved:
                never.append(layout.demo_index(name))
        elif name.endswith(f"_{tag}"):
            idxs.append(layout.demo_index(name))
        else:
            print(f"{name}: another tag than --tag {tag}, skipped", flush=True)
    if never:
        print(f"no image yet for demos {status.ranges(never)}: --retry does not make them, run without --retry",
              flush=True)
    return idxs


def cost_usd(usage) -> float:
    det = getattr(usage, "input_tokens_details", None)
    text = getattr(det, "text_tokens", 0) or 0
    image = getattr(det, "image_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    return (text * PRICE["text"] + image * PRICE["image_in"] + out * PRICE["image_out"]) / 1e6


def edit_image(client, model: str, src: str, prompt: str, size: str, quality: str):
    """One API call with retries on rate limits and server errors. Returns (image BGR, usage)."""
    import openai

    for wait in RETRY_WAIT + [None]:
        try:
            with open(src, "rb") as f:
                r = client.images.edit(model=model, image=f, prompt=prompt, size=size, quality=quality)
            data = base64.b64decode(r.data[0].b64_json)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("the API returned an image that cannot be decoded")
            return img, r.usage
        except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError) as e:
            if getattr(e, "type", None) == "insufficient_quota" or getattr(e, "code", None) in NO_CREDIT_CODES:
                raise NoCredit(str(e)) from e
            if wait is None:
                raise
            print(f"  {type(e).__name__}, retry in {wait} s", flush=True)
            time.sleep(wait)


def reject(run: str, rejected: str, idx: int, name: str, dst: str, old_row: dict | None, check: dict) -> None:
    """Move the image that --retry replaces to <rejected>/000-049/ and add a row to rejected.csv. Hold the lock."""
    old = f"attempt{old_row['attempt']}" if old_row else "original"
    folder = layout.ref_dir(rejected, idx)
    os.makedirs(folder, exist_ok=True)
    target = f"{folder}/{name}_{old}.png"
    k = 1
    while os.path.exists(target):  # an image of the same attempt was rejected before (made by hand)
        k += 1
        target = f"{folder}/{name}_{old}_{k}.png"
    os.replace(dst, target)
    status.append_csv(f"{rejected}/rejected.csv", {
        "demo": idx, "name": name, "attempt": old_row["attempt"] if old_row else "",
        "source": old_row.get("source", "") if old_row else "", "file": os.path.relpath(target, run),
        "room_score": check.get("room_score", ""), "wrist_score": check.get("wrist_score", ""),
        "reason": check.get("reason", ""), "rejected_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        status.REF_REJECTED_COLUMNS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--demos", nargs="*", default=None,
                    help="demo numbers and ranges, e.g. 0-49 60 (default: every ref_sim image; with --retry: only "
                         "these of the failed ones)")
    ap.add_argument("--seed", type=int, default=None,
                    help="variation seed (default 0; in a dataset: its variation_seed)")
    ap.add_argument("--model", default="gpt-image-2.5-sunburst")
    ap.add_argument("--quality", default="medium", help="low, medium, high (auto = the API decides)")
    ap.add_argument("--size", default="2048x1024", help="size the API makes; the file is resized to 1024x512")
    ap.add_argument("--tag", default=None, help="reference number in the file name: demo_NNN_<tag>.png (default 01)")
    ap.add_argument("--workers", type=int, default=3, help="parallel API calls")
    ap.add_argument("--retry", action="store_true", help="make the images in refs/failed_references.txt again")
    ap.add_argument("--redo", action="store_true", help="make the images of --demos again, also the passed ones")
    ap.add_argument("--reason", default=None, help="with --redo: why (goes into refs_rejected/rejected.csv)")
    ap.add_argument("--max_attempts", type=int, default=None,
                    help=f"images per demo at most (default {status.MAX_ATTEMPTS}; in a dataset: its max_attempts)")
    ap.add_argument("--dry_run", action="store_true", help="print the prompt and the variations, call nothing")
    ap.add_argument("--prompt_file", default=None, help="prompt text with {variation} (default: the prompt here)")
    ap.add_argument("--refs", default=None, help="output folder (default <run_dir>/refs)")
    args = ap.parse_args()
    check_size(args.size)
    wanted = status.parse_demos(args.demos)
    if args.redo and (args.retry or not wanted or not args.reason):
        sys.exit('--redo needs --demos and --reason "<why>", and not --retry')

    run = os.path.abspath(args.run_dir)
    if not os.path.isdir(run):
        sys.exit(f"no such folder: {run}")
    refs = os.path.abspath(args.refs) if args.refs else f"{run}/refs"
    rejected = f"{refs}_rejected"
    fixed = {"seed": 0, "tag": status.TAG, "max_attempts": status.MAX_ATTEMPTS}
    if status.is_dataset(run) and refs == f"{run}/refs":  # one draw for the whole dataset
        cfg = status.dataset_config(run)
        fixed = {"seed": int(cfg["variation_seed"]), "tag": cfg["tag"], "max_attempts": int(cfg["max_attempts"])}
        for key, value in fixed.items():
            if getattr(args, key) not in (None, value):
                sys.exit(f"--{key} {getattr(args, key)}: this dataset uses {value} (run_config.json)")
    for key, value in fixed.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    prompt = open(args.prompt_file).read().strip() if args.prompt_file else PROMPT
    if prompt.count("{variation}") != 1:
        sys.exit("the prompt needs {variation} exactly once")
    if not args.dry_run:
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY is not set")
        if importlib.util.find_spec("openai") is None:
            sys.exit("the openai package is missing: bash experiments/policy_data/setup.sh")
    # Locks live in the folder that is written: the run or dataset folder, or --refs when it is outside the run.
    lock_dir = run if status.is_dataset(run) or refs.startswith(run + os.sep) else refs
    if not args.dry_run:
        os.makedirs(refs, exist_ok=True)
    # One run per folder: a second one waits here, then sees the images of the first and makes only the rest.
    run_lock = contextlib.nullcontext() if args.dry_run else status.lock(lock_dir, status.REFS_LOCK)
    with run_lock:
        make(args, run, refs, rejected, prompt, wanted, lock_dir)


def make(args, run: str, refs: str, rejected: str, prompt: str, wanted: list | None, lock_dir: str) -> None:
    csv_path = f"{refs}/references.csv"
    rows = status.read_csv(csv_path)
    sources = status.current_sources(run)  # empty for a run folder
    used = {tuple(r[f] for f in variations.FIELDS) for r in rows}
    attempts = {}  # name -> number of the next attempt (never reused, so every attempt draws a new combination)
    last = {}  # name -> row of the image that is in refs/ now (the highest attempt)
    for r in rows:
        a = int(r.get("attempt") or 0)
        attempts[r["name"]] = max(attempts.get(r["name"], 0), a + 1)
        if r["name"] not in last or a >= int(last[r["name"]].get("attempt") or 0):
            last[r["name"]] = r
    tries = collections.Counter((r["name"], r.get("source") or "") for r in rows)
    checks = {r["name"]: r for r in status.read_csv(f"{refs}/check_references.csv")}
    elsewhere = collections.defaultdict(list)  # name -> images of that name outside their group folder
    for f in glob.glob(f"{refs}/**/*.png", recursive=True):
        m = status.REF_RE.fullmatch(os.path.basename(f))
        if m and os.path.dirname(f) != layout.ref_dir(refs, int(m.group(1))):
            elsewhere[os.path.basename(f)[: -len(".png")]].append(f)

    if args.retry:
        idxs = retry_list(refs, args.tag)
        if wanted is not None:
            idxs = [i for i in idxs if i in set(wanted)]
    elif wanted is not None:
        idxs = wanted
    else:
        files = glob.glob(f"{run}/ref_sim/*/demo_*_ref_sim.png")
        idxs = [layout.demo_index(os.path.basename(f)) for f in files]
    idxs = sorted(set(idxs))

    pending = status.pending_replacements(run) if sources else set()
    claimed = status.cosmos_claims(run) if args.redo else {}
    jobs, gave_up, no_src = [], [], []
    for idx in idxs:
        name = f"{layout.demo_name(idx)}_{args.tag}"
        src = f"{layout.ref_sim_dir(run, idx)}/{layout.demo_name(idx)}_ref_sim.png"
        dst = f"{layout.ref_dir(refs, idx)}/{name}.png"
        source = sources.get(idx, "")
        if idx in pending:
            print(f"{name}: make_dataset.py --replace is not finished for this number, skipped", flush=True)
            continue
        if not os.path.isfile(src):
            print(f"{name}: no frame-0 image ({os.path.relpath(src, run)}), skipped", flush=True)
            no_src.append(name)
            continue
        if args.redo:
            if not os.path.isfile(dst):
                print(f"{name}: no image to make again (make it without --redo), skipped", flush=True)
                continue
            videos = glob.glob(f"{run}/cosmos/*/{layout.chunk(idx)}/{name}.mp4")
            if videos or idx in claimed:
                print(f"{name}: has a Cosmos video{' (being made)' if idx in claimed else ''}: reject it on the review "
                      "page with 'reference image problem' instead, skipped", flush=True)
                continue
        elif os.path.isfile(dst) and not args.retry:
            print(f"{name}: exists, skipped", flush=True)
            continue
        if elsewhere.get(name):
            print(f"{name}: in the wrong folder ({os.path.relpath(elsewhere[name][0], run)}), move it to "
                  f"{os.path.relpath(layout.ref_dir(refs, idx), run)}; not made", flush=True)
            continue
        if args.retry and not os.path.isfile(dst):
            print(f"{name}: no image to replace (moved or deleted), skipped", flush=True)
            continue
        if args.retry and not (name in checks and status.check_is_current(refs, checks[name])):
            print(f"{name}: not checked since it was made, check it first (checks/check_references.py), skipped",
                  flush=True)
            continue
        if args.retry and checks[name].get("passed") == "True":
            print(f"{name}: passed its check, skipped", flush=True)
            continue
        if tries[(name, source)] >= args.max_attempts:
            gave_up.append(idx)
            print(f"{name}: {tries[(name, source)]} images made already (--max_attempts {args.max_attempts}), "
                  "not made again", flush=True)
            continue
        attempt = attempts.get(name, 0)
        v = variations.draw(idx, args.seed, used, attempt=attempt, tag=args.tag)
        used.add(variations.key(v))
        jobs.append((idx, name, attempt, source, args.seed, src, dst, v,
                     prompt.replace("{variation}", variations.sentence(v))))
    if gave_up:
        here = os.path.relpath(os.path.dirname(os.path.abspath(__file__)))
        print(f"reached --max_attempts: demos {status.ranges(gave_up)}")
        if sources:
            for idx in gave_up:
                print(f'  python {here}/make_dataset.py {os.path.relpath(run)} --replace {idx} '
                      f'--reason "the reference image failed {args.max_attempts} times"')
    if not jobs:
        print("nothing to do")
        if not args.dry_run:
            finish(run)
        if no_src:
            sys.exit(1)
        return

    if args.dry_run:
        print("prompt:", prompt)
        for idx, name, attempt, source, seed, src, dst, v, text in jobs:
            print(f"{name} (attempt {attempt}, seed {seed}, {v['place_type']}): {variations.sentence(v)}")
        types = collections.Counter(j[7]["place_type"] for j in jobs)
        print("place types:", ", ".join(f"{t} {types.get(t, 0)}" for t in variations.PLACES))
        print(f"{len(jobs)} images, {args.model} {args.quality} {args.size}, no API call")
        return
    from openai import OpenAI

    client = OpenAI()
    count_lock = threading.Lock()
    results = {"ok": 0, "cost": 0.0, "failed": [], "no_credit": False}

    def work(job):
        idx, name, attempt, source, seed, src, dst, v, text = job
        if results["no_credit"]:
            with count_lock:
                results["failed"].append(name)
            return
        t0 = time.time()
        try:
            img, usage = edit_image(client, args.model, src, text, args.size, args.quality)
        except NoCredit as e:
            with count_lock:
                results["no_credit"] = True
                results["failed"].append(name)
            print(f"{name}: no credit left, stopping: {e}", flush=True)
            return
        except Exception as e:  # noqa: BLE001 - one bad image must not stop the batch
            with count_lock:
                results["failed"].append(name)
            print(f"{name}: FAILED {type(e).__name__}: {e}", flush=True)
            return
        ok, png = cv2.imencode(".png", cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA))
        if not ok:
            with count_lock:
                results["failed"].append(name)
            print(f"{name}: FAILED cannot encode the image", flush=True)
            return
        # Save under a temporary name first (no check or script picks it up), then record, then rename.
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = os.path.join(os.path.dirname(dst), f".{name}.png.tmp")
        with open(tmp, "wb") as f:
            f.write(png.tobytes())
        cost = cost_usd(usage)
        row = {"demo": idx, "name": name, "attempt": attempt, "source": source, "seed": seed, "model": args.model,
               "quality": args.quality, "size": args.size, **v, "prompt": text,
               "input_tokens": getattr(usage, "input_tokens", ""), "output_tokens": getattr(usage, "output_tokens", ""),
               "cost_usd": round(cost, 4), "seconds": round(time.time() - t0, 1),
               "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with status.lock(lock_dir):
            if os.path.isfile(dst):  # --retry or --redo: keep the replaced image outside refs/
                why = {**checks.get(name, {}), "reason": f"redo: {args.reason}"} if args.redo else checks.get(name, {})
                reject(run, rejected, idx, name, dst, last.get(name), why)
            status.append_csv(csv_path, row, COLUMNS)
            os.replace(tmp, dst)
        with count_lock:
            results["ok"] += 1
            results["cost"] += cost
        print(f"{name}: ok, ${cost:.3f}, {row['seconds']} s", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, jobs))
    print(f"{results['ok']}/{len(jobs)} images, ${results['cost']:.2f} -> {refs} (csv: {csv_path})")
    if results["no_credit"]:
        print("stopped: no credit left on the OpenAI account. Add credit, then run the same command again.")
    if results["failed"]:
        print(f"not made: {', '.join(results['failed'])}. Run the same command again to make them.")
    if no_src:
        print(f"no frame-0 image: {', '.join(no_src)}")
    finish(run)
    if results["failed"] or no_src:
        sys.exit(1)


def finish(run: str) -> None:
    """In a dataset folder: update dataset.csv and print the state."""
    if status.is_dataset(run):
        status.update(run)


if __name__ == "__main__":
    main()
