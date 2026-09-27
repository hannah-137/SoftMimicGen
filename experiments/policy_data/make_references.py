"""Make reference images from the frame-0 images with the OpenAI image edit API (gpt-image-2).

  export OPENAI_API_KEY=...
  python experiments/policy_data/make_references.py <run_dir> [--demos 2 7] [--seed 0] [--model gpt-image-2]
      [--quality medium] [--size 2048x1024] [--tag 01] [--workers 3] [--retry] [--dry_run]

Input: <run_dir>/ref_sim/000-049/demo_NNN_ref_sim.png (frame 0, room | wrist, 1024x512, made by make_videos.py).
Output: <run_dir>/refs/000-049/demo_NNN_<tag>.png (1024x512) and <run_dir>/refs/references.csv with one row per
image: the 7 axes and the place type, the prompt, the tokens and the cost.

Every image gets its own variation from variations.py (demo index, seed, tag, attempt): all 7 axes change, the
place types are exact in every group of 50 demos, and no two images in the csv share a combination. The prompt is
static on purpose: no action words (they make the model fold the towel), and it lists what must stay the same. The
API has no seed, so the same call twice gives two different images.

Size: the API needs at least 655,360 pixels, so the image is made at 2048x1024 and resized to 1024x512.
Flow: make_references.py -> checks/check_references.py -> make_references.py --retry -> check again -> run_cosmos.py
--retry reads refs/failed_references.txt. It makes the failed and the missing images with this --tag again, as a new
attempt: same place type, new combination. The replaced image goes to <run_dir>/refs_rejected/.
An image in the wrong folder is not made again, even when its demo counts as missing: move it.
Images that already exist are skipped (without --retry).
--dry_run prints the prompt and the variations and writes nothing. It needs no API key.
The run stops when the account has no credit left (error type insufficient_quota). Rate limits and server errors
are retried.
Needs: bash experiments/policy_data/setup.sh (openai), opencv-python, numpy. Cost per image is printed from the
usage of every call.
"""

import argparse
import base64
import collections
import csv
import glob
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import layout
import variations

OUT_W, OUT_H = 1024, 512
MIN_PIXELS = 655_360  # smallest image the API makes
PROMPT = (
    "Turn this simulator image into a realistic photo. It is a split screen: left the room camera, right the camera on "
    "the robot gripper. Keep the exact same camera viewpoint and framing in both halves, the same robot in the same "
    "pose, and the same towel in exactly the same position, shape, size and orientation. Do not add, remove, move, "
    "fold, bend or reshape anything. There is exactly one towel, flat on the table. {variation} The towel has the "
    "same color and texture on both sides. Change only the materials, textures, lighting and background. "
    "No text, no logos, no watermark."
)
# USD per 1M tokens, gpt-image-2 standard rate (2026-09). Only for the cost column; the bill is what OpenAI charges.
PRICE = {"text": 5.0, "image_in": 8.0, "image_out": 30.0}
COLUMNS = ["demo", "name", "attempt", "seed", "model", "quality", "size", "place_type", *variations.FIELDS, "prompt",
           "input_tokens", "output_tokens", "cost_usd", "seconds", "finished_at"]
RETRY_WAIT = [15, 30, 60, 120, 240]  # seconds between tries after a rate limit or a server error
# Error codes of a 429 that means "no credit left" (the error type is insufficient_quota). No retry for these.
NO_CREDIT_CODES = {"insufficient_quota", "credit_balance_exhausted"}


class NoCredit(Exception):
    """The account has no credit left. Every other call would fail too."""


def read_rows(path: str) -> list:
    return list(csv.DictReader(open(path))) if os.path.isfile(path) else []


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
        sys.exit(f"{path} not found: run checks/check_references.py first")
    lines = [(n, r.strip()) for n, _, r in (line.strip().partition(":") for line in open(path)) if n]
    # A demo whose image is only in the wrong folder is also "missing". Moving the file fixes both, so no API call.
    moved = {layout.demo_index(n) for n, r in lines if r.startswith("wrong folder")}
    idxs = []
    for name, reason in lines:
        if reason.startswith("wrong folder"):
            print(f"{name}: {reason} (not made again)", flush=True)
        elif reason == "missing":  # the name is demo_NNN, no image yet
            if layout.demo_index(name) not in moved:
                idxs.append(layout.demo_index(name))
        elif name.endswith(f"_{tag}"):
            idxs.append(layout.demo_index(name))
        else:
            print(f"{name}: another tag than --tag {tag}, skipped", flush=True)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--demos", type=int, nargs="*", default=None,
                    help="demo indices (default: every ref_sim image; with --retry: only these of the failed ones)")
    ap.add_argument("--seed", type=int, default=0, help="variation seed (default 0)")
    ap.add_argument("--model", default="gpt-image-2")
    ap.add_argument("--quality", default="medium", help="low, medium, high (auto = the API decides)")
    ap.add_argument("--size", default="2048x1024", help="size the API makes; the file is resized to 1024x512")
    ap.add_argument("--tag", default="01", help="reference number in the file name: demo_NNN_<tag>.png")
    ap.add_argument("--workers", type=int, default=3, help="parallel API calls")
    ap.add_argument("--retry", action="store_true", help="make the images in refs/failed_references.txt again")
    ap.add_argument("--dry_run", action="store_true", help="print the prompt and the variations, call nothing")
    args = ap.parse_args()
    check_size(args.size)

    run = os.path.abspath(args.run_dir)
    refs = f"{run}/refs"
    rejected = f"{run}/refs_rejected"
    csv_path = f"{refs}/references.csv"
    rows = read_rows(csv_path)
    used = {tuple(r[f] for f in variations.FIELDS) for r in rows}
    attempts = {}  # name -> number of the next attempt
    for r in rows:
        attempts[r["name"]] = max(attempts.get(r["name"], 0), int(r.get("attempt") or 0) + 1)

    if args.retry:
        idxs = retry_list(refs, args.tag)
        if args.demos is not None:
            idxs = [i for i in idxs if i in set(args.demos)]
    elif args.demos is not None:
        idxs = args.demos
    else:
        files = glob.glob(f"{run}/ref_sim/*/demo_*_ref_sim.png")
        idxs = [layout.demo_index(os.path.basename(f)) for f in files]
    idxs = sorted(set(idxs))

    jobs = []
    for idx in idxs:
        name = f"{layout.demo_name(idx)}_{args.tag}"
        src = f"{layout.ref_sim_dir(run, idx)}/{layout.demo_name(idx)}_ref_sim.png"
        dst = f"{layout.ref_dir(refs, idx)}/{name}.png"
        if not os.path.isfile(src):
            sys.exit(f"missing: {src}")
        if os.path.isfile(dst) and not args.retry:
            print(f"{name}: exists, skipped", flush=True)
            continue
        attempt = attempts.get(name, 0)
        v = variations.draw(idx, args.seed, used, attempt=attempt, tag=args.tag)
        used.add(variations.key(v))
        jobs.append((idx, name, attempt, args.seed, src, dst, v, PROMPT.format(variation=variations.sentence(v))))
    if not jobs:
        print("nothing to do")
        return

    if args.dry_run:
        print("prompt:", PROMPT.format(variation="{variation}"))
        for idx, name, attempt, seed, src, dst, v, prompt in jobs:
            print(f"{name} (attempt {attempt}, seed {seed}, {v['place_type']}): {variations.sentence(v)}")
        types = collections.Counter(j[6]["place_type"] for j in jobs)
        print("place types:", ", ".join(f"{t} {types.get(t, 0)}" for t in variations.PLACES))
        print(f"{len(jobs)} images, {args.model} {args.quality} {args.size}, no API call")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai")
    client = OpenAI()
    lock = threading.Lock()
    results = {"ok": 0, "cost": 0.0, "failed": [], "no_credit": False}

    def work(job):
        idx, name, attempt, seed, src, dst, v, prompt = job
        if results["no_credit"]:
            with lock:
                results["failed"].append(name)
            return
        t0 = time.time()
        try:
            img, usage = edit_image(client, args.model, src, prompt, args.size, args.quality)
        except NoCredit as e:
            with lock:
                results["no_credit"] = True
                results["failed"].append(name)
            print(f"{name}: no credit left, stopping: {e}", flush=True)
            return
        except Exception as e:  # noqa: BLE001 - one bad image must not stop the batch
            with lock:
                results["failed"].append(name)
            print(f"{name}: FAILED {type(e).__name__}: {e}", flush=True)
            return
        if os.path.isfile(dst):  # --retry: keep the replaced image outside refs/
            os.makedirs(rejected, exist_ok=True)
            old = f"attempt{attempt - 1}" if attempt > 0 else "original"
            os.replace(dst, f"{rejected}/{name}_{old}.png")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cv2.imwrite(dst, cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA))
        cost = cost_usd(usage)
        row = {"demo": idx, "name": name, "attempt": attempt, "seed": seed, "model": args.model,
               "quality": args.quality, "size": args.size, **v, "prompt": prompt,
               "input_tokens": getattr(usage, "input_tokens", ""), "output_tokens": getattr(usage, "output_tokens", ""),
               "cost_usd": round(cost, 4), "seconds": round(time.time() - t0, 1),
               "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with lock:
            new = not os.path.isfile(csv_path)
            os.makedirs(refs, exist_ok=True)
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLUMNS)
                if new:
                    w.writeheader()
                w.writerow(row)
            results["ok"] += 1
            results["cost"] += cost
        print(f"{name}: ok, ${cost:.3f}, {row['seconds']} s", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, jobs))
    print(f"{results['ok']}/{len(jobs)} images, ${results['cost']:.2f} -> {refs} (csv: {csv_path})")
    if results["no_credit"]:
        print("stopped: no credit left on the OpenAI account")
    if results["failed"]:
        print(f"not made: {', '.join(results['failed'])}")
        sys.exit(1)


if __name__ == "__main__":
    main()
