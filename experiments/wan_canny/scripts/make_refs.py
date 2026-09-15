"""Stage 4a of the Wan pipeline: photorealistic reference images from the simulator frame with Qwen-Image-Edit-2511
through the ComfyUI API (models: docker/setup_qwen_image_edit.sh).

  python experiments/wan_canny/scripts/make_refs.py <run_dir> --n_ref 10 [--empty] [--seed 0] [--steps 20] [--cfg 4] [--lightning]
      [--prompt TEXT] [--negative TEXT] [--gen_scale 2] [--denoise 1.0] [--force] [--dry_run] [--prefix P] [--task T]
      [--comfy_url URL]

images/<prefix>_ref_sim.png (stage 2, frame 0)  ->  images/<prefix>_ref_ai_01.png ... _<N>.png, one .json each.
Layout check (2026-09-14): every image is scored with quality.check_ref against the simulator (robot and object
outlines, or the static scene for --empty). A failing image moves to images/rejected/ with its score and the slot is
retried with new seeds (--retries), then in weak variation mode; if nothing passes, <name>.fallback.json is written and
make_wan.py makes a prompt-only video (_pNN) for that slot instead, so the count stays and no bad image is used.
--empty adds images/<prefix>_ref_empty.png: the same scene with the manipulated object removed (tasks.REF_EMPTY_PROMPT),
the reference for make_wan.py --n_obj (object colour/material from the prompt, robot/table/background anchored).
N = --n_ref, image k uses seed --seed + k - 1, so the N images are variations of the same instruction. make_wan.py
then makes one Wan video per (reference image, control). Existing images are skipped unless --force; images the
user put there (<prefix>_ref_ai.png or any other suffix) are never touched, both kinds are used by make_wan.py.

Prompt: tasks.REF_PROMPT, deliberately static (no task action words: "folds a towel" made the model fold the towel)
and without colour words; image 1 is faithful to the frame, image k>1 gets tasks.ref_variation(k, seed): one entry
per axis (environment, table, robot condition, lighting, camera look) plus a colour and material for the task's
manipulated object, drawn from the image's seed, so the N images differ in environment and object appearance
while the layout stays fixed and a run is reproducible.
--prompt overrides the whole instruction for every image.
The edit runs at --gen_scale x the frame size (512x512 -> 1024x1024, the humanoid 512x320 -> 1024x640; Qwen-Image
works around 1 MP) and the result is scaled back to the frame size (cv2 INTER_AREA), so the reference has exactly
the video's size and aspect. Defaults follow the ComfyUI "Qwen-Image-Edit 2511" template: 20 steps, cfg 4,
shift 3.1, euler/simple, CFGNorm. --lightning adds the lightx2v 4-step LoRA (4 steps, cfg 1, several times faster,
a little less detail). Template: workflows/qwen_image_edit_2511_api.json (nodes found by class_type).
ComfyUI: same server and GPU as make_wan.py (auto-started on GPU 1 if nothing answers)."""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import cv2

from make_wan import WC_DIR, _one, download, ensure_server, output_files, png_size, submit, upload, wait_done
from tasks import REF_AXES, REF_EMPTY_PROMPT, REF_NEGATIVE, REF_PROMPT, SUBDIRS, TASKS, ref_variation, task_of

TEMPLATE = os.path.join(WC_DIR, "workflows", "qwen_image_edit_2511_api.json")


def gen_size(width: int, height: int, scale: float) -> tuple[int, int]:
    """Edit resolution: the frame size times scale, rounded to multiples of 16 (Qwen-Image latent patch)."""
    return max(16, round(width * scale / 16) * 16), max(16, round(height * scale / 16) * 16)


def build_graph(template: dict, ref_sim_file: str, prompt: str, negative: str, seed: int, steps: int, cfg: float,
                shift: float, width: int, height: int, lightning: bool, out_prefix: str, denoise: float = 1.0) -> dict:
    g = json.loads(json.dumps(template))  # deep copy
    g[_one(g, "LoadImage")]["inputs"]["image"] = ref_sim_file
    g[_one(g, "ImageScale")]["inputs"].update(width=width, height=height)
    g[_one(g, "ModelSamplingAuraFlow")]["inputs"]["shift"] = shift
    ks = _one(g, "KSampler")
    g[ks]["inputs"].update(seed=seed, steps=steps, cfg=cfg, denoise=denoise)
    pos, neg = g[ks]["inputs"]["positive"][0], g[ks]["inputs"]["negative"][0]
    for cond, text in ((pos, prompt), (neg, negative)):
        enc = g[cond]["inputs"]["conditioning"][0] if g[cond]["class_type"] != "TextEncodeQwenImageEditPlus" else cond
        g[enc]["inputs"]["prompt"] = text
    lora = _one(g, "LoraLoaderModelOnly")
    if not lightning:  # bypass the LoRA: the sampler takes the LoRA node's input model
        g[ks]["inputs"]["model"] = g[lora]["inputs"]["model"]
        del g[lora]
    g[_one(g, "SaveImage")]["inputs"]["filename_prefix"] = out_prefix
    return g


def run(run_dir: str, n_ref: int = 1, seed: int = 0, steps: int | None = None, cfg: float | None = None,
        shift: float = 3.1, lightning: bool = False, prompt: str | None = None, negative: str = REF_NEGATIVE,
        empty: bool = False, gen_scale: float = 2.0, denoise: float = 1.0, force: bool = False, dry_run: bool = False, prefix: str | None = None,
        task: str | None = None, comfy_url: str = "http://127.0.0.1:8188", comfy_dir: str = "/workspace/tools/ComfyUI",
        comfy_python: str = "/opt/miniconda3/envs/comfyui/bin/python", comfy_gpu: int | None = 1,
        comfy_log: str = "logs/comfyui.log", template: str = TEMPLATE, timeout_s: float = 900,
        check: bool = True, retries: int = 3) -> dict:
    run_dir = os.path.normpath(run_dir)
    prefix = prefix or os.path.basename(run_dir)
    task = task or task_of(prefix)
    steps = steps if steps is not None else (4 if lightning else 20)
    cfg = cfg if cfg is not None else (1.0 if lightning else 4.0)
    d_img = os.path.join(run_dir, SUBDIRS["images"])
    ref_sim = os.path.join(d_img, f"{prefix}_ref_sim.png")
    if not os.path.isfile(ref_sim):
        raise FileNotFoundError(f"simulator frame not found: {ref_sim} (run make_source.py first)")
    width, height = png_size(ref_sim)
    gw, gh = gen_size(width, height, gen_scale)
    with open(template) as f:
        tpl = json.load(f)

    def prompt_of(k: int) -> str:
        return prompt if prompt is not None else REF_PROMPT.format(object=TASKS[task]["object"], variation=ref_variation(k, seed, task))

    variation_mode = TASKS[task].get("variation", "full")

    def attempts_for(label: str, k: int):
        """[(seed, prompt, mode)] tried in order until one passes the layout check: the task's own mode with 3 seeds,
        then (for varied images of a 'full' task) weak mode with 3 seeds. 01 is faithful, 'empty' removes the object."""
        seeds = [seed + k - 1 + 1000 * a for a in range(retries)]
        if label == "empty":
            return [(sd, REF_EMPTY_PROMPT.format(object=TASKS[task]["object"]), "empty") for sd in seeds]
        if prompt is not None or k == 1:
            return [(sd, prompt if prompt is not None else prompt_of(1), "faithful") for sd in seeds]
        modes = ["full", "weak"] if variation_mode == "full" else ["weak"]
        return [(sd, REF_PROMPT.format(object=TASKS[task]["object"], variation=ref_variation(k, sd, task, m)), m)
                for m in modes for sd in seeds]

    slots = [(f"{k:02d}", k, os.path.join(d_img, f"{prefix}_ref_ai_{k:02d}.png")) for k in range(1, n_ref + 1)]
    if empty:
        slots.append(("empty", 1, os.path.join(d_img, f"{prefix}_ref_empty.png")))
    todo = []
    for lb, k, dst in slots:
        marker = os.path.splitext(dst)[0] + ".fallback.json"
        if (os.path.isfile(dst) or os.path.isfile(marker)) and not force:
            print(f"refs: {os.path.basename(dst)} exists{' (fallback)' if os.path.isfile(marker) else ''}, skipped (--force to redo)")
        else:
            todo.append((lb, k, dst))
    print(f"refs: task={task} prefix={prefix} n_ref={n_ref} empty={empty} todo={[lb for lb, _, _ in todo]} seeds from {seed} "
          f"steps={steps} cfg={cfg} shift={shift} denoise={denoise} lightning={lightning} edit at {gw}x{gh} -> {width}x{height} "
          f"check={'on' if check else 'off'} retries={retries} variation={variation_mode}", flush=True)
    if dry_run:
        for lb, k, dst in todo:
            for sd, pr, mode in attempts_for(lb, k):
                print(f"--- {lb} seed {sd} mode {mode}: {dst}\n    {pr}")
        return {}
    if not todo:
        return {}

    if check:
        import quality  # needs h5py/cv2 and the run's hdf5 (simulator edges and motion mask)
    ensure_server(comfy_url, comfy_dir, comfy_python, comfy_gpu, comfy_log)
    ref_sim_name = upload(comfy_url, ref_sim)
    d_rej = os.path.join(d_img, "rejected")

    def generate(lb: str, sd: int, pr: str, dst: str) -> dict | None:
        t0 = time.time()
        g = build_graph(tpl, ref_sim_name, pr, negative, sd, steps, cfg, shift, gw, gh, lightning, f"refs/{prefix}_ref_{lb}", denoise)
        pid = submit(comfy_url, g)
        ent = wait_done(comfy_url, pid, timeout_s)
        status = ent.get("status", {})
        if status.get("status_str") == "error":
            errs = [m for m in status.get("messages", []) if m and m[0] in ("execution_error", "execution_interrupted")]
            print(f"refs: {lb}: ComfyUI error: " + ("; ".join(f"{m[0]} {m[1].get('node_type')}: {m[1].get('exception_message', '')}" for m in errs) or "unknown")[:500], flush=True)
            return None
        files = output_files(ent)
        if not files:
            return None
        tmp = os.path.join(d_img, f".{prefix}_ref_{lb}.gen.png")  # full-resolution edit, kept only until resized
        download(comfy_url, files[0], tmp)
        img = cv2.imread(tmp, cv2.IMREAD_COLOR)
        os.remove(tmp)
        if img is None:
            return None
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst, img)
        return dict(task=task, prefix=prefix, index=lb, source=os.path.basename(ref_sim), model=tpl[_one(tpl, "UNETLoader")]["inputs"]["unet_name"],
                    lightning=lightning, prompt=pr, negative=negative, seed=sd, steps=steps, cfg=cfg, shift=shift, denoise=denoise,
                    edit_width=gw, edit_height=gh, width=width, height=height, template=os.path.relpath(template),
                    comfy_url=comfy_url, prompt_id=pid, comfy_output=files[0], seconds=round(time.time() - t0, 1),
                    finished_at=datetime.now().isoformat(timespec="seconds"))

    results = {}
    for lb, k, dst in todo:
        tried = []
        for n, (sd, pr, mode) in enumerate(attempts_for(lb, k), 1):
            meta = generate(lb, sd, pr, dst)
            if meta is None:
                tried.append(dict(attempt=n, seed=sd, mode=mode, error="generation failed"))
                continue
            res = quality.check_ref(run_dir, dst, "empty" if lb == "empty" else "var") if check else dict(passed=True, score=None)
            meta.update(mode=mode, attempt=n, check=res, rejected_before=tried)
            if res["passed"]:
                with open(os.path.splitext(dst)[0] + ".json", "w") as f:
                    json.dump(meta, f, indent=1, ensure_ascii=False)
                print(f"refs: {lb}: {os.path.basename(dst)} PASS score {res['score']} (attempt {n}, {mode}, seed {sd}, {meta['seconds']:.0f}s)", flush=True)
                results[lb] = dst
                break
            os.makedirs(d_rej, exist_ok=True)
            base = os.path.join(d_rej, f"{os.path.splitext(os.path.basename(dst))[0]}_try{n}")
            os.replace(dst, base + ".png")
            with open(base + ".json", "w") as f:
                json.dump(meta, f, indent=1, ensure_ascii=False)
            tried.append(dict(attempt=n, seed=sd, mode=mode, score=res["score"], file=os.path.relpath(base + ".png", run_dir)))
            print(f"refs: {lb}: attempt {n} ({mode}, seed {sd}) FAIL score {res['score']} < {res['threshold']}, moved to rejected/", flush=True)
        else:
            # no attempt passed: the slot becomes a prompt-only Wan variation (make_wan.py adds one _pNN per marker)
            marker = os.path.splitext(dst)[0] + ".fallback.json"
            with open(marker, "w") as f:
                json.dump(dict(task=task, prefix=prefix, index=lb, fallback="prompt_only", attempts=tried,
                               finished_at=datetime.now().isoformat(timespec="seconds")), f, indent=1, ensure_ascii=False)
            print(f"refs: {lb}: no attempt passed ({len(tried)} tried) -> fallback to a prompt-only Wan variation ({os.path.basename(marker)})", flush=True)
            results[lb] = marker
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--n_ref", type=int, default=1, help="number of reference images (_01 .. _NN); 0 = none")
    p.add_argument("--empty", action="store_true", help="also make <prefix>_ref_empty.png: the scene without the manipulated object (for make_wan.py --n_obj)")
    p.add_argument("--seed", type=int, default=0, help="seed of image 01; image k uses seed + k - 1")
    p.add_argument("--steps", type=int, default=None, help="default 20 (4 with --lightning)")
    p.add_argument("--cfg", type=float, default=None, help="default 4.0 (1.0 with --lightning)")
    p.add_argument("--shift", type=float, default=3.1)
    p.add_argument("--lightning", action="store_true", help="lightx2v 4-step Lightning LoRA (fast, a little less detail)")
    p.add_argument("--prompt", default=None, help="edit instruction (default tasks.REF_PROMPT with the task scene)")
    p.add_argument("--negative", default=REF_NEGATIVE)
    p.add_argument("--gen_scale", type=float, default=2.0, help="edit resolution = frame size x this (then scaled back)")
    p.add_argument("--denoise", type=float, default=1.0, help="KSampler denoise; < 1 keeps more of the frame's own structure (img2img style)")
    p.add_argument("--force", action="store_true", help="regenerate existing _ref_ai_NN.png")
    p.add_argument("--dry_run", action="store_true", help="print the graphs, do not contact ComfyUI")
    p.add_argument("--prefix", default=None, help="file prefix (default: run_dir folder name)")
    p.add_argument("--task", default=None, help="task key in tasks.TASKS (default: derived from the prefix)")
    p.add_argument("--comfy_url", default="http://127.0.0.1:8188")
    p.add_argument("--comfy_dir", default="/workspace/tools/ComfyUI")
    p.add_argument("--comfy_python", default="/opt/miniconda3/envs/comfyui/bin/python")
    p.add_argument("--comfy_gpu", type=int, default=1, help="GPU for an auto-started ComfyUI (-1: let ComfyUI choose)")
    p.add_argument("--comfy_log", default="logs/comfyui.log")
    p.add_argument("--template", default=TEMPLATE)
    p.add_argument("--timeout", type=float, default=900, help="seconds to wait per image")
    p.add_argument("--no_check", action="store_true", help="skip the layout check (quality.py) and keep every image")
    p.add_argument("--retries", type=int, default=3, help="seeds tried per mode before falling back")
    a = p.parse_args()
    if a.comfy_gpu != 1:
        p.error("this project may only use GPU 1 (GPU 0 belongs to other users, docker/gpu.sh)")
    out = run(a.run_dir, a.n_ref, a.seed, a.steps, a.cfg, a.shift, a.lightning, a.prompt, a.negative, a.empty, a.gen_scale, a.denoise, a.force,
              a.dry_run, a.prefix, a.task, a.comfy_url, a.comfy_dir, a.comfy_python, None if a.comfy_gpu < 0 else a.comfy_gpu,
              a.comfy_log, a.template, a.timeout, not a.no_check, a.retries)
    if out and any(v is None for v in out.values()):
        sys.exit(1)
