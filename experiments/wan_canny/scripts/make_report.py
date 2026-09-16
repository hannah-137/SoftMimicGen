"""Notion report for one run dir: the same page layout as "Franka towel v2 - reference and prompt test" (2026-09-14),
built from the files and .json records in the run dir, uploaded through the Notion API. No LLM in the loop.

  python experiments/wan_canny/scripts/make_report.py <run_dir> [--parent PAGE_ID] [--title TEXT] [--icon EMOJI]
      [--controls a,b] [--dry_run]

Token: ~/.config/notion_token (an internal Notion integration, "access token" type, with the parent page shared to it).
Parent: the CRAFT page by default.

Page (easy English, technical terms kept):
  intro, how it works, table of the kinds found in wans/, inputs (source video + one control video per channel),
  reference images with their image prompts, results (one section per kind: Wan prompt + one video per channel),
Kinds come from wans/<prefix>_wan_<control><suffix>.json: mode "ref" (reference image, suffix "" or _NN), "obj" (empty
reference + object prompt, _oNN), "var" (no reference, prompt only, _pNN).
Uploads go section by section and are attached right away (an unattached Notion upload expires after about an hour).
"""
import argparse
import glob
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

from tasks import CONTROLS, SUBDIRS

API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"
CRAFT_PAGE = "3d0fdfd486a5802faea4eaad0827628b"
TOKEN_FILE = os.path.expanduser("~/.config/notion_token")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # json paths are repo-relative


# ----------------------------------------------------------------------------------------------------- Notion API
class Notion:
    def __init__(self, token: str, dry_run: bool = False):
        self.token, self.dry_run, self.n_upload = token, dry_run, 0

    def _req(self, method: str, path: str, body=None, data: bytes | None = None, ctype: str | None = None):
        headers = {"Authorization": f"Bearer {self.token}", "Notion-Version": NOTION_VERSION}
        if body is not None:
            data, ctype = json.dumps(body).encode(), "application/json"
        if ctype:
            headers["Content-Type"] = ctype
        for attempt in range(5):
            req = urllib.request.Request(API + path, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")
                if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"notion {method} {path} -> HTTP {e.code}: {msg[:800]}") from None

    def upload(self, path: str) -> str:
        """Single-part file upload (files here are < 20 MiB). Returns the file upload id."""
        name = os.path.basename(path)
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        self.n_upload += 1
        if self.dry_run:
            return f"dry-{self.n_upload}-{name}"
        fu = self._req("POST", "/file_uploads", {"filename": name, "content_type": ctype})
        boundary = uuid.uuid4().hex
        with open(path, "rb") as f:
            payload = f.read()
        data = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
                f"Content-Type: {ctype}\r\n\r\n").encode() + payload + f"\r\n--{boundary}--\r\n".encode()
        r = self._req("POST", f"/file_uploads/{fu['id']}/send", data=data, ctype=f"multipart/form-data; boundary={boundary}")
        if r.get("status") != "uploaded":
            raise RuntimeError(f"upload of {path} ended with status {r.get('status')}")
        return fu["id"]

    def create_page(self, parent: str, title: str, icon: str, children: list) -> str:
        if self.dry_run:
            print(f"[dry] create page '{title}' under {parent} with {len(children)} blocks")
            return "dry-page"
        body = {"parent": {"page_id": parent}, "icon": {"type": "emoji", "emoji": icon},
                "properties": {"title": {"title": rt(title)}}, "children": children[:100]}
        page = self._req("POST", "/pages", body)
        if children[100:]:
            self.append(page["id"], children[100:])
        return page["id"]

    def append(self, block_id: str, children: list):
        if self.dry_run:
            print(f"[dry] append {len(children)} blocks: " + ", ".join(b["type"] for b in children))
            return
        for i in range(0, len(children), 100):
            self._req("PATCH", f"/blocks/{block_id}/children", {"children": children[i:i + 100]})


# ----------------------------------------------------------------------------------------------------- blocks
def rt(text: str, bold: bool = False) -> list:
    """Rich text, split into <= 2000 character pieces (Notion limit)."""
    return [{"type": "text", "text": {"content": text[i:i + 2000]}, "annotations": {"bold": bold}}
            for i in range(0, max(len(text), 1), 2000)]


def para(*parts) -> dict:
    """para("plain", ("bold", True), ...)"""
    items = []
    for p in parts:
        items += rt(p[0], True) if isinstance(p, tuple) else rt(p)
    return {"type": "paragraph", "paragraph": {"rich_text": items}}


def heading(level: int, text: str) -> dict:
    return {"type": f"heading_{level}", f"heading_{level}": {"rich_text": rt(text)}}


def bullet(*parts) -> dict:
    b = para(*parts)
    return {"type": "bulleted_list_item", "bulleted_list_item": b["paragraph"]}


def quote(text: str) -> dict:
    return {"type": "quote", "quote": {"rich_text": rt(text)}}


def callout(text: str, emoji: str) -> dict:
    return {"type": "callout", "callout": {"rich_text": rt(text), "icon": {"type": "emoji", "emoji": emoji}}}


def media(kind: str, upload_id: str, caption: str) -> dict:
    return {"type": kind, kind: {"type": "file_upload", "file_upload": {"id": upload_id}, "caption": rt(caption)}}


def columns(blocks: list) -> dict:
    if len(blocks) == 1:  # Notion rejects a column_list with a single column
        return blocks[0]
    return {"type": "column_list", "column_list": {"children": [
        {"type": "column", "column": {"children": [b]}} for b in blocks]}}


def table(rows: list[list[str]]) -> dict:
    return {"type": "table", "table": {"table_width": len(rows[0]), "has_column_header": True, "has_row_header": False,
                                       "children": [{"type": "table_row", "table_row": {"cells": [rt(c) for c in r]}}
                                                    for r in rows]}}


# ----------------------------------------------------------------------------------------------------- run dir
def prompt_text(rec: dict) -> str:
    """Image prompt and negative exactly as sent to Qwen-Image-Edit (from the make_refs.py json), nothing cut."""
    return f"Image prompt: {rec.get('prompt', '')}" + (f" | Negative: {rec['negative']}" if rec.get("negative") else "")


def load_kinds(run_dir: str, prefix: str, controls_filter):
    """[(suffix, mode, ref_image, prompt, {control: video path})] in report order, plus one result json for settings."""
    kinds, sample = {}, None
    for j in sorted(glob.glob(os.path.join(run_dir, SUBDIRS["wans"], f"{prefix}_wan_*.json"))):
        m = json.load(open(j))
        if controls_filter and m["control"] not in controls_filter:
            continue
        mp4 = j[:-5] + ".mp4"
        if not os.path.isfile(mp4):
            continue
        sample = sample or m
        ref = m.get("ref_image")
        ref = ref if not ref or os.path.isabs(ref) else os.path.join(REPO, ref)
        k = kinds.setdefault(m.get("ref_suffix", ""), {"mode": m.get("mode", "ref"), "ref": ref, "prompt": m["prompt"],
                                                        "negative": m.get("negative", ""), "videos": {}, "checks": {}})
        k["videos"][m["control"]] = mp4
        k["checks"][m["control"]] = (m.get("check") or {}).get("score"), m.get("attempt", 1)
    order = {"ref": 0, "obj": 1, "var": 2}
    items = sorted(kinds.items(), key=lambda kv: (order.get(kv[1]["mode"], 9), kv[0]))
    return items, sample


def ref_record(path: str | None) -> dict | None:
    """The make_refs.py json next to a reference image, if any."""
    if not path:
        return None
    j = os.path.splitext(path)[0] + ".json"
    return json.load(open(j)) if os.path.isfile(j) else None


def label(suffix: str) -> str:
    return suffix.lstrip("_") or "ref"


# ----------------------------------------------------------------------------------------------------- report
def build(run_dir: str, notion: Notion, parent: str, title: str | None, icon: str, controls_filter):
    run_dir = os.path.normpath(run_dir)
    prefix = os.path.basename(run_dir)
    kinds, sample = load_kinds(run_dir, prefix, controls_filter)
    if not kinds:
        raise SystemExit(f"no Wan results with json in {run_dir}/{SUBDIRS['wans']}/")
    task, tag = sample["task"], prefix[len(sample["task"]) + 1:]
    used = [c for c in CONTROLS if any(c in k["videos"] for _, k in kinds)]
    title = title or f"{task.replace('_', ' ').capitalize()} {tag} - reference and prompt test"
    n_videos = sum(len(k["videos"]) for _, k in kinds)
    modes = {m: [s for s, k in kinds if k["mode"] == m] for m in ("ref", "obj", "var")}
    names = lambda sfx: ", ".join(label(s) for s in sfx)  # noqa: E731

    # --- intro (created with the page, no uploads)
    rows = [["Kind", "Reference image", "What the prompt says", "Why we try it"]]
    user_refs = [s for s in modes["ref"] if not ref_record(dict(kinds)[s]["ref"])]
    ai_refs = [s for s in modes["ref"] if ref_record(dict(kinds)[s]["ref"])]
    if user_refs:
        rows.append([names(user_refs), "user image", "plain, no colour", "base line"])
    if ai_refs:
        rows.append([names(ai_refs), "AI edit (Qwen-Image-Edit)", "plain, no colour", "the image makes the new look"])
    if modes["obj"]:
        rows.append([names(modes["obj"]), "empty scene (object removed)", "object colour and material",
                     "keep the room, change the object"])
    if modes["var"]:
        rows.append([names(modes["var"]), "none", "object, room, table, light", "the prompt makes the new look"])
    intro = [
        callout(f"We made the same {task} video {n_videos} times. The motion is always the same. Only the look changes. "
                f"Task: {task}. Tag: {tag}. Date: {sample['finished_at'][:10]}.", icon),
        heading(2, "How it works"),
        bullet(("The control video", True), " sets the motion. Wan follows it frame by frame."),
        bullet(("The reference image", True), " sets the look. The robot, the table and the room come from this image."),
        bullet(("The prompt", True), " sets what the reference image does not show."),
        heading(2, f"The {len(kinds)} kinds"),
        table(rows),
        para(f"{len(kinds)} kinds x {len(used)} control videos = {n_videos} videos."),
    ]
    page = notion.create_page(parent, title, icon, intro)

    # --- inputs
    src = os.path.join(run_dir, SUBDIRS["sources"], f"{prefix}_source.mp4")
    blocks = [heading(2, "Inputs"), para(f"All {n_videos} videos use these. We show them one time only.")]
    if os.path.isfile(src):
        blocks.append(media("video", notion.upload(src),
                            f"source video: the simulator render. {sample['frames']} frames, {sample['fps']} fps, "
                            f"{sample['width']}x{sample['height']}."))
    ctrl = [media("video", notion.upload(os.path.join(run_dir, SUBDIRS["edges"], f"{prefix}_{c}.mp4")),
                  f"control {i}: {c}") for i, c in enumerate(used, 1)
            if os.path.isfile(os.path.join(run_dir, SUBDIRS["edges"], f"{prefix}_{c}.mp4"))]
    if ctrl:
        blocks.append(columns(ctrl))
    notion.append(page, blocks)

    # --- reference images (unique files, with their image prompts)
    refs = []
    for s, k in kinds:
        if k["ref"] and k["ref"] not in [r for r, _ in refs]:
            refs.append((k["ref"], s))
    if refs:
        blocks, imgs = [heading(2, "Reference images"),
                        para("Each caption has the full image prompt and negative, exactly as sent to Qwen-Image-Edit.")], []
        for path, s in refs:
            if not os.path.isfile(path):
                continue
            rec = ref_record(path)
            if rec is None:
                cap = f"{label(s)}: user image. No AI record."
            elif kinds and dict(kinds)[s]["mode"] == "obj":
                cap = f"empty (used by {names(modes['obj'])}) | {prompt_text(rec)}"
            else:
                cap = f"{label(s)} | {prompt_text(rec)}"
            imgs.append(media("image", notion.upload(path), cap))
        for i in range(0, len(imgs), 3):
            blocks.append(columns(imgs[i:i + 3]))
        notion.append(page, blocks)

    # --- results: one section per kind
    notion.append(page, [para(("Wan negative prompt (the same for every video):", True)), quote(sample.get("negative", "")),
                         heading(2, "Results"),
                         para(f"Each row is one kind. The {len(used)} videos in a row use the {len(used)} control videos.")])
    for s, k in kinds:
        ref = k["ref"]
        rec = ref_record(ref)
        if k["mode"] == "var":
            head, intro_text = f"{label(s)} - no reference image", "No reference image. The prompt makes the whole look."
        elif k["mode"] == "obj":
            head, intro_text = f"{label(s)} - empty reference", "Reference image: empty scene. The prompt names the object."
        elif rec is None:
            head, intro_text = f"{label(s)} - user reference image", f"Reference image: {os.path.basename(ref)}."
        else:
            head, intro_text = f"{label(s)} - AI-edited reference", f"Reference image: {label(s)} (its image prompt is under Reference images)."
        def vcap(c):
            _, attempt = k["checks"].get(c, (None, 1))
            return c + (f" | attempt {attempt}" if attempt and attempt > 1 else "")
        vids = [media("video", notion.upload(k["videos"][c]), vcap(c)) for c in used if c in k["videos"]]
        notion.append(page, [heading(3, head), para(intro_text), para(("Wan prompt:", True)), quote(k["prompt"]), columns(vids)])

    notion.append(page, quality_blocks(run_dir, prefix, n_videos) + [settings_table(run_dir, prefix, sample)])
    return page, notion.n_upload


def settings_table(run_dir: str, prefix: str, sample: dict) -> dict:
    """Wan settings shared by every video of the run (from the result json files)."""
    secs = [json.load(open(p))["seconds"] for p in glob.glob(os.path.join(run_dir, SUBDIRS["wans"], f"{prefix}_wan_*.json"))]
    return table([["Setting", "Value"],
                  ["Video size", f"{sample['width']} x {sample['height']}, {sample['frames']} frames, {sample['fps']} fps"],
                  ["Steps / cfg / shift", f"{sample['steps']} / {sample['cfg']} / {sample['shift']}"],
                  ["Seed", f"{sample['seed']} for every video"],
                  ["Time", f"{sum(secs) / max(len(secs), 1):.0f} s per video on average"],
                  ["Template", sample["template"]]])


def quality_blocks(run_dir: str, prefix: str, n_videos: int) -> list:
    """'Quality check' section: what the check looks at and how many items it rejected (no scores)."""
    d_img, d_wan = os.path.join(run_dir, SUBDIRS["images"]), os.path.join(run_dir, SUBDIRS["wans"])
    rej_img = len(glob.glob(os.path.join(d_img, "rejected", "*.png")))
    rej_vid = len(glob.glob(os.path.join(d_wan, "rejected", "*.mp4")))
    fallbacks = len(glob.glob(os.path.join(d_img, f"{prefix}_ref_*.fallback.json")))
    return [{"type": "divider", "divider": {}}, heading(2, "Quality check"),
            para("Every reference image and every video is checked against the simulator. The check compares the outlines "
                 "of the robot and the object with the object borders from the simulator. The background is not checked, "
                 "so it can change. The check looks at position only, not at image quality."),
            table([["Item", "Count"],
                   ["Videos in this report (all passed)", str(n_videos)],
                   ["Reference images rejected (kept in images/rejected/)", str(rej_img)],
                   ["Reference slots that fell back to a prompt-only video", str(fallbacks)],
                   ["Videos rejected and made again (kept in wans/rejected/)", str(rej_vid)]])]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--parent", default=CRAFT_PAGE, help="Notion parent page id (default: CRAFT)")
    p.add_argument("--title", default=None, help="page title (default: '<Task> <tag> - reference and prompt test')")
    p.add_argument("--icon", default="\U0001F3AC")
    p.add_argument("--controls", default=None, help="only these controls (comma-separated)")
    p.add_argument("--dry_run", action="store_true", help="print the plan, upload nothing, create nothing")
    a = p.parse_args()
    token = "" if a.dry_run else open(TOKEN_FILE).read().strip()
    page, n = build(a.run_dir, Notion(token, a.dry_run), a.parent, a.title, a.icon,
                    set(a.controls.split(",")) if a.controls else None)
    print(f"report: {n} files uploaded" + ("" if a.dry_run else f", page https://www.notion.so/{page.replace('-', '')}"))
    sys.exit(0)
