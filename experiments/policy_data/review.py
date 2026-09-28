"""Review page: look at the Cosmos videos (or the simulator demos) of a dataset and approve or reject them.

  python experiments/policy_data/review.py <dataset_dir> [--sim] [--group 000-049] [--port 8765] [--host 0.0.0.0]

It starts a small web server (Python standard library only) and prints the address with a secret token. Only
requests with the token are served, and only .mp4 and .png files inside the dataset folder. The token is kept in
<dataset>/.review_token, so the address stays the same when review.py starts again.

Videos tab: every Cosmos video of the group, with the simulator video (it plays together with the Cosmos video) and
the reference image side by side, and the variation of the image (place, table, lighting, towel, robot). Mark each
video approved or rejected, tick "reference image problem" (only with Reject) when the image is the cause, and write
one line. Submit writes review.csv (the current verdict of each video; a new submit replaces it, U and Submit remove
it) and adds every submitted line to review_history.csv. Then dataset.csv is updated. Submit as often as you like;
videos without a verdict stay unreviewed. A line without a verdict is not saved: choose Approve or Reject first.
When the verdict changed in another tab since this page loaded it, the submit of that video is refused: reload.
Simulator tab (--sim starts there): the simulator demos of the group, for the check right after make_dataset.py
--take. Reject a demo that looks wrong; status.py then prints the make_dataset.py --replace command. It writes
sim_review.csv.
Rejected tab: the rejected videos (the ones still in place and the ones moved to cosmos_rejected/) and the rejected
reference images of the group, to look at only.
Keys: A approve, R reject, U clear, arrow down or J next, arrow up or K previous, space play or pause. In the text
field, Enter moves to the next demo.

review.py is the only writer of review.csv, review_history.csv and sim_review.csv. It writes them under the dataset
lock (see status.py), so other scripts can run at the same time.
From another machine: run the ssh command that the script prints on that machine, then open the address. With VSCode
Remote-SSH, run the ssh command in a VSCode terminal and forward the port in the Ports panel (VSCode often does it by
itself).
"""

import argparse
import hmac
import http.server
import json
import mimetypes
import os
import re
import secrets
import socket
import sys
import time
import urllib.parse

import layout
import status

FIELDS = ["place", "table", "lighting", "robot", "towel_color", "towel_material", "towel_pattern"]
SERVED = (".mp4", ".png")
DEFAULT_PORT = 8765


def variations(ds: str) -> dict:
    """(reference name, source) -> row of refs/references.csv with the highest attempt."""
    out = {}
    for r in status.read_csv(f"{ds}/refs/references.csv"):
        k = (r["name"], r.get("source") or "")
        if k not in out or int(r.get("attempt") or 0) >= int(out[k].get("attempt") or 0):
            out[k] = r
    return out


def type_counts(rows: list) -> list:
    """[[place type, approved, total], ...] over the whole dataset, strong light last."""
    counts = {}
    for r in rows:
        keys = [r["place_type"]] + (["strong light"] if str(r["strong_light"]) == "1" else [])
        for k in keys:
            c = counts.setdefault(k, [0, 0])
            c[0] += r["state"] == "approved"
            c[1] += 1
    order = sorted(k for k in counts if k != "strong light") + (["strong light"] if "strong light" in counts else [])
    return [[k, *counts[k]] for k in order]


def group_states(rows: list, group: str) -> dict:
    states = {}
    for r in rows:
        if r["group"] == group:
            states[r["state"]] = states.get(r["state"], 0) + 1
    return states


def items(ds: str, mode: str, group: str) -> dict:
    """Everything the page shows for one tab and group."""
    with status.lock(ds):
        rows, video_rows, problems = status.collect(ds)
    var = variations(ds)
    groups = sorted({r["group"] for r in rows})
    group = group if group in groups else (groups[0] if groups else "")
    out = []
    if mode == "videos":
        reviews = {r["video"]: r for r in status.read_csv(f"{ds}/review.csv")}
        for r in rows:
            if r["group"] != group or not r["video"] or " " in r["video"]:
                continue  # no video yet, or a conflict (two videos): status.py lists it
            vid = status.video_id(r["video"])
            rev = reviews.get(vid, {})
            v = var.get((r["reference"], r["source"]), {})
            out.append({
                "key": vid, "demo": r["demo"], "name": r["reference"], "source": r["source"], "state": r["state"],
                "note": r["note"], "place_type": r["place_type"], "strong_light": str(r["strong_light"]) == "1",
                "video": vid, "variation": {f: v.get(f, "") for f in FIELDS},
                "files": {"sim": f"demos/{r['group']}/{layout.demo_name(r['demo'])}/{layout.demo_name(r['demo'])}"
                                 "_source.mp4",
                          "cosmos": r["video"],
                          "ref": f"refs/{r['group']}/{r['reference']}.png"},
                "verdict": rev.get("verdict", ""), "ref_problem": rev.get("ref_problem", "") == "1",
                "review": rev.get("review", ""), "reviewed_at": rev.get("reviewed_at", ""),
                "rejected_before": r["videos_rejected"]})
    elif mode == "sim":
        sims = {(int(r["demo"]), r["source"]): r for r in status.read_csv(f"{ds}/sim_review.csv")}
        for r in rows:
            if r["group"] != group:
                continue
            rev = sims.get((r["demo"], r["source"]), {})
            name = layout.demo_name(r["demo"])
            out.append({
                "key": f"{r['demo']}|{r['source']}", "demo": r["demo"], "name": name, "source": r["source"],
                "state": r["state"], "note": r["note"], "place_type": r["place_type"],
                "strong_light": str(r["strong_light"]) == "1",
                "files": {"sim": f"demos/{r['group']}/{name}/{name}_source.mp4",
                          "ref": f"ref_sim/{r['group']}/{name}_ref_sim.png"},
                "verdict": rev.get("verdict", ""), "review": rev.get("review", ""),
                "reviewed_at": rev.get("reviewed_at", "")})
    else:  # rejected: to look at only
        for r in rows:
            if r["group"] == group and r["review"] == "rejected" and r["video"] and " " not in r["video"]:
                out.append({"kind": "video", "demo": r["demo"], "name": r["reference"], "file": r["video"],
                            "review": r["review_text"], "source": r["source"],
                            "where": "still in place (run_cosmos.py --redo_bad makes it again, next version)"})
        for v in video_rows:
            if v["where"] == "rejected" and layout.chunk(int(v["demo"])) == group:
                out.append({"kind": "video", "demo": int(v["demo"]), "name": v["name"], "file": v["file"],
                            "review": v["review_text"], "source": v["source"], "where": "cosmos_rejected/"})
        for r in status.read_csv(f"{ds}/refs_rejected/rejected.csv"):
            if layout.chunk(int(r["demo"])) == group:
                out.append({"kind": "image", "demo": int(r["demo"]), "name": r["name"], "file": r["file"],
                            "review": r["reason"], "source": r.get("source", ""),
                            "where": f"refs_rejected/ (room {r['room_score']}, wrist {r['wrist_score']})"})
        out.sort(key=lambda x: (x["demo"], x["kind"], x["name"]))
    return {"dataset": os.path.basename(ds), "mode": mode, "group": group, "groups": groups, "items": out,
            "states": group_states(rows, group), "types": type_counts(rows), "problems": len(problems)}


def _item_error(it) -> str:
    if not isinstance(it, dict):
        return "an item is not an object"
    if it.get("verdict", "") not in ("approved", "rejected", ""):
        return f"{it.get('key', '?')}: unknown verdict {it.get('verdict')!r}"
    return ""


def submit(ds: str, body: dict) -> dict:
    """Write the verdicts of one submit. A video (or demo) whose saved verdict changed since the page loaded it
    (loaded_at is not the stored reviewed_at) is refused. -> {"updated": {key: saved row}, "errors": [...],
    "states", "types"}"""
    mode, group, now = body.get("mode"), str(body.get("group", "")), time.strftime("%Y-%m-%dT%H:%M:%S")
    if mode not in ("videos", "sim") or not isinstance(body.get("items"), list):
        return {"updated": {}, "errors": ["bad request"]}
    updated, errors, history = {}, [], []
    with status.lock(ds):
        rows, _, _ = status.collect(ds)
        if mode == "videos":
            path = f"{ds}/review.csv"
            saved = {r["video"]: r for r in status.read_csv(path)}
            known = {status.video_id(r["video"]): r for r in rows if r["video"] and " " not in r["video"]}
            for it in body["items"]:
                err = _item_error(it)
                vid = str(it.get("key", "")) if not err else ""
                r = known.get(vid)
                if not err and r is None:
                    err = f"{vid}: not a current video any more (made again or replaced), not saved"
                if not err and saved.get(vid, {}).get("reviewed_at", "") != str(it.get("loaded_at", "")):
                    err = (f"{r['reference']}: changed in another tab at {saved.get(vid, {}).get('reviewed_at') or '?'}"
                           ": reload the page")
                if err:
                    errors.append(err)
                    continue
                verdict = it.get("verdict", "")
                row = {"video": vid, "demo": r["demo"], "name": r["reference"], "source": r["source"],
                       "verdict": verdict, "ref_problem": "1" if it.get("ref_problem") and verdict == "rejected"
                       else "", "review": str(it.get("review", "")).strip()[:500], "reviewed_at": now if verdict
                       else ""}
                if verdict:
                    saved[vid] = row
                else:
                    saved.pop(vid, None)
                history.append({"kind": "video", **row, "reviewed_at": now})
                updated[vid] = {"verdict": verdict, "ref_problem": row["ref_problem"] == "1",
                                "review": row["review"] if verdict else "", "reviewed_at": row["reviewed_at"]}
            status.write_csv(path, list(saved.values()), status.REVIEW_COLUMNS)
        else:
            path = f"{ds}/sim_review.csv"
            saved = {f"{int(r['demo'])}|{r['source']}": r for r in status.read_csv(path)}
            current = {f"{r['demo']}|{r['source']}" for r in rows}
            for it in body["items"]:
                err = _item_error(it)
                key = str(it.get("key", "")) if not err else ""
                if not err and key not in current:
                    err = f"{key.split('|')[0]}: its simulator demo changed (--replace), not saved"
                if not err and saved.get(key, {}).get("reviewed_at", "") != str(it.get("loaded_at", "")):
                    err = f"demo {key.split('|')[0]}: changed in another tab: reload the page"
                if err:
                    errors.append(err)
                    continue
                verdict = it.get("verdict", "")
                n, src = key.split("|", 1)
                row = {"demo": int(n), "source": src, "verdict": verdict,
                       "review": str(it.get("review", "")).strip()[:500], "reviewed_at": now if verdict else ""}
                if verdict:
                    saved[key] = row
                else:
                    saved.pop(key, None)
                history.append({"kind": "sim", "video": "", "name": layout.demo_name(int(n)), "ref_problem": "",
                                **row, "reviewed_at": now})
                updated[key] = {"verdict": verdict, "review": row["review"] if verdict else "",
                                "reviewed_at": row["reviewed_at"]}
            status.write_csv(path, sorted(saved.values(), key=lambda r: int(r["demo"])), status.SIM_REVIEW_COLUMNS)
        for h in history:
            status.append_csv(f"{ds}/review_history.csv", h, status.REVIEW_HISTORY_COLUMNS)
        rows, video_rows, _ = status.collect(ds)  # the same as status.update, once
        status.write_csv(f"{ds}/dataset.csv", rows, status.DATASET_COLUMNS)
        status.write_csv(f"{ds}/videos.csv", video_rows, status.VIDEO_COLUMNS)
    return {"updated": updated, "errors": errors, "states": group_states(rows, group), "types": type_counts(rows)}


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "review"
    ds = ""
    token = ""
    start_mode = "videos"
    start_group = ""

    def log_message(self, fmt, *args):
        pass  # quiet: the page shows errors itself

    def _ok(self, query: dict) -> bool:
        given = query.get("t", [""])[0].encode("utf-8", "surrogateescape")
        if hmac.compare_digest(given, self.token.encode()):
            return True
        self.send_error(403, "wrong or missing token")
        return False

    def _json(self, obj, code: int = 200) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(url.query)
        if not self._ok(query):
            return
        try:
            if url.path == "/":
                data = PAGE.replace("__MODE__", self.start_mode).replace("__GROUP__", self.start_group).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            elif url.path == "/api/items":
                mode = query.get("mode", ["videos"])[0]
                self._json(items(self.ds, mode if mode in ("videos", "sim", "rejected") else "videos",
                                 query.get("group", [""])[0]))
            elif url.path.startswith("/file/"):
                self._file(urllib.parse.unquote(url.path[len("/file/"):]))
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001 - answer instead of dropping the connection
            self._json({"errors": [f"server error: {type(e).__name__}: {e}"]}, 500)

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        if not self._ok(urllib.parse.parse_qs(url.query)):
            return
        if url.path != "/api/submit":
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length", "-1"))
            if n < 0:
                self._json({"updated": {}, "errors": ["bad request (no Content-Length)"]}, 400)
                return
            body = json.loads(self.rfile.read(n) or b"{}")
            self._json(submit(self.ds, body if isinstance(body, dict) else {}))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except ValueError:
            self._json({"updated": {}, "errors": ["bad request"]}, 400)
        except Exception as e:  # noqa: BLE001 - answer instead of dropping the connection
            self._json({"updated": {}, "errors": [f"server error: {type(e).__name__}: {e}"]}, 500)

    def _file(self, rel: str) -> None:
        """Serve a .mp4 or .png inside the dataset folder, with byte ranges (the browser needs them to seek)."""
        if "\0" in rel:
            self.send_error(404)
            return
        root = os.path.realpath(self.ds)
        path = os.path.realpath(os.path.join(root, rel))
        if not path.startswith(root + os.sep) or not path.endswith(SERVED) or not os.path.isfile(path):
            self.send_error(404)
            return
        size = os.path.getsize(path)
        start, end = 0, size - 1
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip())
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                block = f.read(min(1 << 20, left))
                if not block:
                    break
                self.wfile.write(block)
                left -= len(block)


def own_address() -> str:
    """The address other machines reach this one on (the interface of the default route), not 127.x."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # no packet is sent: this only picks the interface
            ip = s.getsockname()[0]
        if not ip.startswith("127."):
            return ip
    except OSError:
        pass
    return "<address of this machine>"


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # like the server: old closed connections are fine
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def token_of(ds: str) -> str:
    """The token of this dataset: kept in <dataset>/.review_token (only the owner can read it)."""
    path = f"{ds}/.review_token"
    if os.path.isfile(path):
        token = open(path).read().strip()
        if token:
            return token
    token = secrets.token_urlsafe(16)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token + "\n")
    return token


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", help="dataset folder made by make_dataset.py --take")
    ap.add_argument("--sim", action="store_true", help="start on the simulator tab")
    ap.add_argument("--group", default="", help="group to show first, e.g. 000-049 (default: the first)")
    ap.add_argument("--port", type=int, default=None, help=f"port (default: the first free from {DEFAULT_PORT})")
    ap.add_argument("--host", default="0.0.0.0", help="address to listen on (default all; the token protects it)")
    args = ap.parse_args()
    ds = os.path.abspath(args.dataset_dir)
    if not status.is_dataset(ds):
        sys.exit(f"{ds} is not a dataset folder (no sources.csv)")
    if args.port is not None:
        if not port_is_free(args.host, args.port):
            sys.exit(f"port {args.port} is in use (another review.py?): stop it or give another --port")
        port = args.port
    else:
        port = next((p for p in range(DEFAULT_PORT, DEFAULT_PORT + 50) if port_is_free(args.host, p)), None)
        if port is None:
            sys.exit(f"no free port in {DEFAULT_PORT}-{DEFAULT_PORT + 49}: give --port")
    Handler.ds, Handler.token = ds, token_of(ds)
    Handler.start_mode, Handler.start_group = ("sim" if args.sim else "videos"), args.group
    server = http.server.ThreadingHTTPServer((args.host, port), Handler)
    server.daemon_threads = True
    far = "" if args.host in ("127.0.0.1", "localhost") else (
        f"  from another one:  ssh -N -L {port}:{own_address()}:{port} <user>@<server>   (then open the address "
        "above)\n")
    print(f"review page for {os.path.relpath(ds)}\n"
          f"  on this machine:   http://localhost:{port}/?t={Handler.token}\n{far}"
          "  stop with Ctrl-C", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset review</title>
<style>
:root { --bg: #ffffff; --fg: #16181d; --muted: #667085; --line: #d0d5dd; --card: #f8f9fb; --ok: #16803c;
        --bad: #c4320a; --accent: #2e5bdb; --warn: #b54708; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #111318; --fg: #e8eaf0; --muted: #98a2b3; --line: #344054; --card: #1a1d24; --ok: #32d583;
          --bad: #f97066; --accent: #7ea4ff; --warn: #fdb022; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.45 system-ui, -apple-system, sans-serif; }
header { position: sticky; top: 0; z-index: 3; background: var(--bg); border-bottom: 1px solid var(--line);
         padding: 8px 16px; display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; }
header b { font-size: 15px; }
.tabs button { border: 1px solid var(--line); background: var(--card); color: var(--fg); padding: 4px 10px;
               border-radius: 6px; cursor: pointer; }
.tabs button.on { background: var(--accent); border-color: var(--accent); color: #fff; }
select { background: var(--card); color: var(--fg); border: 1px solid var(--line); border-radius: 6px; padding: 3px; }
.muted { color: var(--muted); }
#types { padding: 6px 16px; font-size: 12px; }
main { padding-bottom: 70px; }
.item { border: 1px solid var(--line); border-radius: 8px; margin: 12px 16px; padding: 10px; background: var(--card); }
.item.focus { outline: 2px solid var(--accent); }
.head { display: flex; flex-wrap: wrap; gap: 4px 12px; align-items: baseline; }
.head .name { font-weight: 600; }
.var { font-size: 12px; margin: 4px 0 8px; }
.media { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.media.two { grid-template-columns: 2fr 1fr; }
.media figure { margin: 0; }
.media figcaption { font-size: 11px; color: var(--muted); }
video, .media img { width: 100%; aspect-ratio: 2 / 1; background: #000; display: block; border-radius: 4px; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 8px; }
.v { border: 1px solid var(--line); background: var(--bg); color: var(--fg); border-radius: 6px; padding: 5px 12px;
     cursor: pointer; font-weight: 600; }
.v.on.approved { background: var(--ok); border-color: var(--ok); color: #fff; }
.v.on.rejected { background: var(--bad); border-color: var(--bad); color: #fff; }
label.dis { opacity: .5; }
input.review { flex: 1; min-width: 200px; padding: 5px 8px; border: 1px solid var(--line); border-radius: 6px;
               background: var(--bg); color: var(--fg); }
.changed { color: var(--warn); font-size: 12px; }
footer { position: fixed; bottom: 0; left: 0; right: 0; z-index: 3; background: var(--bg);
         border-top: 1px solid var(--line); padding: 8px 16px; display: flex; flex-wrap: wrap; gap: 8px 16px;
         align-items: center; }
#submit { background: var(--accent); color: #fff; border: 0; border-radius: 6px; padding: 7px 16px; font-weight: 600;
          cursor: pointer; }
#submit:disabled { opacity: .5; cursor: default; }
.err { color: var(--bad); }
@media (max-width: 800px) { .media, .media.two { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <b id="title">Dataset review</b>
  <span class="tabs">
    <button data-mode="videos">Videos</button><button data-mode="sim">Simulator</button>
    <button data-mode="rejected">Rejected</button>
  </span>
  <label>Group <select id="group"></select></label>
  <span id="summary" class="muted"></span>
</header>
<div id="types" class="muted"></div>
<main id="list"></main>
<footer>
  <button id="submit" disabled>Submit</button>
  <span id="msg" class="muted"></span>
  <span class="muted">A approve &middot; R reject &middot; U clear &middot; &darr;/J next &middot; &uarr;/K back
    &middot; Space play</span>
</footer>
<script>
"use strict";
const T = new URLSearchParams(location.search).get("t") || "";
let MODE = "__MODE__", GROUP = "__GROUP__", ITEMS = [], FOCUS = 0, SAVING = false;
const changed = new Map();  // key -> {verdict, ref_problem, review}: edits not submitted yet
const $ = (id) => document.getElementById(id);
const fileUrl = (rel) => "/file/" + rel.split("/").map(encodeURIComponent).join("/") + "?t=" + encodeURIComponent(T);
async function api(path, opts) {
  let r;
  try { r = await fetch(path + (path.includes("?") ? "&" : "?") + "t=" + encodeURIComponent(T), opts); }
  catch (e) { throw new Error("the review server does not answer (stopped?); your changes are kept on this page"); }
  if (r.status === 403) throw new Error("the review server does not accept this address: open the address it printed");
  let d = null;
  try { d = await r.json(); } catch (e) { /* not json */ }
  if (!r.ok) throw new Error((d && d.errors && d.errors.join("; ")) || `server error ${r.status}`);
  return d;
}
function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") e.className = v; else if (k === "text") e.textContent = v; else e.setAttribute(k, v);
  }
  for (const k of kids) if (k) e.append(k);
  return e;
}
const observer = new IntersectionObserver((entries) => {
  for (const en of entries) {
    for (const m of en.target.querySelectorAll("video, img[data-src]")) {
      if (en.isIntersecting && !m.src && m.dataset.src) m.src = m.dataset.src;
      if (!en.isIntersecting && m.tagName === "VIDEO") m.pause();
    }
  }
}, { rootMargin: "300px" });

function showMsg(text, isErr) { $("msg").className = isErr ? "err" : "muted"; $("msg").textContent = text; }
function summary(states, types) {
  const done = ITEMS.filter((i) => i.verdict).length;
  const st = Object.entries(states || {}).map(([k, v]) => `${k} ${v}`).join(", ");
  const nv = ITEMS.filter((i) => i.kind === "video").length, ni = ITEMS.filter((i) => i.kind === "image").length;
  const count = MODE === "rejected" ? `${nv} rejected videos, ${ni} rejected images`
    : `reviewed ${done}/${ITEMS.length}`;
  $("summary").textContent = `${count} | group: ${st}`;
  $("types").textContent = "approved per place type (whole dataset): "
    + (types || []).map(([k, a, n]) => `${k} ${a}/${n}`).join(", ");
}

async function load() {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.mode === MODE));
  $("list").textContent = "loading...";
  let d;
  try { d = await api(`/api/items?mode=${MODE}&group=${encodeURIComponent(GROUP)}`); }
  catch (e) { $("list").textContent = String(e.message || e); return; }
  GROUP = d.group; ITEMS = d.items; FOCUS = 0; changed.clear(); updateSubmit();
  $("title").textContent = d.dataset;
  const sel = $("group"); sel.textContent = "";
  for (const g of d.groups) sel.append(el("option", { value: g, text: g }));
  sel.value = GROUP;
  summary(d.states, d.types);
  if (d.problems) showMsg(`${d.problems} problems: see status.py`, true);
  render();
}

function render() {
  observer.disconnect();
  const list = $("list"); list.textContent = "";
  if (!ITEMS.length) {
    list.append(el("p", { class: "muted", style: "padding:16px",
      text: MODE === "videos" ? "No Cosmos video in this group yet." : "Nothing here." }));
    return;
  }
  ITEMS.forEach((it, i) => {
    const card = MODE === "rejected" ? rejectedCard(it, i) : reviewCard(it, i);
    card.addEventListener("click", () => setFocus(i, false));
    list.append(card); observer.observe(card);
    if (card._ui) paint(i);
  });
  setFocus(0, false);
}

function headOf(it) {
  const h = el("div", { class: "head" }, el("span", { class: "name", text: it.name }),
    el("span", { class: "muted", text: `${it.place_type || ""}${it.strong_light ? " · strong light" : ""}` }),
    el("span", { class: "muted", text: `source ${it.source}` }),
    el("span", { class: "muted", text: `state ${it.state}${it.note ? " (" + it.note + ")" : ""}` }));
  if (it.rejected_before) h.append(el("span", { class: "muted", text: `${it.rejected_before} rejected before` }));
  h.append(el("span", { class: "muted saved" }));
  return h;
}

function reviewCard(it, i) {
  const card = el("div", { class: "item", id: "item" + i });
  card.append(headOf(it));
  if (it.variation) {
    const v = it.variation;
    card.append(el("div", { class: "var muted", text: [v.place, `table: ${v.table}`, `light: ${v.lighting}`,
      `towel: ${v.towel_color} ${v.towel_material}, ${v.towel_pattern}`, v.robot].filter(Boolean).join(" | ") }));
  }
  const media = el("div", { class: "media" + (MODE === "sim" ? " two" : "") });
  const sim = el("video", { "data-src": fileUrl(it.files.sim), preload: "none", muted: "", playsinline: "",
    ...(MODE === "sim" ? { controls: "" } : {}) });
  sim.muted = true;
  media.append(el("figure", {}, sim, el("figcaption", { text: "simulator" })));
  if (MODE === "videos") {
    const cos = el("video", { "data-src": fileUrl(it.files.cosmos), preload: "none", controls: "", muted: "",
      playsinline: "" });
    cos.muted = true;
    const sync = () => { if (Math.abs(sim.currentTime - cos.currentTime) > 0.1) sim.currentTime = cos.currentTime; };
    cos.addEventListener("play", () => {
      if (!sim.src) sim.src = sim.dataset.src;
      sync(); sim.play().catch(() => {});
    });
    cos.addEventListener("pause", () => sim.pause());
    cos.addEventListener("seeked", sync);
    cos.addEventListener("timeupdate", sync);
    media.append(el("figure", {}, cos, el("figcaption", { text: "Cosmos" })));
  }
  const img = el("img", { "data-src": fileUrl(it.files.ref), alt: "" });
  media.append(el("figure", {}, img, el("figcaption", { text: MODE === "videos" ? "reference image" : "frame 0" })));
  card.append(media);

  const c = el("div", { class: "controls" });
  const ok = el("button", { class: "v", text: MODE === "sim" ? "Looks right (A)" : "Approve (A)" });
  const bad = el("button", { class: "v", text: MODE === "sim" ? "Looks wrong (R)" : "Reject (R)" });
  ok.onclick = () => { setFocus(i, false); setVerdict(i, "approved"); ok.blur(); };
  bad.onclick = () => { setFocus(i, false); setVerdict(i, "rejected"); bad.blur(); };
  c.append(ok, bad);
  let box = null, boxLabel = null;
  if (MODE === "videos") {
    box = el("input", { type: "checkbox" });
    box.onchange = () => { setFocus(i, false); edit(i, { ref_problem: box.checked }); box.blur(); };
    boxLabel = el("label", { title: "only with Reject: the reference image is the cause" }, box,
      " reference image problem");
    c.append(boxLabel);
  }
  const txt = el("input", { type: "text", class: "review", placeholder: "one line (why)", maxlength: "500" });
  txt.onfocus = () => setFocus(i, false);
  txt.oninput = () => edit(i, { review: txt.value });
  txt.onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); txt.blur(); setFocus(i + 1, true); } };
  c.append(txt, el("span", { class: "changed" }));
  card.append(c);
  card._ui = { ok, bad, box, boxLabel, txt };
  return card;
}

function rejectedCard(it, i) {
  const card = el("div", { class: "item", id: "item" + i });
  card.append(el("div", { class: "head" }, el("span", { class: "name", text: `${it.name} (${it.kind})` }),
    el("span", { class: "muted", text: `source ${it.source}` }), el("span", { class: "muted", text: it.where })));
  card.append(el("div", { class: "var muted", text: it.review || "" }));
  const m = it.kind === "video" ? el("video", { "data-src": fileUrl(it.file), controls: "", preload: "none" })
                                : el("img", { "data-src": fileUrl(it.file), alt: "" });
  card.append(el("div", { class: "media two" }, el("figure", {}, m)));
  return card;
}

function saved(i) {
  const it = ITEMS[i];
  return { verdict: it.verdict || "", ref_problem: !!it.ref_problem, review: it.review || "" };
}
function current(i) { return changed.get(ITEMS[i].key) || saved(i); }
const same = (a, b) => a.verdict === b.verdict && a.ref_problem === b.ref_problem && a.review === b.review;
function edit(i, patch) {
  if (i < 0 || i >= ITEMS.length || MODE === "rejected") return;
  const next = { ...current(i), ...patch };
  if (next.verdict !== "rejected") next.ref_problem = false;  // the tick belongs to Reject only
  if (same(next, saved(i))) changed.delete(ITEMS[i].key); else changed.set(ITEMS[i].key, next);
  paint(i); updateSubmit();
}
function setVerdict(i, v) { edit(i, { verdict: v }); }
function paint(i) {
  const card = $("item" + i); if (!card || !card._ui) return;
  const it = ITEMS[i], cur = current(i), ui = card._ui;
  ui.ok.className = "v" + (cur.verdict === "approved" ? " on approved" : "");
  ui.bad.className = "v" + (cur.verdict === "rejected" ? " on rejected" : "");
  if (ui.box) {
    ui.box.checked = cur.ref_problem; ui.box.disabled = cur.verdict !== "rejected";
    ui.boxLabel.className = ui.box.disabled ? "dis" : "";
  }
  if (document.activeElement !== ui.txt) ui.txt.value = cur.review;
  const dirty = changed.has(it.key);
  let note = dirty ? "not submitted" : "";
  if (dirty && !cur.verdict && (cur.review || saved(i).verdict === "")) note = "choose Approve or Reject";
  if (dirty && !cur.verdict && saved(i).verdict && !cur.review) note = "not submitted: the verdict will be removed";
  card.querySelector(".changed").textContent = note;
  card.querySelector(".saved").textContent = it.reviewed_at ? `saved ${it.reviewed_at}` : "";
}
function setFocus(i, scroll) {
  if (i < 0 || i >= ITEMS.length) return;
  document.querySelectorAll(".item.focus").forEach((e) => e.classList.remove("focus"));
  FOCUS = i;
  const card = $("item" + i);
  if (!card) return;
  card.classList.add("focus");
  if (scroll) card.scrollIntoView({ behavior: "smooth", block: "center" });
}
function sendable() {
  // a line or a tick without a verdict is not sent; removing a saved verdict (U) is sent
  return [...changed.entries()].filter(([k, c]) => {
    const i = ITEMS.findIndex((it) => it.key === k);
    return i >= 0 && (c.verdict || (saved(i).verdict && !c.review));
  });
}
function updateSubmit() {
  const n = sendable().length, waiting = changed.size - n;
  $("submit").disabled = SAVING || n === 0;
  $("submit").textContent = n ? `Submit (${n})` : "Submit";
  if (!SAVING && waiting) showMsg(`${waiting} with a line but no verdict: choose Approve or Reject`, false);
}

$("submit").onclick = async () => {
  const sent = new Map(sendable().map(([k, c]) => [k, { ...c }]));
  const body = { mode: MODE, group: GROUP, items: [...sent.entries()].map(([k, c]) => {
    const it = ITEMS.find((x) => x.key === k);
    return { key: k, verdict: c.verdict, ref_problem: c.ref_problem, review: c.review,
             loaded_at: it.reviewed_at || "" };
  }) };
  SAVING = true; updateSubmit(); showMsg("saving...", false);
  try {
    const r = await api("/api/submit", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) });
    for (const [k, row] of Object.entries(r.updated || {})) {
      const i = ITEMS.findIndex((it) => it.key === k);
      if (i < 0) continue;
      Object.assign(ITEMS[i], row);
      const now = changed.get(k);
      if (now && same(now, sent.get(k))) changed.delete(k);  // edits made while saving stay
      paint(i);
    }
    summary(r.states, r.types);
    const n = Object.keys(r.updated || {}).length, errs = r.errors || [];
    SAVING = false; updateSubmit();
    showMsg(`saved ${n}` + (errs.length ? " | not saved: " + errs.join("; ") : ""), errs.length > 0);
  } catch (e) {
    SAVING = false; updateSubmit(); showMsg("not saved: " + (e.message || e), true);
  }
};
document.querySelectorAll(".tabs button").forEach((b) => b.onclick = () => {
  if (changed.size && !confirm("Changes are not submitted. Leave this tab?")) return;
  MODE = b.dataset.mode; b.blur(); load();
});
$("group").onchange = () => {
  if (changed.size && !confirm("Changes are not submitted. Leave this group?")) { $("group").value = GROUP; return; }
  GROUP = $("group").value; $("group").blur(); load();
};
document.addEventListener("keydown", (e) => {
  const t = e.target;
  if ((t.tagName === "INPUT" && t.type === "text") || t.tagName === "SELECT" || e.ctrlKey || e.metaKey || e.altKey) {
    return;
  }
  const k = e.key.toLowerCase();
  if (k === "a") setVerdict(FOCUS, "approved");
  else if (k === "r") setVerdict(FOCUS, "rejected");
  else if (k === "u") setVerdict(FOCUS, "");
  else if (k === "arrowdown" || k === "j") { e.preventDefault(); setFocus(FOCUS + 1, true); }
  else if (k === "arrowup" || k === "k") { e.preventDefault(); setFocus(FOCUS - 1, true); }
  else if (k === " ") {
    e.preventDefault();
    const card = $("item" + FOCUS);
    const vids = card ? card.querySelectorAll("video") : [];
    const main = vids[vids.length - 1];
    if (main) {
      if (!main.src) main.src = main.dataset.src;
      if (main.paused) main.play().catch(() => {}); else main.pause();
    }
  }
});
window.addEventListener("beforeunload", (e) => { if (changed.size) { e.preventDefault(); e.returnValue = ""; } });
load();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
