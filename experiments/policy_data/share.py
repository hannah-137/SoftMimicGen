"""Build or update the share folder of a dataset: the Cosmos videos sorted by review result, the list files and a
view-only page.

    python experiments/policy_data/share.py <dataset> <share folder> [--owner UID:GID]

Run it again after new videos or new reviews. It copies only files that are new or changed, moves a demo folder
when its review result changed, and rewrites the lists, the page and the README. The share folder is a copy, not
hard links: a later change of the dataset reaches it only with the next run.

Layout of the share folder:
    videos/<O|weak|X|not_reviewed>/demo_NNN/    the files of one demo (see README)
    lists/                                      review.csv, videos.csv, dataset.csv, ... (never .review_token)
    index.html, serve.py, README.md               the page, its small web server (a copy of share_serve.py), the notes

A demo is in O when its video is approved, in weak or X when it is marked weak or rejected with a reason, else in
not_reviewed (the rule of status.py). Only the current video of a demo is shared. The files of one demo are the
Cosmos video and its settings, the reference image, frame 0 of the simulator, the simulator video with every step
(source_all), its edge video (geoedge_all), its instance mask video (instance_all), and a text file with the two prompts.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time

import layout
import status

FOLDERS = ["O", "weak", "X", "not_reviewed"]
FOLDER_OF = {"approved": "O", "weak": "weak", "rejected": "X"}
VERDICT_OF = {"O": "approved", "weak": "weak", "X": "rejected", "not_reviewed": ""}
LISTS = ["review.csv", "review_history.csv", "videos.csv", "dataset.csv", "sources.csv", "replacements.csv",
         "run_config.json", "refs/references.csv", "refs/check_references.csv", "refs_rejected/rejected.csv"]
SIM_FILES = ["source_all", "geoedge_all", "instance_all"]


def folder_of(rev: dict, version: int) -> str:
    """The share folder of a video from its review.csv row: O, weak, X or not_reviewed."""
    verdict = rev.get("verdict", "")
    if verdict == "approved":
        return "O"
    if verdict in status.NEEDS_REASON and status.has_reason(rev, version):
        return FOLDER_OF[verdict]
    return "not_reviewed"


def cloth(row: dict) -> str:
    """The cloth words of a reference image row (the column names differ between spec versions)."""
    for word in ("fabric", "towel"):
        parts = [row.get(f"{word}_{k}", "") for k in ("thickness", "color", "material")]
        if any(parts):
            extra = [row.get(f"{word}_{k}", "") for k in ("condition", "pattern")]
            return " ".join(p for p in parts if p) + "".join(f", {e}" for e in extra if e)
    return ""


def collect(ds: str) -> list:
    """One item per demo with a current video: the files to share and the words of the page."""
    version = status.review_items(ds)
    reviews = {r["video"]: r for r in status.read_csv(f"{ds}/review.csv")}
    refs = {}
    for r in status.read_csv(f"{ds}/refs/references.csv"):
        refs[r["name"]] = r
    dataset = {int(r["demo"]): r for r in status.read_csv(f"{ds}/dataset.csv")}
    items = []
    for v in status.read_csv(f"{ds}/videos.csv"):
        if v.get("where") != "current":
            continue
        n, name = int(v["demo"]), v["name"]
        video_id = status.video_id(v["file"])
        rev = reviews.get(video_id, v)
        demo = layout.demo_name(n)
        ref_name = dataset.get(n, {}).get("reference") or name
        ref = refs.get(ref_name, {})
        try:
            settings = json.load(open(f"{ds}/{v['file'][:-4]}.json"))
        except (OSError, ValueError):
            settings = {}
        info = settings.get("policy_data", {})
        if isinstance(info, str):  # older json files hold this field as text
            info = {}
        files = {f"{name}.mp4": f"{ds}/{v['file']}", f"{name}.json": f"{ds}/{v['file'][:-4]}.json",
                 f"{ref_name}.png": f"{layout.ref_dir(f'{ds}/refs', n)}/{ref_name}.png",
                 f"{demo}_ref_sim.png": f"{layout.ref_sim_dir(ds, n)}/{demo}_ref_sim.png"}
        for k in SIM_FILES:
            files[f"{demo}_{k}.mp4"] = f"{layout.demo_dir(ds, n)}/{demo}_{k}.mp4"
        items.append({
            "n": n, "name": name, "ref": ref_name, "f": folder_of(rev, version), "v": rev.get("verdict", ""),
            "s": v.get("spec") or ref.get("spec", ""),
            "r": status.reason_ids(rev.get("reasons"), version), "views": status.views_level(rev.get("views")),
            "refp": rev.get("ref_problem") == "1", "note": (rev.get("review") or v.get("review_text") or "").strip(),
            "src": v.get("source", ""),
            "place": ref.get("place", ""), "table": ref.get("table", ""), "light": ref.get("lighting", ""),
            "cloth": cloth(ref), "floor": ref.get("floor", ""),
            "prompt": settings.get("prompt", ""), "ip": ref.get("prompt", ""), "im": ref.get("model", ""),
            "files": files,
        })
    return sorted(items, key=lambda x: x["n"]), version


def same_file(src: str, dst: str) -> bool:
    try:
        a, b = os.stat(src), os.stat(dst)
    except OSError:
        return False
    return a.st_size == b.st_size and int(a.st_mtime) == int(b.st_mtime)


def copy_file(src: str, dst: str) -> None:
    """Copy with the time stamp kept; a half-written file never has the final name."""
    tmp = dst + ".tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def place_videos(items: list, out: str) -> dict:
    """Copy, move and remove the demo folders under <out>/videos so they match the items. Returns counts."""
    stats = {"copied": 0, "bytes": 0, "moved": 0, "removed": 0, "missing": []}
    where = {}  # demo folder name -> share folder it is in now
    for folder in FOLDERS:
        os.makedirs(f"{out}/videos/{folder}", exist_ok=True)
        for d in os.listdir(f"{out}/videos/{folder}"):
            where[d] = folder
    wanted = {layout.demo_name(x["n"]) for x in items}
    for demo, folder in sorted(where.items()):
        if demo not in wanted:
            shutil.rmtree(f"{out}/videos/{folder}/{demo}")
            stats["removed"] += 1
            print(f"removed {folder}/{demo} (no current video)")
    for x in items:
        demo = layout.demo_name(x["n"])
        target = f"{out}/videos/{x['f']}/{demo}"
        if demo in where and where[demo] != x["f"]:
            os.rename(f"{out}/videos/{where[demo]}/{demo}", target)
            stats["moved"] += 1
        os.makedirs(target, exist_ok=True)
        for dst_name, src in x["files"].items():
            dst = f"{target}/{dst_name}"
            if not os.path.isfile(src):
                stats["missing"].append(os.path.relpath(src, start=os.path.dirname(out)))
                continue
            if not same_file(src, dst):
                copy_file(src, dst)
                stats["copied"] += 1
                stats["bytes"] += os.path.getsize(src)
        keep = set(x["files"]) | {f"{demo}_prompts.txt"}
        for f in os.listdir(target):
            if f not in keep:
                os.remove(f"{target}/{f}")
        with open(f"{target}/{demo}_prompts.txt", "w") as f:
            f.write(f"{demo} (prompt version {x['s']})\n\n"
                    f"Reference image prompt (image model: {x['im']}):\n{x['ip']}\n\n"
                    f"Cosmos video prompt (control: edge):\n{x['prompt']}\n")
    return stats


def copy_lists(ds: str, out: str) -> None:
    shutil.rmtree(f"{out}/lists", ignore_errors=True)
    for rel in LISTS:
        src = f"{ds}/{rel}"
        if os.path.isfile(src):
            os.makedirs(os.path.dirname(f"{out}/lists/{rel}"), exist_ok=True)
            shutil.copy2(src, f"{out}/lists/{rel}")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg: #f6f7f9; --card: #ffffff; --text: #1c2128; --dim: #5b6572; --line: #d6dbe1; --accent: #2563eb;
        --ok: #15803d; --weak: #b45309; --bad: #b91c1c; --none: #6b7280; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #14171c; --card: #1d2229; --text: #e6e9ee; --dim: #9aa4b2; --line: #343b46; --accent: #6ea0ff;
          --ok: #4ade80; --weak: #fbbf24; --bad: #f87171; --none: #9ca3af; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { position: sticky; top: 0; z-index: 5; background: var(--bg); border-bottom: 1px solid var(--line); padding: 10px 16px; }
h1 { font-size: 17px; margin: 0 0 2px; }
.sub { color: var(--dim); margin: 0 0 8px; }
.bar { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; }
.group { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
.group > span { color: var(--dim); margin-right: 2px; }
button.f { border: 1px solid var(--line); background: var(--card); color: var(--text); border-radius: 6px; padding: 3px 9px;
           font: inherit; cursor: pointer; }
button.f.on { background: var(--accent); border-color: var(--accent); color: #fff; }
select, input { border: 1px solid var(--line); background: var(--card); color: var(--text); border-radius: 6px; padding: 3px 6px; font: inherit; }
input { width: 110px; }
#count { color: var(--dim); margin-left: auto; }
main { padding: 12px 16px 60px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; margin-bottom: 12px; }
.head { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 6px; }
.head b { font-size: 15px; margin-right: 4px; }
.chip { border-radius: 10px; padding: 1px 8px; font-size: 12px; border: 1px solid var(--line); }
.approved { color: var(--ok); border-color: var(--ok); } .weak { color: var(--weak); border-color: var(--weak); }
.rejected { color: var(--bad); border-color: var(--bad); } .none { color: var(--none); }
.reason { background: transparent; color: var(--dim); }
.media { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
figure { margin: 0; min-width: 0; }
figcaption { color: var(--dim); font-size: 12px; margin-bottom: 2px; }
video, img.ref { width: 100%; aspect-ratio: 2 / 1; display: block; background: #000; border-radius: 4px; }
.scene { color: var(--dim); margin-top: 6px; font-size: 13px; }
.scene b { color: var(--text); font-weight: 600; }
.note { margin-top: 4px; }
.more { margin-top: 4px; font-size: 13px; }
.more a { color: var(--accent); margin-right: 10px; text-decoration: none; }
details { margin-top: 4px; color: var(--dim); font-size: 13px; }
#end { text-align: center; color: var(--dim); padding: 16px; }
@media (max-width: 900px) { .media { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <p class="sub">__SUB__ Left half of each video: room camera. Right half: wrist camera.</p>
  <div class="bar">
    <div class="group" id="g-verdict"><span>Review</span></div>
    <div class="group" id="g-views"><span>Two views</span></div>
    <div class="group" id="g-spec"><span>Version</span></div>
    <div class="group"><span>Reason</span><select id="reason"></select></div>
    <div class="group"><span>Demo</span><input id="demo" placeholder="317 or 300-349"></div>
    <span id="count"></span>
  </div>
</header>
<main id="list"></main>
<div id="end"></div>
<script>
const DATA = __DATA__;
const REASONS = __REASONS__;
const LEVELS = __LEVELS__;
const VERDICT = {approved: "O approved", weak: "△ weak", rejected: "X rejected", "": "not reviewed"};
const state = {verdict: "all", views: "all", spec: "all", reason: "all", demo: ""};
const list = document.getElementById("list"), end = document.getElementById("end");
let shown = [], next = 0;
const pad = n => String(n).padStart(3, "0");
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

function match(x) {
  if (state.verdict !== "all" && (x.v || "none") !== state.verdict) return false;
  if (state.views !== "all" && (x.views || "none") !== state.views) return false;
  if (state.spec !== "all" && x.s !== state.spec) return false;
  if (state.reason !== "all" && !x.r.includes(state.reason)) return false;
  const q = state.demo.trim();
  if (q) {
    const m = q.match(/^(\d+)\s*-\s*(\d+)$/);
    if (m) { if (x.n < +m[1] || x.n > +m[2]) return false; }
    else if (/^\d+$/.test(q)) { if (x.n !== +q) return false; }
  }
  return true;
}

function card(x) {
  const dir = `videos/${x.f}/demo_${pad(x.n)}`, demo = `${dir}/demo_${pad(x.n)}`;
  const el = document.createElement("section");
  el.className = "card";
  const reasons = x.r.map(r => `<span class="chip reason">${esc(REASONS[r] || r)}</span>`).join("")
    + (x.refp ? `<span class="chip reason">reference image problem</span>` : "");
  const views = x.views ? `<span class="chip reason">two views: ${esc(LEVELS[x.views] || x.views)}</span>` : "";
  const scene = [["Place", x.place], ["Table", x.table], ["Lighting", x.light], ["Cloth", x.cloth], ["Floor", x.floor]]
    .filter(p => p[1]).map(p => `<b>${p[0]}:</b> ${esc(p[1])}`).join(" &nbsp;·&nbsp; ");
  el.innerHTML = `
    <div class="head"><b>demo ${pad(x.n)}</b>
      <span class="chip">${esc(x.s)}</span>
      <span class="chip ${x.v || "none"}">${VERDICT[x.v]}</span>${views}${reasons}</div>
    <div class="media">
      <figure><figcaption>Simulator (every step)</figcaption>
        <video muted loop playsinline controls preload="none" poster="${demo}_ref_sim.png" data-src="${demo}_source_all.mp4"></video></figure>
      <figure><figcaption>Cosmos video</figcaption>
        <video muted loop playsinline controls preload="none" poster="${dir}/${x.ref}.png" data-src="${dir}/${x.name}.mp4"></video></figure>
      <figure><figcaption>Reference image (frame 0 of the Cosmos video)</figcaption>
        <a href="${dir}/${x.ref}.png" target="_blank"><img class="ref" loading="lazy" src="${dir}/${x.ref}.png" alt="reference image"></a></figure>
    </div>
    ${scene ? `<div class="scene">${scene}</div>` : ""}
    ${x.note ? `<div class="note"><b>Review note:</b> ${esc(x.note)}</div>` : ""}
    <div class="more">More files:
      <a href="${demo}_geoedge_all.mp4" target="_blank">edge control</a>
      <a href="${demo}_instance_all.mp4" target="_blank">instance</a>
      <a href="${dir}/${x.name}.json" target="_blank">settings (json)</a>
      <a href="${demo}_prompts.txt" target="_blank">prompts (txt)</a>
      <span style="color: var(--dim)">simulator source: ${esc(x.src)}</span></div>
    <details><summary>Reference image prompt</summary>${esc(x.ip)}</details>
    <details><summary>Cosmos video prompt</summary>${esc(x.prompt)}</details>`;
  el.querySelectorAll("video").forEach(v => watch.observe(v));
  return el;
}

// A video loads and plays only while it is on the screen. This keeps the page light with hundreds of videos.
const watch = new IntersectionObserver(entries => {
  for (const e of entries) {
    const v = e.target;
    if (e.isIntersecting) {
      if (!v.getAttribute("src")) v.src = v.dataset.src;
      v.play().catch(() => {});
    } else if (v.getAttribute("src")) {
      v.pause(); v.removeAttribute("src"); v.load();
    }
  }
}, {rootMargin: "150px"});

function more() {
  const stop = Math.min(next + 8, shown.length);
  for (; next < stop; next++) list.appendChild(card(shown[next]));
  end.textContent = next < shown.length ? "Scroll for more" : (shown.length ? "End of the list" : "No video matches the filters");
}
new IntersectionObserver(e => { if (e[0].isIntersecting && next < shown.length) more(); }, {rootMargin: "600px"}).observe(end);

function render() {
  list.querySelectorAll("video").forEach(v => watch.unobserve(v));
  list.innerHTML = ""; next = 0;
  shown = DATA.filter(match);
  document.getElementById("count").textContent = `${shown.length} of ${DATA.length} videos`;
  document.querySelectorAll("button.f").forEach(b => b.classList.toggle("on", state[b.dataset.k] === b.dataset.v));
  history.replaceState(null, "", "#" + new URLSearchParams(state).toString());
  more();
  window.scrollTo(0, 0);
}

function buttons(id, key, values) {
  const g = document.getElementById(id);
  for (const [value, label] of values) {
    const b = document.createElement("button");
    b.className = "f"; b.dataset.k = key; b.dataset.v = value; b.textContent = label;
    b.onclick = () => { state[key] = value; render(); };
    g.appendChild(b);
  }
}
const count = f => DATA.filter(f).length;
buttons("g-verdict", "verdict", [["all", `All ${DATA.length}`], ["approved", `O ${count(x => x.v === "approved")}`],
  ["weak", `△ ${count(x => x.v === "weak")}`], ["rejected", `X ${count(x => x.v === "rejected")}`],
  ["none", `Not reviewed ${count(x => !x.v)}`]]);
buttons("g-views", "views", [["all", "All"], ...Object.entries(LEVELS).map(([k, v]) => [k, `${v} ${count(x => x.views === k)}`])]);
const specs = [...new Set(DATA.map(x => x.s))].sort();
buttons("g-spec", "spec", [["all", "All"], ...specs.map(s => [s, `${s} ${count(x => x.s === s)}`])]);
const sel = document.getElementById("reason");
sel.innerHTML = `<option value="all">All</option>` + Object.entries(REASONS)
  .map(([k, v]) => `<option value="${k}">${esc(v)} (${count(x => x.r.includes(k))})</option>`).join("");
sel.onchange = () => { state.reason = sel.value; render(); };
const box = document.getElementById("demo");
box.oninput = () => { state.demo = box.value; render(); };
for (const [k, v] of new URLSearchParams(location.hash.slice(1))) if (k in state) state[k] = v;
sel.value = state.reason; box.value = state.demo;
render();
</script>
</body>
</html>
"""

README = """# __TITLE__

Cosmos videos of the dataset `__DS__`, sorted by the review result. Updated __WHEN__.

- __N__ videos: __COUNTS__.
- `videos/<O|weak|X|not_reviewed>/demo_NNN/`: the files of one demo. O = approved, weak = usable but weak,
  X = rejected. A video marked weak or rejected without a reason is in `not_reviewed`.
  - `demo_NNN_TT.mp4`: Cosmos video, 1024x512, 20 fps, one frame per simulator step (131 to 133 frames).
    Left half: room camera. Right half: wrist camera. `demo_NNN_TT.json`: settings of the Cosmos run (prompt, seed).
  - `demo_NNN_TT.png`: reference image (frame 0 of the Cosmos video). `demo_NNN_ref_sim.png`: frame 0 of the simulator.
  - `demo_NNN_source_all.mp4`: simulator video with the same frames. `demo_NNN_geoedge_all.mp4`: its edge video,
    the control input of Cosmos. `demo_NNN_instance_all.mp4`: instance mask, one color per object, same frames.
  - `demo_NNN_prompts.txt`: the prompt of the reference image and the prompt of the Cosmos video.
- `lists/`: the list files of the dataset. `review.csv` has the verdict, the reasons and the level of the two views
  of every video. `refs/references.csv` has the prompt and the scene words of every reference image.
- The hdf5 file is not here.

`index.html` shows every video next to its simulator video and its reference image, with filters. To open it from
another computer, start the small web server of this folder and forward its port:

    python3 serve.py 8766        # then open http://localhost:8766

The server listens on this machine only. Several people can use it at the same time.
"""


def write_page(items: list, version: int, out: str, ds: str) -> str:
    title = os.path.basename(os.path.normpath(out)).replace("_", " ")
    n = {f: sum(1 for x in items if x["f"] == f) for f in FOLDERS}
    counts = f"{n['O']} approved (O), {n['weak']} weak (△), {n['X']} rejected (X), {n['not_reviewed']} not reviewed"
    try:
        from zoneinfo import ZoneInfo
        from datetime import datetime
        when = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M %Z")
    except Exception:  # noqa: BLE001 - no time zone data: fall back to the clock of the machine
        when = time.strftime("%Y-%m-%d %H:%M %Z")
    sub = f"{len(items)} Cosmos videos of {os.path.basename(ds)}: {counts}. Updated {when}."
    data = [{k: v for k, v in x.items() if k != "files"} for x in items]
    page = (PAGE.replace("__TITLE__", title).replace("__SUB__", sub)
            .replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
            .replace("__REASONS__", json.dumps(dict(status.reasons_of(version))))
            .replace("__LEVELS__", json.dumps(dict(status.VIEW_LEVELS) if version == 2 else {})))
    with open(f"{out}/index.html", "w") as f:
        f.write(page)
    with open(f"{out}/README.md", "w") as f:
        f.write(README.replace("__TITLE__", title).replace("__DS__", os.path.basename(ds)).replace("__WHEN__", when)
                .replace("__N__", str(len(items))).replace("__COUNTS__", counts))
    shutil.copy2(os.path.join(os.path.dirname(os.path.abspath(__file__)), "share_serve.py"), f"{out}/serve.py")
    return counts


def set_owner(out: str, owner: str) -> None:
    """chown every file and folder of the share to UID:GID; folders 775, files 664."""
    uid, gid = (int(x) for x in owner.split(":"))
    for root, dirs, files in os.walk(out):
        os.chown(root, uid, gid)
        os.chmod(root, 0o775)
        for f in files:
            p = os.path.join(root, f)
            os.chown(p, uid, gid)
            os.chmod(p, 0o664)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("out", help="the share folder (made when missing)")
    ap.add_argument("--owner", help="UID:GID for every file of the share (when the script runs as root)")
    args = ap.parse_args()
    ds, out = os.path.normpath(args.dataset), os.path.normpath(args.out)
    if not status.is_dataset(ds):
        sys.exit(f"{ds} is not a dataset (no sources.csv)")
    t0 = time.time()
    items, version = collect(ds)
    os.makedirs(out, exist_ok=True)
    stats = place_videos(items, out)
    copy_lists(ds, out)
    counts = write_page(items, version, out, ds)
    if args.owner:
        set_owner(out, args.owner)
    print(f"{len(items)} videos | {counts}")
    print(f"copied {stats['copied']} files ({stats['bytes'] / 1e9:.2f} GB) | moved {stats['moved']} demo folders | "
          f"removed {stats['removed']} | missing source files {len(stats['missing'])} | {time.time() - t0:.0f} s")
    for m in stats["missing"][:20]:
        print("  missing:", m)


if __name__ == "__main__":
    main()
