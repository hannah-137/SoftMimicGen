"""Layout check of one reference image. checks/check_references.py calls check() for every image.

Question: do the table, the towel and the robot have the same shape, size and position in the reference image as in
frame 0 of the simulator (sim)? The look (color, material, light) may differ.

Input per camera view (room, wrist), see view(): the sim masks of the table, the towel and the robot (from the
instance ids), the sim RGB frame, the sim normals and the sim geometry edges. Images are BGR (OpenCV order).

The sim mask is only a prior. The check looks for the true boundary in the reference image:
- table: every straight edge of the sim table polygon is moved along its normal (+-40 px) to the clear step line
  NEAREST to the sim edge, not to the strongest step in the window. Step contrast = 8 px Lab window means (or a thin
  rim line), fitted as a straight line per mean offset d. The profile s1(d) is the best line score at every d in
  [-40, 40]. Edge selection rule:
    1. Near line: the best line within +-NEAR_PRIOR (4 px) of the sim edge with score >= C_MIN (20) confirms the
       sim edge. It is moved to a stronger local peak within SECOND_DIST (6 px). A peak within SEP_MARGIN px of it
       is the line itself. The near line is dropped when it is only the foot of a connected line (no valley between
       the two) more than FOOT (2) x stronger, UNLESS it is a step of its own (separate-line test): its contrast
       with a step window too narrow to reach the peak (gap - SEP_MARGIN = 3 px, at least 2 px, per sample, only
       where the two lines are more than 4 px apart) is still >= C_MIN. The 8 px window pulls the near line toward
       the peak. So the separate line is placed again as the parallel line d (|d| <= 6, more than 4 px from the
       peak) with the best narrow score. A flank of the peak has no contrast of its own there and is the same line.
       A real edge next to a stripe (table side face, base rail) stays, and the stripe counts as another line in
       rule 4.
    2. Else the candidates are the local peaks of s1 (over +-PEAK_W = 3 px) with score >= C_MIN. A peak farther than
       NEAR_ZONE (8 px) from the sim edge must stand alone: s1 must fall below PEAK_DROP (0.5) x the peak on both
       sides inside the window. A ramp to the window end or a wide hump of lines (rug, floor pattern, gravel) is
       not a candidate. The nearest candidate is taken.
    3. No near line and no candidate = missing. Half-length test: a tilted line (|a - b| >= SKEW_MIN = 6 px) or a
       line farther than NEAR_ZONE from the sim edge must be a step on both halves of the edge (each half >=
       HALF_MIN = C_MIN on its unblocked samples, at least HALF_N = 5). A line with a step on one half only runs
       from a real edge into the floor texture. It is rejected and the rule runs again on parallel lines (a = b).
       When that line is rejected too, the edge is missing.
    4. Two peaks are two lines only when s1 dips between them below VALLEY (0.85) x the lower one. A plateau or a
       shoulder is the same wide step. Ambiguous (the found line cannot be told from another line): another line
       within AMB_DIST (16 px) with >= SECOND_RATIO (0.8) x the found score; a found line farther than NEAR_ZONE
       from the sim edge with any other line >= 0.8 x its score; a found line within NEAR_ZONE with another line
       more than 1 / NEAR_WEAK = 2 x stronger. A clear line at the sim edge with a stronger line far away is ok.
    5. Stability (per edge): the line found again from the sim polygon grown and shrunk by INV_K (8) px must lie on
       the line of the exact run (p95 <= TABLE_INV_MAX = 3 px), matched edge by edge. An edge whose line moves with
       the prior is "unstable". No evidence at all (no matched edge in either moved run) = not reliable.
  Per edge status: ok / ambiguous / missing / unstable / contact (outside is the towel or the robot: not a table
  edge) / border (at the view border). Ambiguous and unstable edges are not measured. A missing edge with the sim
  background outside (a void edge: the table and the floor of the image can look the same) is not measured either.
  The offset comes from the ok edges. A missing edge with sim ground outside fails, except along the view border (a
  thin ground wedge in the sim that the image fills with table).
  Known limits: a table edge that is not visible (same color as the floor, or only a texture change) next to a
  clear line farther out (a wall base, a cabinet, a tile or rug line) is taken as the table edge when that line stands
  alone. A thin table edge at the border of the wrist view cannot be measured.
- towel: GrabCut with a trimap from the sim mask. The result counts only when it does not move with the prior (run
  again with the prior grown and shrunk by 8 px, the found outline moves <= 2 px). A result that moves with the
  prior is "no boundary found" (not a failure), except when the exact run and a moved run both put the towel more
  than N_PX off and smaller than the sim.
- robot (wrist view only): active contour along the normal (+-16 px) with a smoothness term. Internal edges (sim
  robot on both sides of the step) are blocked. Less than half of the free outline measured = no result.
- interior (table top face, towel): a foreign region drawn inside the object (floor on the table, a label on the
  towel). V1 = classical look split (color + texture) with a straight rim and a two-side difference. V2 = a
  connected region whose chroma differs >= 10 from the ring 4..8 px around it. A patch counts when either fires. On
  the room table only V1 counts (V2 fires on colored light, shadows and reflections there).
- table look: the table top must look the same in the room view and in the wrist view (median Lab color and fine
  texture of the interior). This catches a table drawn as another material in one view.
- floor look (wrist table): the image model can draw the floor of the room view on a part of the wrist table. It
  invents a table edge that the sim does not have, often along the towel, so there is no straight rim to find. The
  room view gives two looks in blocks of 16 px: the table (its interior) and the floor (the sim void beside the
  table, below its back edge). A block of the wrist table is floor-like when it is clearly nearer to the floor look
  than to the table look (lightness after the offset between the views, color, fine texture). The largest
  connected group of such blocks counts. Known limits: a floor that looks like the table, a drawn surface that
  does not look like the room floor, strips thinner than about 20 px, and a floor on more than about half of the
  wrist table when it differs from the table mainly in lightness (the offset between the views then comes from the
  floor).

Pass rule (see check()): the image fails when
  - the towel (both views), the table (both views) or the robot (wrist view) is more than N_PX = 8 px off (p95 of
    the measured boundary, or 2 N_PX + 2 at one place), or
  - a table edge with sim ground outside is missing, or no table edge at all is found in a view with void edges, or
  - a foreign patch covers more than PATCH_MAX = 5 % of the table top or of the towel, or
  - the table top looks different in the two views, or
  - one part of the wrist table, at least FLOOR_MAX = 1.2 % of it and FL_GROUP_MIN = 4 blocks, looks like the floor
    of the room view.
A boundary that cannot be measured is not a failure. Needs numpy and opencv-python.
"""

import cv2
import numpy as np

OBJECTS = ("table", "towel", "robot")
FRONT = {"table": ("towel", "robot"), "towel": ("robot",), "robot": ()}  # objects that may lie on top
VIEWS = ("room", "wrist")

# table edge fit
W_STEP = 8          # px: window on each side of a candidate boundary for the step contrast
S_LINE = 40         # px: search range of a straight polygon edge along its normal
C_MIN = 20.0        # Lab units: a found boundary needs at least this score, else "missing"
TILT = 0.25         # max |a - b| / edge length (about 14 degrees)
POLY_EPS = 2.5      # px: approxPolyDP tolerance
MIN_EDGE = 20       # px: shorter polygon edges are not measured
MIN_SAMPLES = 10    # a polygon edge needs this many free samples
BORDER = 4          # px: samples this close to the view border are not measured
CONTACT = 3         # px: outside neighbor distance for the contact and void tests
GAMMA_LINE = 0.15   # Lab units per px: small cost of the distance from the prior (breaks ties only)
NEAR_PRIOR = 4      # px: a clear line this close to the prior confirms the prior (the nearest clear line)
PEAK_W = 3          # px: a candidate line is a local peak of the score profile over this half width
NEAR_ZONE = 8       # px: a peak this close to the prior needs no isolation test
PEAK_DROP = 0.5     # a far peak stands alone when the profile falls below this share of it on both sides
SECOND_DIST = 6     # px: lines closer than this are the same line (a near line snaps to such a peak)
AMB_DIST = 16       # px: two lines within this distance that are both clear make the edge ambiguous ...
SECOND_RATIO = 0.8  # ... when the second has this share of the found score (also any second peak for a far line)
NEAR_WEAK = 0.5     # a near line is ambiguous only when another separate peak is stronger than score / NEAR_WEAK
FOOT = 2.0          # a near line is the foot of a connected (no valley) line this much stronger: dropped
VALLEY = 0.85       # two peaks are two lines only when the profile between them falls below this share of the lower one
SEP_MARGIN = 3      # px: the separate-line test uses a step window of (gap to the peak - SEP_MARGIN) px, at least 2
SKEW_MIN = 6        # px: a line tilted by at least this (|a - b|) must be a step on both halves of the edge
HALF_MIN = C_MIN    # each half of a tilted or far line needs this score, else the line is rejected (skew)
HALF_N = 5          # valid samples needed in a half for the half test
LONG_EDGE = 40      # px: the max rule uses edges at least this long
RIDGE_R = 3         # px: half width of the thin-line (ridge) term

# robot contour
S_PT = 16           # px: search range of a contour point along its normal
GAMMA_PT = 0.5      # Lab units per px: cost of the distance from the prior
LAMBDA = 1.5        # cost per px of jump between neighbor points (2 px apart)
MAX_JUMP = 4        # max jump between neighbor points
ROBOT_MIN_FRAC = 0.5
ROBOT_MAX_SAT = 0.05

# towel GrabCut
D_IN = 16           # px: prior deeper than this is hard foreground
D_OUT = 64          # px: further outside than this is hard background
PAL_K = 6           # colors per palette (core vs far background) for the band pre-label
PAL_MARGIN = 10     # Lab units: the palette pre-label is used only when one palette is this much nearer
GC_ITERS = 5
TOWEL_CONTACT = 4   # px: outline this close to the robot is skipped
MIN_OUTLINE = 50    # px of outline needed for a boundary measure
MIN_BG = 0.02       # hard background must cover this fraction of the view, else no boundary test
INV_K = 8           # px: invariance test grows and shrinks the prior by this
INV_MAX = 2.0       # px: found outline may move this much (p95) between the runs
TABLE_INV_MAX = 3.0  # px: table lines are fitted on a 1 px offset grid from two priors, so 1 px more

# interior check
INT_ERODE = 8       # px cut off the object (and added around the other objects) before the interior check
INT_SIGMA = 3.0     # blur of the Lab image before the interior check (kills grain)
INT_MIN_FRAC = 0.01  # a patch below this fraction of the interior is ignored (and below 400 px)
PATCH_LINE = 0.5    # V1: fraction of the patch rim on long straight edges needed to flag it
PATCH_DIFF = 3.0    # V1: look difference (robust z units) between a patch and its surroundings
LINE_MIN = 40       # px: shortest straight edge that counts
LINE_R = 12         # px: rim pixels this close to a straight edge count as on it
TOP_DEG = 20        # sim normal within this angle of the main table normal = top face
HUE_T = 10.0        # V2: Lab a,b distance between a patch and the ring around it
CHROMA_T = (0.75, 1.25)  # V2: same hue, chroma may scale by this (highlight, shadow)
CAND_K = 0.6        # V2: candidate pixels differ from the main look by this share of HUE_T
THICK_R = 4         # px: half of a patch must lie deeper than this from the interior border
THICK_FRAC = 0.5
NEAR_FOUND = 10     # px: a candidate mostly within this of the found boundary is the boundary, not a patch
NEAR_FOUND_FRAC = 0.7

# floor look (wrist table against the room view)
FL_BLOCK = 16       # px: side of one block
FL_COVER = 0.6      # a block needs this share of its pixels inside the region
FL_W_L = 0.6        # weight of lightness, after the offset between the views (a and b of Lab count 1)
FL_W_TEX = 8.0      # weight of the log fine texture
FL_SIGMAS = (1.0, 2.0, 4.0)  # px: scales of the fine texture
FL_TEX_EPS = 0.3    # L units added before the log of the texture
FL_KNN = 3          # the distance to a look is the mean over this many nearest blocks of that look
FL_MARGIN = 2.0     # a block is floor-like when it is this much nearer to the floor look than to the table look
FL_REF_MIN = 5      # blocks needed for each look of the room view, else the test does not run
FL_GAP = 8          # px: the floor zone of the room view keeps this distance from the sim objects
FL_GROUP_MIN = 4    # blocks: a smaller group of floor-like blocks does not count

# pass rule
N_PX = 8            # px: a boundary may lie this far from the sim boundary (p95), and 2 * N_PX + 2 at one place
PATCH_MAX = 0.05    # a foreign patch may cover this share of the interior
MIN_AREA = 400      # px: smaller objects are skipped
LOOK_MIN_PX = 2000  # px of interior needed in each view for the interior check, the table look and the floor look tests
LOOK_DL = 20.0      # table look, max difference between the views: lightness (L of Lab, 0..100)
LOOK_DC = 12.0      # ... color (a, b of Lab)
LOOK_DTEX = 0.5     # ... fine texture (log scale)
FLOOR_MAX = 0.012   # one group of floor-like blocks may cover less than this share of the wrist table interior


# ---------------- small helpers ----------------
def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def dilate(m, r):
    return cv2.dilate(m.astype(np.uint8), disk(r)).astype(bool) if r > 0 else m.copy()


def erode(m, r):
    return cv2.erode(m.astype(np.uint8), disk(r)).astype(bool) if r > 0 else m.copy()


def erode_in(m, r):
    """Erode with the view border treated as outside (cv2.erode treats it as inside)."""
    return erode(np.pad(m, r), r)[r:-r, r:-r]


def fill_holes(m):
    ff = np.pad(m, 1).astype(np.uint8)
    cv2.floodFill(ff, None, (0, 0), 2)
    return ff[1:-1, 1:-1] != 2


def outline(m):
    """1 px inner boundary. The view border counts as outside."""
    padded = np.pad(m, 1).astype(np.uint8)
    inner = cv2.erode(padded, np.ones((3, 3), np.uint8))
    return (padded - inner)[1:-1, 1:-1].astype(bool)


def dist_to(b):
    """Distance from every pixel to the nearest True pixel of b."""
    if not b.any():
        return np.full(b.shape, 1e4, np.float32)
    return cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 5)


def components(m, min_area):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    return [(lab == i) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def seg_match(pts, nrms, segs, seg_nrms, tol):
    """Index of the segment (whose normal agrees with the point's normal, angle <= about 20 degrees)
    within tol of every point, -1 when there is none: the edge the point belongs to."""
    if len(pts) == 0 or not segs:
        return np.full(len(pts), -1)
    s = np.stack(segs).astype(np.float64)  # (m, 2, 2)
    a, b = s[:, 0], s[:, 1]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(axis=1), 1e-9)
    ap = pts[:, None, :].astype(np.float64) - a[None]
    u = np.clip((ap * ab[None]).sum(axis=2) / L2[None], 0.0, 1.0)
    d = np.linalg.norm(ap - u[..., None] * ab[None], axis=2)
    dot = nrms.astype(np.float64) @ np.stack(seg_nrms).astype(np.float64).T  # same outward direction
    d[dot < 0.94] = np.inf
    j = d.argmin(axis=1)
    return np.where(d[np.arange(len(pts)), j] <= tol, j, -1)


def lab_of(img):
    return cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32), (0, 0), 1.0)


def others_of(masks, name):
    o = np.zeros_like(masks[name])
    for k in OBJECTS:
        if k != name:
            o |= masks[k]
    return o


# ---------------- profiles along normals ----------------
def profiles(lab, pts, nrm, S, w=W_STEP):
    """Lab values along the normal of every point. pts, nrm: (n, 2) as (x, y). Returns (n, 2S+2w+1, 3)."""
    t = np.arange(-S - w, S + w + 1, dtype=np.float32)
    mx = (pts[:, 0:1] + t[None] * nrm[:, 0:1]).astype(np.float32)
    my = (pts[:, 1:2] + t[None] * nrm[:, 1:2]).astype(np.float32)
    prof = cv2.remap(lab, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return prof, mx, my


def step_scores(prof, w=W_STEP):
    """Step contrast at every offset t in [-S, S]: |mean(outside w px) - mean(inside w px)|."""
    c = np.concatenate([np.zeros_like(prof[:, :1]), np.cumsum(prof, axis=1)], axis=1)
    T = prof.shape[1] - 2 * w
    i = np.arange(T) + w
    inside = (c[:, i] - c[:, i - w]) / w
    outside = (c[:, i + w + 1] - c[:, i + 1]) / w
    return np.linalg.norm(outside - inside, axis=2)


def ridge_scores(prof, w=W_STEP, r=RIDGE_R):
    """Thin-line term: |Lab at t - mean of Lab at t-r and t+r|. A drawn rim between two equal colors
    gives a small step contrast but a clear ridge."""
    T = prof.shape[1] - 2 * w
    i = np.arange(T) + w
    return np.linalg.norm(prof[:, i] - 0.5 * (prof[:, i - r] + prof[:, i + r]), axis=2)


def edge_scores(prof):
    return np.maximum(step_scores(prof), ridge_scores(prof))


def blocked(bm, mx, my, S, w=W_STEP):
    """True where the moved point is blocked or outside the image."""
    h, w_ = bm.shape
    sl = slice(w, w + 2 * S + 1)
    x, y = mx[:, sl], my[:, sl]
    out = (x < 0) | (x > w_ - 1) | (y < 0) | (y > h - 1)
    xi, yi = np.clip(np.round(x).astype(int), 0, w_ - 1), np.clip(np.round(y).astype(int), 0, h - 1)
    return out | bm[yi, xi]


def contours_of(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c[:, 0, :] for c in cs if cv2.contourArea(c) > 200]


def outward(pts, nrm, mask):
    """Flip normals so they point out of the mask. A probe point outside the view counts as outside."""
    h, w = mask.shape
    x = np.round(pts[:, 0] + CONTACT * nrm[:, 0]).astype(int)
    y = np.round(pts[:, 1] + CONTACT * nrm[:, 1]).astype(int)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    inside[inside] = mask[y[inside], x[inside]]
    nrm = nrm.copy()
    nrm[inside] *= -1
    return nrm


def sample_flags(pts, nrm, masks, name, void):
    """Per sample: at the view border, a front object right outside (contact), sim black outside (void)."""
    h, w = masks[name].shape
    xo, yo = pts[:, 0] + CONTACT * nrm[:, 0], pts[:, 1] + CONTACT * nrm[:, 1]
    border = ~((pts[:, 0] >= BORDER) & (pts[:, 0] <= w - 1 - BORDER)
               & (pts[:, 1] >= BORDER) & (pts[:, 1] <= h - 1 - BORDER))
    # the outside step window (W_STEP px) must fit in the view, else the edge cannot be judged there
    xw, yw = pts[:, 0] + W_STEP * nrm[:, 0], pts[:, 1] + W_STEP * nrm[:, 1]
    border |= ~((xw >= 0) & (xw <= w - 1) & (yw >= 0) & (yw <= h - 1))
    x, y = np.clip(np.round(xo).astype(int), 0, w - 1), np.clip(np.round(yo).astype(int), 0, h - 1)
    contact = np.zeros(len(pts), bool)
    for f in FRONT[name]:
        contact |= masks[f][y, x]
    isvoid = void[y, x] if void is not None else np.zeros(len(pts), bool)
    return border, contact, isvoid


# ---------------- table: straight edges ----------------
def line_profile(st, u, L, tilt=TILT):
    """Score of every straight line t_i = a + (b - a) u_i over a grid of (a, b), |a - b| <= tilt * L,
    and the best line per mean offset d = (a + b) / 2. Returns off, sc (raw grid), pen (grid with the
    small distance cost, used only to pick a line among equals), s1 (raw score of the best line per d),
    ab (flat grid index of that line per d)."""
    S = (st.shape[1] - 1) // 2
    off = np.arange(-S, S + 1)
    a, b = off[:, None, None], off[None, :, None]
    idx = np.round(a + (b - a) * u[None, None, :]).astype(int) + S
    sc = st[np.arange(len(u))[None, None, :], idx].mean(axis=2)
    sc[np.abs(off[:, None] - off[None, :]) > tilt * L] = -1
    pen = sc - GAMMA_LINE * (np.abs(off[:, None]) + np.abs(off[None, :])) / 2
    d = np.round((off[:, None] + off[None, :]) / 2).astype(int).ravel() + S
    order = np.argsort(pen.ravel(), kind="stable")  # ascending: the last write per d is its max
    s1 = np.full(2 * S + 1, -1.0)
    ab = np.zeros(2 * S + 1, int)
    s1[d[order]] = sc.ravel()[order]
    ab[d[order]] = order
    return off, sc, pen, s1, ab


def peaks_of(s1, S):
    """Local peaks of the profile (over +-PEAK_W) with score >= C_MIN, as (d, score, isolated).
    Isolated: the profile falls below PEAK_DROP x the peak on both sides inside the window."""
    out = []
    for i in range(PEAK_W, 2 * S + 1 - PEAK_W):
        v = s1[i]
        if v < C_MIN or v < s1[i - PEAK_W:i + PEAK_W + 1].max() or (i > 0 and s1[i - 1] >= v):
            continue
        iso = s1[:i + 1].min() <= PEAK_DROP * v and s1[i:].min() <= PEAK_DROP * v
        out.append((i - S, float(v), bool(iso)))
    return out


def apart(s1, S, d0, score, p):
    """True when peak p is another line than the line at d0: the profile dips between the two (below
    VALLEY x the lower one). A flat plateau or a shoulder is the same wide step."""
    lo, hi = sorted((d0, p[0]))
    return s1[lo + S:hi + S + 1].min() < VALLEY * min(score, p[1])


def line_score(stw, u, a, b):
    """Mean score of the line t_i = a + (b - a) u_i on one score matrix (n, 2S+1)."""
    S = (stw.shape[1] - 1) // 2
    idx = np.clip(np.round(a + (b - a) * u).astype(int) + S, 0, 2 * S)
    return float(stw[np.arange(len(u)), idx].mean())


def half_scores(st, valid, u, a, b):
    """Score of the line on the first and the second half of the edge (valid samples only; nan when a
    half has fewer than HALF_N valid samples)."""
    S = (st.shape[1] - 1) // 2
    idx = np.clip(np.round(a + (b - a) * u).astype(int) + S, 0, 2 * S)
    ii = np.arange(len(u))
    v, s = valid[ii, idx], st[ii, idx]
    out = []
    for m in (u <= 0.5, u > 0.5):
        m = m & v
        out.append(float(s[m].mean()) if m.sum() >= HALF_N else float("nan"))
    return out


def fit_line(st, u, L, st_n=None, valid=None):
    """The clear line nearest to the prior (see the module docstring, edge selection rule).
    st_n: {w: score matrix with a w px step window} for the separate-line test; valid: (n, 2S+1) False
    where a sample is blocked (an object on top) for the half-length test. Returns a, b, score, d (mean
    offset), fit (ok / ambiguous / missing), prior score, the strongest other line (second, second_a,
    second_b), the candidate list and skew (1 when a tilted or far line was rejected by the half test)."""
    first = None
    for tilt in (TILT, 0.0):  # second pass: parallel lines only, after a skewed line was rejected
        out = select_line(st, u, L, st_n, tilt)
        rejected = False
        if out["fit"] != "missing" and valid is not None:
            # a tilted line, or a line away from the sim edge, must be a step along the whole edge
            if abs(out["a"] - out["b"]) >= SKEW_MIN or abs(out["d"]) > NEAR_ZONE:
                h = half_scores(st, valid, u, out["a"], out["b"])
                out.update(h1=h[0], h2=h[1])
                rejected = not np.isnan(h).any() and min(h) < HALF_MIN
        if not rejected:
            break
        first = first or out
        out = None
    if out is None:  # both passes rejected: no line along the whole edge
        out = dict(first, fit="missing", skew=1)
    elif first is not None:
        out["skew"] = 1
    return out


def select_line(st, u, L, st_n, tilt):
    """One pass of the edge selection rule on the line grid with |a - b| <= tilt * L."""
    S = (st.shape[1] - 1) // 2
    off, sc, pen, s1, ab = line_profile(st, u, L, tilt)
    prior = float(sc[S, S])
    peaks = peaks_of(s1, S)
    cands = [p for p in peaks if abs(p[0]) <= NEAR_ZONE or p[2]]

    def line_of(d):
        k = np.unravel_index(ab[d + S], sc.shape)
        return int(off[k[0]]), int(off[k[1]])

    def own_score(a0, b0, p):
        """Step contrast of the line (a0, b0) measured with a window too narrow to reach the line of
        peak p (per sample: gap - SEP_MARGIN px, at least 2), on the samples where the two lines are
        more than SEP_MARGIN + 1 px apart. A flank (foot) of p has no contrast of its own there.
        -1 when the lines cross or run together on more than half of the samples (the same line)."""
        if st_n is None:
            return -1.0
        ap, bp = line_of(p[0])
        t0 = a0 + (b0 - a0) * u
        gap = np.abs(ap + (bp - ap) * u - t0)
        keep = gap > SEP_MARGIN + 1
        if keep.sum() < max(HALF_N, 0.5 * len(u)):
            return -1.0
        w2 = np.clip(np.round(gap - SEP_MARGIN).astype(int), 2, W_STEP)
        idx = np.clip(np.round(t0).astype(int) + S, 0, 2 * S)
        ii = np.arange(len(u))
        v = np.array([st_n[w2[i]][i, idx[i]] for i in ii[keep]])
        return float(v.mean())

    def separate(a0, b0, p):
        """True when the line (a0, b0) is a step of its own next to peak p (own_score >= C_MIN)."""
        return own_score(a0, b0, p) >= C_MIN

    def relocate(p):
        """The line of its own next to a stronger peak p, if any: the wide window pulls the near line
        toward the peak, so the parallel lines d = a = b over the near zone (not within SEP_MARGIN + 1 px
        of the peak) are scored with the narrow window and the best one is returned as
        (d, a, b, own score), or None when none reaches C_MIN."""
        best = None
        for d in range(-NEAR_PRIOR - 2, NEAR_PRIOR + 3):
            if abs(d - p[0]) <= SEP_MARGIN + 1:
                continue
            v = own_score(d, d, p)
            if v >= C_MIN and (best is None or v > best[3]):
                best = (d, d, d, v)
        return best

    # 1. a clear line within NEAR_PRIOR confirms the prior; it snaps to a stronger peak within SECOND_DIST
    #    unless it is a step of its own (separate): then it stays and that peak counts as another line
    near = np.abs(off) <= NEAR_PRIOR
    sub = np.where(near[:, None] & near[None, :], pen, -2)
    k = np.unravel_index(np.argmax(sub), sc.shape)
    chosen, sep = None, []
    if sc[k] >= C_MIN:
        a0, b0, score = int(off[k[0]]), int(off[k[1]]), float(sc[k])
        d0 = int(round((a0 + b0) / 2))
        snap = [p for p in peaks if abs(p[0] - d0) <= SECOND_DIST and p[1] > score]
        if snap:
            p = max(snap, key=lambda q: q[1])
            if abs(p[0] - d0) <= SEP_MARGIN:  # the peak itself: the same line
                r = None
            else:
                r = relocate(p) if not separate(a0, b0, p) else (d0, a0, b0, score)
            if r is not None:  # a step of its own next to the peak: stays, the peak is another line
                d0, a0, b0, score = r
                sep.append(p)
            else:
                d0, score = p[:2]
                a0, b0 = line_of(d0)
        # a near line on the flank of a connected (no valley between) line more than FOOT times stronger
        # is the foot of that line, not a line of its own: dropped, the candidate peaks decide. A line
        # with a step of its own (separate) is kept.
        feet = [p for p in peaks
                if p[1] > FOOT * score and abs(p[0] - d0) > SECOND_DIST and not apart(s1, S, d0, score, p)]
        foot = False
        for p in feet:
            r = relocate(p) if not separate(a0, b0, p) else (d0, a0, b0, score)
            if r is not None:
                d0, a0, b0, score = r
                sep.append(p)
            else:
                foot = True
        if not foot:
            chosen = (d0, score, a0, b0)
    # 2. else the nearest candidate peak
    if chosen is None and cands:
        d0, score, _ = min(cands, key=lambda p: (abs(p[0]), -p[1]))
        chosen = (d0, score) + line_of(d0)
    out = {"a": 0, "b": 0, "d": 0, "score": float(s1.max()), "prior": prior, "fit": "missing", "skew": 0,
           "second": -1.0, "second_a": 0, "second_b": 0, "cands": [list(p) for p in peaks]}
    if chosen is None:
        return out
    d0, score, a0, b0 = chosen
    out.update(a=a0, b=b0, d=d0, score=score, fit="ok")
    # 4. ambiguity against every other peak that is another line (see apart), and the separate peaks
    others = [p for p in peaks if abs(p[0] - d0) > SECOND_DIST and apart(s1, S, d0, score, p)]
    others += [p for p in sep if p not in others and abs(p[0] - d0) > SEP_MARGIN]
    if others:
        d2, s2, _ = max(others, key=lambda p: p[1])
        a2, b2 = line_of(d2)
        out.update(second=s2, second_a=a2, second_b=b2)
        double = any(abs(p[0] - d0) <= AMB_DIST and p[1] >= SECOND_RATIO * score for p in others)
        if abs(d0) > NEAR_ZONE:
            amb = double or s2 >= SECOND_RATIO * score
        else:
            amb = double or s2 >= score / NEAR_WEAK
        if amb:
            out["fit"] = "ambiguous"
    return out


def border_room(pts, nrm, shape, S):
    """Median distance from the samples along the outward normal to the view border (clipped at S)."""
    h, w = shape
    t = np.full(len(pts), float(S))
    for i, lim in ((0, w - 1), (1, h - 1)):
        n = nrm[:, i]
        pos, neg = n > 1e-6, n < -1e-6
        t[pos] = np.minimum(t[pos], (lim - pts[pos, i]) / n[pos])
        t[neg] = np.minimum(t[neg], pts[neg, i] / -n[neg])
    return float(np.clip(np.median(t), 0, S))


def fit_edge(lab, pts, nrm, u, L, bm, S):
    """Score profile along the normals of the free samples of one edge, then the best straight line.
    Also the profiles with narrow step windows (2..W_STEP px) for the separate-line test."""
    prof, mx, my = profiles(lab, pts, nrm, S)
    blk = blocked(bm, mx, my, S)
    st = edge_scores(prof)
    st[blk] = 0
    ridge = ridge_scores(prof)
    st_n = {}
    for w in range(2, W_STEP + 1):
        # the step window of w px is cut from the profile of W_STEP px on each side (same offsets)
        c = W_STEP - w
        s = np.maximum(step_scores(prof[:, c:prof.shape[1] - c], w), ridge)
        s[blk] = 0
        st_n[w] = s
    return fit_line(st, u, L, st_n, ~blk)


def move_vertices(poly, shifts, mask):
    """Move the polygon vertices: intersection of the two moved edge lines, or the mean move when nearly parallel."""
    n = len(poly)
    new = poly.copy()
    for k in range(n):
        e0, e1 = (k - 1) % n, k
        q0, q1, q2 = poly[e0], poly[k], poly[(k + 1) % n]
        lines = []
        for (pa, pb, (a, b), at_end) in ((q0, q1, shifts[e0], True), (q1, q2, shifts[e1], False)):
            L = np.linalg.norm(pb - pa)
            if L < 1e-6:
                lines.append(None)
                continue
            d = (pb - pa) / L
            nr = outward(np.array([(pa + pb) / 2]), np.array([[-d[1], d[0]]], np.float32), mask)[0]
            off = b if at_end else a
            lines.append((pa + a * nr, pb + b * nr, off * nr))
        if lines[0] is None or lines[1] is None:
            continue
        (p, q, m0), (r, s_, m1) = lines
        d0, d1 = q - p, s_ - r
        cross = d0[0] * d1[1] - d0[1] * d1[0]
        if abs(cross) / (np.linalg.norm(d0) * np.linalg.norm(d1) + 1e-9) > 0.35:  # angle > 20 deg
            tt = ((r - p)[0] * d1[1] - (r - p)[1] * d1[0]) / cross
            new[k] = p + tt * d0
        else:
            new[k] = q1 + (m0 + m1) / 2
    return new


def table_edges(lab, masks, void, name="table", S=S_LINE, trim=0):
    """Move every straight edge of the sim table polygon to the true boundary.
    Returns the edge list, the found polygon mask and, for the ok edges, the sample points with their offsets
    (without the first and last `trim` px of each edge)."""
    mask = masks[name]
    front = np.zeros_like(mask)
    for f in FRONT[name]:
        front |= masks[f]
    bm = dilate(front, W_STEP + 2)  # the step window must not reach into an object on top
    edges, polys, prior_segs, found_segs, prior_nrm, prior_k = [], [], [], [], [], []
    found_gate, gate_samples, gate_nrm, gate_k, t_gate = [], [], [], [], []
    for c in contours_of(mask):
        poly = cv2.approxPolyDP(c.reshape(-1, 1, 2), POLY_EPS, True)[:, 0, :].astype(np.float32)
        n = len(poly)
        shifts = np.zeros((n, 2))
        for k in range(n):
            p0, p1 = poly[k], poly[(k + 1) % n]
            L = float(np.linalg.norm(p1 - p0))
            if L < MIN_EDGE:
                continue
            d = (p1 - p0) / L
            m = int(L)
            s = (np.arange(m) + 0.5) * (L / m)
            pts = p0[None] + s[:, None] * d[None]
            nrm = outward(pts, np.repeat(np.array([[-d[1], d[0]]], np.float32), m, 0), mask)
            nrm0 = nrm[0]
            border, contact, isvoid = sample_flags(pts, nrm, masks, name, void)
            free = ~border & ~contact
            e = {"k": k, "len": L, "a": 0, "b": 0, "score": 0.0, "second": 0.0, "void": False}
            if free.sum() < MIN_SAMPLES:
                e["status"] = "contact" if contact.sum() >= border.sum() else "border"
                edges.append(e)
                continue
            e["void"] = bool(isvoid[free].mean() > 0.5)
            if not e["void"]:
                free &= ~isvoid  # an edge with sim ground outside is fitted on its non-void samples only
                if free.sum() < MIN_SAMPLES:
                    e["status"] = "border"
                    edges.append(e)
                    continue
            f = fit_edge(lab, pts[free], nrm[free], s[free] / L, L, bm, S)
            e.update(f, status=f["fit"], room=border_room(pts[free], nrm[free], mask.shape, S))
            a, b = f["a"], f["b"]
            if f["fit"] == "ok":
                shifts[k] = (a, b)
                t = a + (b - a) * s[free] / L
                fp = pts[free] + t[:, None] * nrm0[None]
                mid = (s[free] >= trim) & (s[free] <= L - trim)
                found_gate.append(fp[mid])
                gate_samples.append(pts[free][mid])
                gate_nrm.append(np.repeat(nrm0[None], int(mid.sum()), 0))
                gate_k.append(np.full(int(mid.sum()), k))
                prior_nrm.append(nrm0)
                prior_k.append(k)
                prior_segs.append(np.stack([p0, p1]))
                found_segs.append(np.stack([p0 + a * nrm0, p1 + b * nrm0]))
                t_gate.append(t)
            edges.append(e)
        polys.append(move_vertices(poly, shifts, mask))
    found = cv2.fillPoly(np.zeros(mask.shape, np.uint8), [np.round(p).astype(np.int32) for p in polys], 1).astype(bool)
    cat = lambda xs, shape: np.concatenate(xs) if xs else np.zeros(shape, np.float32)  # noqa: E731
    return {"edges": edges, "found": found, "found_gate": cat(found_gate, (0, 2)),
            "gate_samples": cat(gate_samples, (0, 2)), "gate_nrm": cat(gate_nrm, (0, 2)), "prior_nrm": prior_nrm,
            "gate_k": np.concatenate(gate_k) if gate_k else np.zeros(0, int),
            "prior_segs": prior_segs, "found_segs": found_segs, "prior_k": prior_k, "t_gate": cat(t_gate, 0)}


def offset_polygon(mask, k):
    """Move every straight edge of the mask polygon by k px along its outward normal (corners stay sharp)."""
    polys = []
    for c in contours_of(mask):
        poly = cv2.approxPolyDP(c.reshape(-1, 1, 2), POLY_EPS, True)[:, 0, :].astype(np.float32)
        n = len(poly)
        lines = []
        for i in range(n):
            p0, p1 = poly[i], poly[(i + 1) % n]
            d = p1 - p0
            d /= max(np.linalg.norm(d), 1e-6)
            nr = outward(np.array([(p0 + p1) / 2]), np.array([[-d[1], d[0]]], np.float32), mask)[0]
            lines.append((p0 + k * nr, d, nr))
        new = poly.copy()
        for i in range(n):
            (p, d0, n0), (r, d1, n1) = lines[i - 1], lines[i]
            cross = d0[0] * d1[1] - d0[1] * d1[0]
            if abs(cross) > 0.35:
                tt = ((r - p)[0] * d1[1] - (r - p)[1] * d1[0]) / cross
                new[i] = p + tt * d0
            else:
                new[i] = poly[i] + k * (n0 + n1) / 2
        polys.append(np.round(new).astype(np.int32))
    return cv2.fillPoly(np.zeros(mask.shape, np.uint8), polys, 1).astype(bool)


def grow_masks(masks, name, k):
    """Grow (k > 0) or shrink (k < 0) one object by |k| px. Objects on top stay on top and stay adjacent
    (only the outer boundary moves). table, towel: polygon edges move (corners stay sharp). robot: round."""
    out = {n: m.copy() for n, m in masks.items()}
    front = np.zeros_like(masks[name])
    for f in FRONT[name]:
        front |= masks[f]
    if name == "robot":
        new = dilate(masks[name], k) if k > 0 else erode(masks[name], -k)
    else:
        new = offset_polygon(masks[name], k)
        if k < 0:
            new |= masks[name] & dilate(front, -k + 2)  # keep the ring next to the objects on top
    new &= ~front
    if k > 0:
        for n in out:
            if n != name and n not in FRONT[name]:
                out[n] &= ~new
    else:
        freed = masks[name] & ~new
        both = (masks["table"] | masks["towel"]).astype(np.uint8)
        filled = cv2.morphologyEx(both, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
        if name != "table":
            out["table"] |= freed & filled
    out[name] = new
    return out


def check_table(lab, masks, void):
    """Table of one view: edge fit and stability of every edge against the prior. Returns the numbers for failures()
    (with the edge list) and the found table region."""
    r = table_edges(lab, masks, void)
    edges = r["edges"]
    found = r["found"] & ~others_of(masks, "table")
    by_k = {e["k"]: e for e in edges}
    # Stability: the lines found again from the sim polygon grown and shrunk by INV_K px must lie on the lines of
    # the exact run, edge by edge (p95 <= TABLE_INV_MAX; the INV_K + 4 px at the edge ends are left out because the
    # moved polygon has other corners). A moved sample point within INV_K + 3 px of an exact-run found edge (its
    # sim position, same outward normal) belongs to that edge. An edge whose line moves with the prior is
    # "unstable": not measured. An edge the exact run did not find gives no evidence, nor does a moved run with no
    # matched sample. No evidence at all = not reliable.
    move_of = {}
    if len(r["found_gate"]) >= MIN_SAMPLES:
        for k in (INV_K, -INV_K):
            m2 = grow_masks(masks, "table", k)
            if m2["table"].sum() < MIN_AREA:
                continue
            r2 = table_edges(lab, m2, void, trim=INV_K + 4)
            if len(r2["found_gate"]) < MIN_SAMPLES:
                continue
            match = seg_match(r2["gate_samples"], r2["gate_nrm"], r["prior_segs"], r["prior_nrm"], INV_K + 3)
            for j, (seg, kj) in enumerate(zip(r["found_segs"], r["prior_k"])):
                pts = r2["found_gate"][match == j]
                if len(pts) < MIN_SAMPLES:
                    continue
                d = seg[1] - seg[0]
                cross = (pts[:, 0] - seg[0][0]) * d[1] - (pts[:, 1] - seg[0][1]) * d[0]
                dist = np.abs(cross) / max(np.linalg.norm(d), 1e-6)
                move_of[kj] = max(move_of.get(kj, 0.0), float(np.percentile(dist, 95)))
    unstable = {k for k, m in move_of.items() if m > TABLE_INV_MAX}
    for k, m in move_of.items():
        by_k[k]["move"] = m
    for k in unstable:
        by_k[k]["fit"] = by_k[k]["status"] = "unstable"
    free_len = sum(e["len"] for e in edges if e["status"] not in ("contact", "border"))
    meas_len = sum(e["len"] for e in edges if e.get("fit") == "ok")
    n_ok = sum(e["status"] == "ok" for e in edges)
    row = {"edges": edges, "measured_frac": meas_len / free_len if free_len else 0.0, "n_ok": n_ok,
           "n_gated": sum(e["status"] not in ("contact", "border") for e in edges)}
    # offsets over the stable ok edges only; the max rule uses the long edges when there is one
    gk = r["gate_k"]
    keep = np.array([k not in unstable for k in gk], bool)
    tg = r["t_gate"][keep]
    tl = tg[np.array([by_k[k]["len"] >= LONG_EDGE for k in gk], bool)[keep]]
    row["off_p95"] = float(np.percentile(np.abs(tg), 95)) if len(tg) else float("nan")
    row["off_max"] = float(np.abs(tl if len(tl) else tg).max()) if len(tg) else float("nan")
    row["move"] = float(max(move_of.values())) if move_of else float("nan")
    row["reliable"] = bool(move_of) and n_ok > 0
    return row, found


# ---------------- robot: active contour ----------------
def robot_internal(robot_mask, geo_lines, sim_view):
    """Internal edges of the sim robot (link edges from the geoedge frame, material edges from the sim RGB),
    at least 4 px inside the silhouette. They are part of the prior: the reference robot has them too."""
    lines = np.zeros(robot_mask.shape, bool)
    if geo_lines is not None:
        lines |= geo_lines
    if sim_view is not None:
        lines |= cv2.Canny(cv2.cvtColor(sim_view, cv2.COLOR_BGR2GRAY), 50, 150) > 0
    deep = cv2.erode(robot_mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    return lines & deep


def resample_contour(c, stride=2):
    d = np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)
    L = np.concatenate([[0], np.cumsum(d)])
    s = np.arange(0, L[-1], stride)
    x = np.interp(s, L, np.append(c[:, 0], c[0, 0]))
    y = np.interp(s, L, np.append(c[:, 1], c[0, 1]))
    return np.stack([x, y], 1).astype(np.float32)


def contour_normals(pts, k=6):
    nxt, prv = np.roll(pts, -k, 0), np.roll(pts, k, 0)
    d = nxt - prv
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
    return np.stack([-d[:, 1], d[:, 0]], 1).astype(np.float32)


def dp_contour(st, lam=LAMBDA, max_jump=MAX_JUMP):
    """Smooth offsets along a contour: maximize sum st[i, t_i] - lam * sum |t_i - t_{i-1}|, |jump| <= max_jump."""
    n, T = st.shape
    dj = np.abs(np.arange(T)[:, None] - np.arange(T)[None, :])
    jump = np.where(dj <= max_jump, lam * dj, 1e9)
    best, back = np.zeros((n, T)), np.zeros((n, T), int)
    best[0] = st[0]
    for i in range(1, n):
        cand = best[i - 1][None, :] - jump
        back[i] = cand.argmax(axis=1)
        best[i] = cand.max(axis=1) + st[i]
    t = np.zeros(n, int)
    t[-1] = best[-1].argmax()
    for i in range(n - 1, 0, -1):
        t[i - 1] = back[i, t[i]]
    return t - (T - 1) // 2


def robot_contour(lab, masks, internal, S=S_PT):
    """Move every robot contour point along its normal, smooth with a DP along the contour."""
    mask = masks["robot"]
    bm = cv2.dilate(internal.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    # the step window must not reach into the towel (21 x 21 block), the towel lies under the robot
    bm |= cv2.dilate(masks["towel"].astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool) & ~mask
    polys, t_all, meas_all, free_all = [], [], [], []
    h, w = mask.shape
    for c in contours_of(mask):
        pts = resample_contour(c.astype(np.float32))
        if len(pts) < 20:
            continue
        nrm = outward(pts, contour_normals(pts), mask)
        border, contact, _ = sample_flags(pts, nrm, masks, "robot", None)
        ok = ~border & ~contact
        prof, mx, my = profiles(lab, pts, nrm, S)
        st = step_scores(prof)
        st[blocked(bm, mx, my, S)] = 0
        # a step whose whole outside window still lies inside the sim robot is an internal edge (a link
        # or gripper edge), not the silhouette: blocked
        sl = slice(W_STEP, W_STEP + 2 * S + 1)
        xo = np.clip(np.round(mx[:, sl] + (W_STEP + 1) * nrm[:, 0:1]).astype(int), 0, w - 1)
        yo = np.clip(np.round(my[:, sl] + (W_STEP + 1) * nrm[:, 1:2]).astype(int), 0, h - 1)
        st[mask[yo, xo]] = 0
        st[~ok] = 0
        weak = st.max(axis=1) < C_MIN
        st[weak] = 0
        t = dp_contour(st - GAMMA_PT * np.abs(np.arange(-S, S + 1))[None, :])
        measured = ok & ~weak
        polys.append(pts + t[:, None] * nrm)
        t_all.append(t)
        meas_all.append(measured)
        free_all.append(ok)
    found = cv2.fillPoly(np.zeros(mask.shape, np.uint8), [np.round(p).astype(np.int32) for p in polys], 1).astype(bool)
    cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt)  # noqa: E731
    return {"found": found, "t": cat(t_all, int), "measured": cat(meas_all, bool), "free": cat(free_all, bool)}


def check_robot(lab, masks, internal):
    """Robot of one view. Returns the numbers for failures() and the found robot region."""
    r = robot_contour(lab, masks, internal)
    m = r["measured"]
    t_meas = r["t"][m]
    sat = np.abs(t_meas) >= S_PT  # the search range ended there: no boundary found, counted apart
    t = t_meas[~sat]
    n_free = int(r["free"].sum())
    row = {"measured_frac": float(m.sum() / n_free) if n_free else 0.0,
           "off_p95": float(np.percentile(np.abs(t), 95)) if len(t) else float("nan"),
           "off_max": float(np.abs(t).max()) if len(t) else float("nan"),
           "saturated_frac": float(sat.mean()) if len(t_meas) else float("nan")}
    return row, r["found"]


# ---------------- towel: GrabCut ----------------
def palette(lab, mask, k=PAL_K):
    pts = lab[mask].astype(np.float32)
    if len(pts) < 50:
        return None
    if len(pts) > 20000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 20000, replace=False)]
    cv2.setRNGSeed(0)  # k-means++ seeds come from the cv2 RNG: fix it, so a rerun gives the same result
    stop = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, cen = cv2.kmeans(pts, min(k, len(pts)), None, stop, 2, cv2.KMEANS_PP_CENTERS)
    return cen


def nearest_dist(lab, cen):
    d = np.full(lab.shape[:2], np.inf, np.float32)
    for c in cen:
        d = np.minimum(d, np.sqrt(((lab - c) ** 2).sum(axis=2)))
    return d


def dist_in(m):
    """Distance of each mask pixel to the mask boundary. The image border is not a boundary."""
    p = np.pad(m.astype(np.uint8), 8, mode="edge")
    return cv2.distanceTransform(p, cv2.DIST_L2, 5)[8:-8, 8:-8]


def segment(img, fg_prior, hard_bg, d_in=D_IN, d_out=D_OUT):
    """GrabCut with a trimap from the prior. Hard foreground = prior deeper than d_in px. Hard background =
    further than d_out px outside the prior, and hard_bg. The band between is pre-labeled by the sim mask,
    overruled by the nearer Lab palette (core vs far background) where that is clear.
    Returns found (holes filled) and a limit-hit flag (found outline on the outer limit)."""
    din, dout = dist_in(fg_prior), dist_in(~fg_prior)
    far = ~fg_prior & ~dilate(hard_bg, 4) & (dout > d_out)
    if far.sum() < MIN_BG * far.size:
        return None, False
    core = fg_prior & (din > d_in)
    if core.sum() < 200:
        core = fg_prior & (din >= 0.5 * din.max())
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    band = ~core & ~far & ~hard_bg
    fg_pal, bg_pal = palette(lab, core), palette(lab, far | hard_bg)
    tri = np.full(img.shape[:2], cv2.GC_BGD, np.uint8)
    tri[band & fg_prior] = cv2.GC_PR_FGD
    tri[band & ~fg_prior] = cv2.GC_PR_BGD
    if fg_pal is not None and bg_pal is not None:
        gap = nearest_dist(lab, bg_pal) - nearest_dist(lab, fg_pal)
        tri[band & (gap > PAL_MARGIN)] = cv2.GC_PR_FGD
        tri[band & (gap < -PAL_MARGIN)] = cv2.GC_PR_BGD
    tri[core] = cv2.GC_FGD
    gc = tri.copy()
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img, gc, None, bgd, fgd, GC_ITERS, cv2.GC_INIT_WITH_MASK)
    raw = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    keep = np.zeros_like(raw)
    for c in components(raw, 1):
        if (c & core).any():
            keep |= c
    found = fill_holes(keep)
    ob = outline(found)
    hit = (ob & (dout >= d_out - 1.5)).sum() / max(ob.sum(), 1)
    return found, bool(hit >= 0.01)


def boundary_offsets(prior, found, skip):
    """Distances prior outline -> found outline and back (p95, max), and the direction of the parts that are off
    (+ = found is larger than the prior there)."""
    op, of = outline(prior), outline(found)
    bp, bf = op & ~skip, of & ~skip  # counted pixels; distances go to the full outlines
    if bp.sum() < MIN_OUTLINE or bf.sum() < MIN_OUTLINE:
        return None
    d_pf, d_fp = dist_to(of)[bp], dist_to(op)[bf]
    sign = np.where(found[bp], 1.0, -1.0)
    off = d_pf > NEAR_PRIOR
    return dict(p95_prior=float(np.percentile(d_pf, 95)), max_prior=float(d_pf.max()),
                p95_found=float(np.percentile(d_fp, 95)), max_found=float(d_fp.max()),
                direction=float((sign[off] * d_pf[off]).mean()) if off.any() else 0.0)


def towel_region(img, masks, prior=None):
    """GrabCut towel with the robot cut out."""
    fg = masks["towel"] if prior is None else prior
    found, limit = segment(img, fg, np.zeros_like(fg))
    if found is not None:
        found &= ~masks["robot"]
    return found, limit


def check_towel(img, masks):
    """Towel of one view. Returns the numbers for failures() and the found towel region (None when the view has too
    little background for GrabCut)."""
    fg = masks["towel"]
    skip = dilate(masks["robot"], TOWEL_CONTACT)
    found, limit = towel_region(img, masks)
    nan = float("nan")
    row = {"off_p95": nan, "off_max": nan, "off_dir": nan, "reliable": False, "leak": bool(limit),
           "run_p95": [], "run_dir": []}
    if found is None:
        return row, None
    off = boundary_offsets(fg, found, skip)
    if off:
        row.update(off_dir=off["direction"], off_p95=max(off["p95_prior"], off["p95_found"]),
                   off_max=max(off["max_prior"], off["max_found"]))
    # Stability: run again with the prior grown and shrunk by INV_K px. The found outline must stay.
    # A moved run whose outline collapsed (fewer than MIN_OUTLINE px) counts as an infinite move.
    # Each moved run also keeps its own offset and direction for failures(): a towel drawn smaller or larger moves
    # with the prior but keeps its offset direction in every run.
    moves = []
    bf0 = outline(found) & ~skip
    for k in (INV_K, -INV_K):
        prior = (dilate(fg, k) if k > 0 else erode(fg, -k)) & ~masks["robot"]
        if prior.sum() < MIN_AREA:
            moves.append(nan)
            continue
        f2, _ = towel_region(img, masks, prior)
        if f2 is None:
            moves.append(nan)
            continue
        bf2 = outline(f2) & ~skip
        moves.append(max(np.percentile(dist_to(bf2)[bf0], 95), np.percentile(dist_to(bf0)[bf2], 95))
                     if bf0.any() and bf2.any() else float("inf"))
        off2 = boundary_offsets(fg, f2, skip)
        if off2:
            row["run_p95"].append(max(off2["p95_prior"], off2["p95_found"]))
            row["run_dir"].append(off2["direction"])
    move = float(np.nanmax(moves)) if not all(np.isnan(moves)) else nan
    row["reliable"] = off is not None and not np.isnan(move) and move <= INV_MAX
    return row, found


# ---------------- interior check ----------------
def top_face(normals, table):
    """Table pixels whose sim normal is close to the main table normal (the top face)."""
    nv = normals.astype(np.float32) / 127.5 - 1.0
    tin = erode(table, 3)
    if tin.sum() < 100:
        return np.ones(table.shape, bool)
    tn = np.median(nv[tin], axis=0)
    tn /= max(np.linalg.norm(tn), 1e-6)
    cosang = (nv * tn).sum(axis=2) / np.maximum(np.linalg.norm(nv, axis=2), 1e-6)
    return cosang > np.cos(np.radians(TOP_DEG))


def features_v1(img):
    """Per-pixel look: smoothed Lab color and fine texture energy (two scales)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    f = [cv2.GaussianBlur(lab[..., i], (0, 0), INT_SIGMA) for i in range(3)]
    for sg in (1.0, 2.0):
        hp = L - cv2.GaussianBlur(L, (0, 0), sg)
        f.append(np.log(np.sqrt(cv2.GaussianBlur(hp ** 2, (0, 0), 6)) + 1.0))
    return np.stack(f, axis=2)


def straight_edges(img, region):
    """Pixels within LINE_R px of a long straight image edge inside the region (Hough on Canny)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    e = cv2.Canny(cv2.GaussianBlur(lab[..., 0], (0, 0), 1.5), 30, 90)
    for i in (1, 2):
        e |= cv2.Canny(cv2.GaussianBlur(lab[..., i], (0, 0), 1.5), 15, 45)
    e[~erode(region, 2)] = 0
    lines = cv2.HoughLinesP(e, 1, np.pi / 180, threshold=30, minLineLength=LINE_MIN, maxLineGap=4)
    lm = np.zeros(region.shape, np.uint8)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(lm, (int(x1), int(y1)), (int(x2), int(y2)), 1, 1)
    return dilate(lm.astype(bool), LINE_R)


def interior_v1(img, interior):
    """V1: k-means (k = 2) on color + texture. The minor look is a patch when its rim runs along long straight
    sharp lines and its look differs from the ring around it. Returns the patch masks."""
    F = features_v1(img)
    vals = F[interior]
    floor = np.array([2.0, 1.0, 1.0, 0.1, 0.1])
    med = np.median(vals, axis=0)
    mad = np.maximum(1.4826 * np.median(np.abs(vals - med), axis=0), floor)
    zi = ((vals - med) / mad).astype(np.float32)
    zi[:, 0] *= 0.5  # lightness counts half: shading and highlights are allowed
    cv2.setRNGSeed(0)
    stop = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    _, lbl, _ = cv2.kmeans(zi, 2, None, stop, 3, cv2.KMEANS_PP_CENTERS)
    lbl = lbl.ravel()
    minor = int(np.argmin(np.bincount(lbl, minlength=2)))
    vmaj = vals[lbl != minor]
    med2 = np.median(vmaj, axis=0)
    mad2 = np.maximum(1.4826 * np.median(np.abs(vmaj - med2), axis=0), floor)
    z = (F - med2) / mad2
    z[..., 0] *= 0.5
    cand = np.zeros_like(interior)
    cand[interior] = lbl == minor
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, disk(3)).astype(bool)
    lines = straight_edges(img, interior)
    out = []
    for c in components(cand, max(MIN_AREA, INT_MIN_FRAC * interior.sum())):
        rim = outline(c) & erode_in(interior, 3)
        if rim.sum() < 20:
            continue
        line = float(lines[rim].mean())
        inside, outside = erode(c, 4) & interior, dilate(c, 8) & ~dilate(c, 4) & interior
        if inside.sum() < 50 or outside.sum() < 50:
            continue
        diff = float(np.linalg.norm(np.median(z[inside], axis=0) - np.median(z[outside], axis=0)))
        if line >= PATCH_LINE and diff >= PATCH_DIFF:
            out.append(c)
    return out


def chroma_dist(ab, ab0):
    """Distance of chroma ab to the reference chroma ab0 scaled by t in CHROMA_T (same hue, other chroma = 0)."""
    n0 = float(np.linalg.norm(ab0))
    if n0 < 1e-3:
        return np.linalg.norm(ab, axis=-1)
    t = np.clip((ab * ab0).sum(axis=-1) / (n0 * n0), CHROMA_T[0], CHROMA_T[1])
    return np.linalg.norm(ab - t[..., None] * ab0, axis=-1)


def interior_hue(img, interior):
    """V2 (hue): candidate pixels differ in blurred Lab a, b from the interior median. A connected candidate
    (>= max(400 px, 1 % of the interior), not a thin band) is a patch when its inner part differs >= HUE_T in
    chroma from the ring 4..8 px outside it. No brightness, no rim rule. Returns the patch masks."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab = np.stack([cv2.GaussianBlur(lab[..., i] - 128.0, (0, 0), INT_SIGMA) for i in (1, 2)], axis=2)
    med = np.median(ab[interior], axis=0)
    dist = np.linalg.norm(ab - med, axis=2)
    cand = (interior & (dist >= CAND_K * HUE_T)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, disk(3)).astype(bool)
    main = interior & ~cand
    deep = erode_in(interior, THICK_R)
    out = []
    for c in components(cand, max(MIN_AREA, INT_MIN_FRAC * interior.sum())):
        if (c & deep).sum() < THICK_FRAC * c.sum():
            continue  # a thin band along the interior border
        inside = erode(c, 4) & interior
        ring = dilate(c, 8) & ~dilate(c, 4) & main
        if inside.sum() < 50:
            inside = c
        if ring.sum() < 50:
            ring = main if main.sum() >= 50 else interior & ~c
        if ring.sum() < 50:
            continue
        if float(chroma_dist(np.median(ab[inside], axis=0), np.median(ab[ring], axis=0))) >= HUE_T:
            out.append(c)
    return out


def interior_of(masks, name, normals):
    """Interior of an object for the patch and the look tests: the object cut by INT_ERODE px, without INT_ERODE px
    around the other objects. Table: the top face only."""
    region = masks[name] & top_face(normals, masks[name]) if name == "table" else masks[name]
    return erode(region, INT_ERODE) & ~dilate(others_of(masks, name), INT_ERODE)


def interior_check(img, interior, found_outline, use_hue=True):
    """Share of the interior of an object that shows a foreign patch. A patch that mostly lies within NEAR_FOUND px
    of the found boundary is the boundary offset, not a patch. With use_hue False (room table) only V1 counts: V2
    fires on colored light, shadows and reflections there."""
    if interior.sum() < LOOK_MIN_PX:
        return 0.0
    near = dilate(found_outline, NEAR_FOUND)
    patch = np.zeros_like(interior)
    for c in interior_v1(img, interior) + (interior_hue(img, interior) if use_hue else []):
        if (c & near).sum() < NEAR_FOUND_FRAC * c.sum():
            patch |= c
    return float(patch.sum() / interior.sum())


def look(img, m):
    """Look of image img inside m: median L (0..100), median a and b of Lab, and the fine texture (log of the
    median |high-pass L| over the median L)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    hp = np.abs(L - cv2.GaussianBlur(L, (0, 0), 2.0))
    tex = float(np.log(100.0 * np.median(hp[m]) / max(np.median(L[m]), 8.0) + 1.0))
    a, b = float(np.median(lab[..., 1][m]) - 128), float(np.median(lab[..., 2][m]) - 128)
    return float(np.median(L[m]) / 2.55), a, b, tex


# ---------------- floor look ----------------
def floor_zone(masks, void):
    """Where the room view shows the floor: sim void below the top row of the table, FL_GAP px away from the sim
    objects."""
    rows = np.where(masks["table"].any(axis=1))[0]
    if len(rows) == 0:
        return np.zeros_like(void)
    zone = void.copy()
    zone[: rows.min()] = False
    return zone & ~dilate(masks["table"] | masks["towel"] | masks["robot"], FL_GAP)


def pixel_looks(img):
    """Per pixel: L (0..100), a and b of Lab, and the fine texture |L - blur(L)| at each scale of FL_SIGMAS."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] / 2.55
    tex = [np.abs(L - cv2.GaussianBlur(L, (0, 0), s)) for s in FL_SIGMAS]
    return np.stack([L, lab[..., 1] - 128.0, lab[..., 2] - 128.0] + tex, axis=2)


def block_looks(F, m):
    """Median look of every block of FL_BLOCK px with at least FL_COVER of its pixels in m.
    Returns (block row and column, looks)."""
    cells, looks = [], []
    for y in range(0, m.shape[0], FL_BLOCK):
        for x in range(0, m.shape[1], FL_BLOCK):
            inside = m[y:y + FL_BLOCK, x:x + FL_BLOCK]
            if inside.mean() >= FL_COVER:
                cells.append((y // FL_BLOCK, x // FL_BLOCK))
                looks.append(np.median(F[y:y + FL_BLOCK, x:x + FL_BLOCK][inside], axis=0))
    return np.array(cells, int).reshape(-1, 2), np.array(looks, np.float32).reshape(-1, F.shape[2])


def look_vectors(looks, L0):
    """Block looks as weighted vectors. L0 is the main lightness of the view: it takes out the light offset
    between the two views."""
    tex = FL_W_TEX * np.log(looks[:, 3:] + FL_TEX_EPS)
    return np.concatenate([FL_W_L * (looks[:, :1] - L0), looks[:, 1:3], tex], axis=1)


def look_dist(a, ref):
    """Mean distance of each row of a to its FL_KNN nearest rows of ref."""
    d = np.linalg.norm(a[:, None] - ref[None], axis=2)
    return np.sort(d, axis=1)[:, :min(FL_KNN, len(ref))].mean(axis=1)


def floor_look(room, room_view, room_interior, wrist, wrist_interior):
    """Share of the wrist table interior that looks like the floor of the room view, and the mask of that part.
    room, wrist: the two halves of the reference image. The room view gives the table look (room_interior) and
    the floor look (floor_zone). A wrist block is floor-like when it is FL_MARGIN nearer to the floor look than to
    the table look. The share is the largest connected group of floor-like blocks over all wrist blocks.
    Returns (0.0, None) when a look cannot be measured, or when no group has FL_GROUP_MIN blocks."""
    Fw, Fr = pixel_looks(wrist), pixel_looks(room)
    cells, wrist_looks = block_looks(Fw, wrist_interior)
    _, table = block_looks(Fr, room_interior)
    _, floor = block_looks(Fr, floor_zone(room_view["masks"], room_view["void"]))
    if wrist_interior.sum() < LOOK_MIN_PX or len(wrist_looks) == 0 or min(len(table), len(floor)) < FL_REF_MIN:
        return 0.0, None
    L_room, L_wrist = float(np.median(table[:, 0])), float(np.median(wrist_looks[:, 0]))
    w = look_vectors(wrist_looks, L_wrist)
    hit = look_dist(w, look_vectors(table, L_room)) - look_dist(w, look_vectors(floor, L_room)) > FL_MARGIN
    grid = np.zeros((cells[:, 0].max() + 1, cells[:, 1].max() + 1), np.uint8)
    grid[cells[hit, 0], cells[hit, 1]] = 1
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    if n < 2:
        return 0.0, None
    k = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    if stats[k, cv2.CC_STAT_AREA] < FL_GROUP_MIN:
        return 0.0, None
    region = np.zeros(wrist_interior.shape, bool)
    for r, c in cells[labels[cells[:, 0], cells[:, 1]] == k]:
        region[r * FL_BLOCK:(r + 1) * FL_BLOCK, c * FL_BLOCK:(c + 1) * FL_BLOCK] = True
    return float(stats[k, cv2.CC_STAT_AREA] / len(wrist_looks)), region & wrist_interior


# ---------------- pass rule ----------------
def off_limit(row):
    """True when a measured boundary is too far from the sim boundary."""
    return row["off_p95"] > N_PX or row["off_max"] > 2 * N_PX + 2


def failures(name, row):
    """Why one object of one view fails, as short texts. An empty list = ok, or not measured."""
    out = []
    if name == "robot":
        if row["measured_frac"] < ROBOT_MIN_FRAC or np.isnan(row["off_p95"]):
            return out  # too little of the free outline measured: no result
        if row["saturated_frac"] >= ROBOT_MAX_SAT:
            out.append("outline not found")
        if off_limit(row):
            out.append(f"outline {row['off_p95']:.0f} px off")
        return out
    patch = f"another surface on {100 * row['patch_frac']:.0f} % of it" if row["patch_frac"] > PATCH_MAX else ""
    if name == "table":
        if patch:
            out.append(patch)
        if row["n_gated"] == 0:
            return out
        missing = [e for e in row["edges"] if e["status"] == "missing"]
        # A missing edge with sim ground outside fails, except along the view border (the border is within the
        # search range): that is a thin ground wedge of the sim that the image fills with table.
        if any(not e["void"] and e["room"] >= S_LINE for e in missing):
            out.append("edge missing")
        # A missing void edge alone is not a failure. No measured edge at all in such a view is one.
        if row["measured_frac"] == 0 and any(e["void"] for e in missing):
            out.append("no edge found")
        if not np.isnan(row["off_p95"]) and off_limit(row):
            out.append(f"edge {row['off_p95']:.0f} px off")
        return out
    if np.isnan(row["off_p95"]):  # towel, no boundary to measure
        return [patch] if patch else out
    if row["reliable"]:
        if patch:
            out.append(patch)
        if off_limit(row):
            out.append(f"outline {row['off_p95']:.0f} px off")
        return out
    # Not reliable (the outline moved with the prior): no boundary found, except when the exact run and a moved run
    # both put the towel more than N_PX off and SMALLER than the sim. A towel drawn smaller leaves table pixels
    # inside the hard foreground, which is what makes GrabCut move with the prior. A larger result that moves with
    # the prior is a leak on a table of the same color. leak = the GrabCut region reached the outer limit.
    if patch:
        out.append(patch + ", or the towel is smaller")
    if (not row["leak"] and row["off_p95"] > N_PX and row["off_dir"] < 0
            and any(p > N_PX and d < 0 for p, d in zip(row["run_p95"], row["run_dir"]))):
        out.append(f"smaller than in the simulator ({row['off_p95']:.0f} px)")
    return out


# ---------------- one image ----------------
def view(sim, masks, void, normals, edges):
    """Input of one camera view, all from frame 0 of the demo and all (H, W).
    sim: RGB frame as BGR uint8. masks: {"table", "towel", "robot"} -> bool. void: bool, True where the sim shows
    nothing (instance id 0). normals: uint8, (xyz + 1) * 127.5. edges: bool, geometry edges of the sim."""
    return {"masks": masks, "void": void, "normals": normals, "internal": robot_internal(masks["robot"], edges, sim)}


def check(ref, views):
    """Check one reference image. ref: (H, 2 W, 3) BGR, room | wrist. views: {"room": view(...), "wrist": view(...)}.
    Returns (reasons, found). reasons: why the image fails, an empty list = pass. found: {(view, object): region
    found in the image, or None} for draw(), and ("wrist", "floor"): the part of the wrist table that looks like the
    room floor, when the image fails for it."""
    W = ref.shape[1] // 2
    reasons, found, looks, tables = [], {}, {}, {}
    for i, vname in enumerate(VIEWS):
        v, img = views[vname], np.ascontiguousarray(ref[:, i * W:(i + 1) * W])
        masks, lab = v["masks"], lab_of(img)
        for name in OBJECTS:
            if (name == "robot" and vname != "wrist") or masks[name].sum() < MIN_AREA:
                continue  # the room robot is not checked
            if name == "table":
                row, region = check_table(lab, masks, v["void"])
            elif name == "towel":
                row, region = check_towel(img, masks)
            else:
                row, region = check_robot(lab, masks, v["internal"])
            found[(vname, name)] = region
            if name != "robot":
                interior = interior_of(masks, name, v["normals"])
                border = region if (region is not None and row["reliable"]) else masks[name]
                use_hue = not (name == "table" and vname == "room")
                row["patch_frac"] = interior_check(img, interior, outline(border), use_hue)
                if name == "table":
                    tables[vname] = (img, interior)
                if name == "table" and interior.sum() >= LOOK_MIN_PX:
                    looks[vname] = look(img, interior)
            fails = failures(name, row)
            if fails:
                reasons.append(f"{vname} {name}: " + ", ".join(fails))
    if len(looks) == 2:  # the table top must look the same in both views
        (L0, a0, b0, t0), (L1, a1, b1, t1) = looks["room"], looks["wrist"]
        dL, dC, dT = abs(L0 - L1), float(np.hypot(a0 - a1, b0 - b1)), abs(t0 - t1)
        if dL > LOOK_DL or dC > LOOK_DC or dT > LOOK_DTEX:
            reasons.append(f"table: room and wrist views differ (lightness {dL:.0f}, color {dC:.0f}, texture {dT:.2f})")
    if len(tables) == 2:  # floor drawn on the wrist table
        share, region = floor_look(tables["room"][0], views["room"], tables["room"][1], *tables["wrist"])
        if share >= FLOOR_MAX:
            reasons.append(f"wrist table: looks like the room floor on {100 * share:.1f} % of it")
            found[("wrist", "floor")] = region
    return reasons, found


def draw(ref, views, found):
    """The reference image with the sim outlines in red and the found outlines in green. The part of the wrist
    table that looks like the room floor is outlined in blue."""
    out = ref.copy()
    W = ref.shape[1] // 2
    for i, vname in enumerate(VIEWS):
        part = out[:, i * W:(i + 1) * W]
        for name in OBJECTS:
            part[outline(views[vname]["masks"][name])] = (0, 0, 255)
        for name in OBJECTS:
            region = found.get((vname, name))
            if region is not None:
                part[outline(region)] = (0, 255, 0)
        floor = found.get((vname, "floor"))
        if floor is not None:
            part[outline(floor)] = (255, 0, 0)
    return out
