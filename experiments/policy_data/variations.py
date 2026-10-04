"""Variations of the reference images. The lists and the prompts of each spec version are in specs/<version>.json
(see SPEC.md, change history). The rules of the dataset (place type shares, strong light slots) are here and are the
same for every version, so the shares stay exact when a dataset mixes versions.

Every reference image gets one value on each axis of its spec (v2: place, table, lighting, robot, towel color,
towel material and towel pattern). All axes change in every image. Generation rules:
- place: first a place type, then a place of that type from the spec. The types are exact in every group of BLOCK
  (50) demos, the same groups as the demo folders (000-049, 050-099, ...): 10 outdoor (OUTDOOR_SHARE, 20%) and 4 of
  each of the 10 indoor types (8% each). The 50 types of a group are shuffled over its demos. Places inside a type
  have the same chance.
- lighting: indoor lighting for an indoor place, outdoor lighting for an outdoor place. In every group of BLOCK
  demos, STRONG_PER_BLOCK (2, 4%) of the indoor demos get strong colored ambient light instead.
- other axes: every item has the same chance.
- no repeat: draw() never returns a combination whose key() is in `used`.
The place type and the strong light slot depend on the demo, the seed and the tag only, not on the attempt or the
spec version. A new attempt for the same image (make_references.py --retry) keeps them and draws the rest again.
Each tag (01, 02, ...) has its own shuffle.

    spec = load("v2")                # specs/v2.json; a path to a json file also works (for tests)
    v = spec.draw(demo=2, seed=0)    # same demo, seed, attempt and tag -> same choice
    spec.sentence(v)                 # the axis sentences for the image prompt
    spec.image_prompt(v)             # the whole image prompt
    spec.cosmos_prompt(v)            # the Cosmos prompt with the towel sentence of this image
    spec.key(v)                      # tuple of the axis values

Spec file (json): "version" (the name, the same as the file name), "changes" (what is new in this version),
"image" (model, quality, size and prompt of the image API; the prompt has {variation} once, where the axis sentences
go), "axes" (the list of axes, in order), "sentence" (the axis sentences, with {<axis name>}), "cosmos" (prompt, and
sentence: filled from the axes and put after the first sentence of the prompt; "" for none), "card" (one line for the
review page, with {<axis name>}). Axis "place" has "places" (place type -> list, every type of PLACE_TYPES), axis
"lighting" has "indoor" ("{source}, {color}, {level}" with the lists indoor_sources, indoor_colors, indoor_levels),
"strong" ("{color}" with strong_colors) and "outdoor" (a list). Every other axis has "values": a list, or, with
"by": "<an earlier axis>", one list per value of that axis (v3: the thicknesses that fit each material; v6: the
floors that fit each place, "by": "place"). "note" is free text. The draw order is lighting first, then the other
axes in order (as in v1 and v2).
"""

import glob
import json
import os
import random

SPECS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specs")
CONTROLS = ("edge", "depth")  # control videos that run_cosmos.py can give to Cosmos (key cosmos.controls)
OUTDOOR_SHARE = 0.2
BLOCK = 50  # the place type shares are exact in every group of BLOCK demos
STRONG_PER_BLOCK = 2  # indoor demos with strong colored light in every group of BLOCK demos (4%)
# The place types, in the order of the shuffle. Every spec has places for each of them.
PLACE_TYPES = ["laboratory", "workshop", "factory_warehouse", "office", "school", "home", "medical_care",
               "shop_restaurant_hotel", "studio_venue_sports", "other_indoor", "outdoor"]


def block_types() -> list:
    """The place types of one group of BLOCK demos: OUTDOOR_SHARE outdoor, the rest split equally over the indoor
    types."""
    indoor = [t for t in PLACE_TYPES if t != "outdoor"]
    n_out = round(BLOCK * OUTDOOR_SHARE)
    n_in = (BLOCK - n_out) // len(indoor)
    if n_out + n_in * len(indoor) != BLOCK:
        raise ValueError(f"BLOCK {BLOCK} cannot hold exact shares for {len(indoor)} indoor types")
    return ["outdoor"] * n_out + [t for t in indoor for _ in range(n_in)]


def place_type(demo: int, seed: int, tag: str = "") -> str:
    """The place type of a demo: its slot in the shuffled types of its group of BLOCK demos."""
    types = block_types()
    random.Random(f"{seed}:{tag}:group{demo // BLOCK}").shuffle(types)
    return types[demo % BLOCK]


def strong_light(demo: int, seed: int, tag: str = "") -> bool:
    """True for the STRONG_PER_BLOCK indoor slots of the group that get strong colored light."""
    group = demo // BLOCK
    types = block_types()
    random.Random(f"{seed}:{tag}:group{group}").shuffle(types)
    indoor = [i for i, t in enumerate(types) if t != "outdoor"]
    return demo % BLOCK in random.Random(f"{seed}:{tag}:group{group}:strong").sample(indoor, STRONG_PER_BLOCK)


class Spec:
    """One spec version: the axis lists, the image prompt and the Cosmos prompt."""

    def __init__(self, data: dict, path: str):
        self.path = path
        self.version = data["version"]
        self.image = data["image"]
        self.cosmos = data["cosmos"]
        self.controls = data["cosmos"].get("controls", ["edge"])
        self.template = data["sentence"]
        self.card_template = data.get("card", "")
        self.axes = {a["name"]: a for a in data["axes"]}
        self.fields = [a["name"] for a in data["axes"]]

    def key(self, v: dict) -> tuple:
        """The axis values that identify one combination."""
        return tuple(v[f] for f in self.fields)

    def row_key(self, row: dict) -> tuple:
        """key() of a row of references.csv (empty for an axis the row does not have)."""
        return tuple(row.get(f) or "" for f in self.fields)

    def draw(self, demo: int, seed: int, used: set | None = None, attempt: int = 0, tag: str = "") -> dict:
        """One variation for a demo: the axes plus "place_type". Same demo, seed, attempt and tag give the same
        choice. A combination whose key() is in `used` is skipped, so two images never share a combination."""
        ptype = place_type(demo, seed, tag)
        strong = strong_light(demo, seed, tag)
        light, places = self.axes["lighting"], self.axes["place"]["places"]
        rng = random.Random(f"{seed}:{tag}:{demo}:{attempt}")
        for _ in range(1000):
            if ptype == "outdoor":
                lighting = rng.choice(light["outdoor"])
            elif strong:
                lighting = light["strong"].format(color=rng.choice(light["strong_colors"]))
            else:
                lighting = light["indoor"].format(source=rng.choice(light["indoor_sources"]),
                                                  color=rng.choice(light["indoor_colors"]),
                                                  level=rng.choice(light["indoor_levels"]))
            v = {"place_type": ptype}
            for f in self.fields:
                if f == "lighting":
                    v[f] = lighting
                elif f == "place":
                    v[f] = rng.choice(places[ptype])
                elif "by" in self.axes[f]:  # the list depends on the value of an earlier axis
                    v[f] = rng.choice(self.axes[f]["values"][v[self.axes[f]["by"]]])
                else:
                    v[f] = rng.choice(self.axes[f]["values"])
            if used is None or self.key(v) not in used:
                return v
        raise RuntimeError("no unused combination found")

    def sentence(self, v: dict) -> str:
        """The axis sentences for the image prompt."""
        return self.template.format(**v)

    def image_prompt(self, v: dict) -> str:
        return self.image["prompt"].replace("{variation}", self.sentence(v))

    def cosmos_prompt(self, v: dict) -> str:
        """The Cosmos prompt, with the sentence of the spec (filled from v) after its first sentence."""
        prompt, sentence = self.cosmos["prompt"], self.cosmos.get("sentence", "")
        if not sentence:
            return prompt
        head, sep, tail = prompt.partition(". ")
        text = sentence.format(**v)
        return f"{head}. {text} {tail}" if sep else f"{prompt} {text}"

    def card(self, v: dict) -> str:
        """The line for the review page ("" when v does not have the axes)."""
        try:
            return self.card_template.format(**v)
        except (KeyError, IndexError, ValueError):
            return ""


def versions() -> list:
    """The version names in specs/."""
    return sorted(os.path.basename(p)[: -len(".json")] for p in glob.glob(f"{SPECS_DIR}/*.json"))


_loaded = {}


def load(spec: str) -> Spec:
    """A spec by version name (v2 -> specs/v2.json) or by the path of a json file (for tests; its version must not
    be the name of a file in specs/). Raises ValueError with the reason when the file is missing or wrong."""
    by_path = spec.endswith(".json")
    path = os.path.abspath(spec if by_path else os.path.join(SPECS_DIR, f"{spec}.json"))
    if path in _loaded:
        return _loaded[path]
    if not os.path.isfile(path):
        raise ValueError(f"spec {spec}: {path} not found (versions: {', '.join(versions()) or 'none'})")
    try:
        with open(path) as f:
            data = json.load(f)
        s = Spec(data, path)
        _check(s)
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise ValueError(f"spec {spec} ({path}): {type(e).__name__}: {e}") from None
    if not by_path and s.version != spec:
        raise ValueError(f"spec {spec}: {path} says version {s.version!r}")
    if by_path and s.version in versions() and path != os.path.join(SPECS_DIR, f"{s.version}.json"):
        raise ValueError(f"{path}: version {s.version!r} is the name of specs/{s.version}.json: use another name")
    _loaded[path] = s
    return s


def _check(s: Spec) -> None:
    """Check a spec file: every part is there and every template uses known names only."""
    if not isinstance(s.version, str) or not s.version:
        raise ValueError("version must be a name")
    for k in ("model", "quality", "size", "prompt"):
        if not isinstance(s.image.get(k), str) or not s.image[k]:
            raise ValueError(f"image.{k} is missing")
    if s.image["prompt"].count("{variation}") != 1:
        raise ValueError("image.prompt needs {variation} exactly once")
    if "place" not in s.axes or "lighting" not in s.axes or len(s.axes) != len(s.fields):
        raise ValueError("axes need place and lighting, and each name once")
    places = s.axes["place"].get("places", {})
    if set(places) != set(PLACE_TYPES) or not all(places[t] for t in PLACE_TYPES):
        raise ValueError(f"axis place needs places for each of {', '.join(PLACE_TYPES)}")
    light = s.axes["lighting"]
    for k in ("indoor_sources", "indoor_colors", "indoor_levels", "strong_colors", "outdoor"):
        if not isinstance(light.get(k), list) or not light[k]:
            raise ValueError(f"axis lighting needs the list {k}")
    light["indoor"].format(source="s", color="c", level="l")
    light["strong"].format(color="c")
    for i, f in enumerate(s.fields):
        if f in ("place", "lighting"):
            continue
        values, by = s.axes[f].get("values"), s.axes[f].get("by")
        if by is None:
            if not isinstance(values, list) or not values:
                raise ValueError(f"axis {f} needs a list of values")
            continue
        keys = [x for xs in places.values() for x in xs] if by == "place" else s.axes.get(by, {}).get("values")
        if by not in s.fields[:i] or not isinstance(keys, list):
            raise ValueError(f"axis {f}: by must name the place axis or an earlier axis with a list of values")
        if not isinstance(values, dict) or set(values) != set(keys) or not all(values.values()):
            raise ValueError(f"axis {f} needs a list of values for each value of {by}")
    names = {f: f for f in s.fields} | {"place_type": "place_type"}
    for where, text in (("sentence", s.template), ("cosmos.sentence", s.cosmos.get("sentence", "")),
                        ("card", s.card_template)):
        try:
            text.format(**names)
        except KeyError as e:
            raise ValueError(f"{where} uses {{{e.args[0]}}}, which is not an axis") from None
    if not isinstance(s.cosmos.get("prompt"), str) or not s.cosmos["prompt"]:
        raise ValueError("cosmos.prompt is missing")
    if (not isinstance(s.controls, list) or "edge" not in s.controls or len(set(s.controls)) != len(s.controls)
            or not set(s.controls) <= set(CONTROLS)):
        raise ValueError(f"cosmos.controls must be a list with edge, from {', '.join(CONTROLS)}")


if __name__ == "__main__":
    import sys

    s = load(sys.argv[1] if len(sys.argv) > 1 else "v2")
    n_places = {t: len(p) for t, p in s.axes["place"]["places"].items()}
    print(f"spec {s.version} ({s.path})")
    print(f"places {sum(n_places.values())} {n_places}")
    for f in s.fields:
        if f not in ("place", "lighting"):
            values = s.axes[f]["values"]
            print(f"{f} {len(values)}" if isinstance(values, list)
                  else f"{f} {len({x for group in values.values() for x in group})} (by {s.axes[f]['by']})")
    v = s.draw(2, 0)
    print(s.image_prompt(v))
    print(s.cosmos_prompt(v))
    print("cosmos controls:", ", ".join(s.controls))
