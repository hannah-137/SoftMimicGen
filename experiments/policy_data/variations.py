"""Variation table for the reference images. All lists live here. They are the same as the reference image spec.

Every reference image gets one value on each of 7 axes: place, table, lighting, robot, towel color, towel material
and towel pattern. All 7 axes change in every image. Generation rules:
- place: first a place type, then a place inside it. The types are exact in every group of BLOCK (50) demos, the
  same groups as the demo folders (000-049, 050-099, ...): 10 outdoor (OUTDOOR_SHARE, 20%) and 4 of each of the 10
  indoor types (8% each). The 50 types of a group are shuffled over its demos. Places inside a type have the same
  chance.
- lighting: indoor lighting for an indoor place, outdoor lighting for an outdoor place.
- other axes: every item has the same chance.
- no repeat: draw() never returns a combination whose key() is in `used`.
A new attempt for the same image (make_references.py --retry) keeps the place type and draws the rest again, so the
shares stay exact. Each tag (01, 02, ...) has its own shuffle.

    v = draw(demo=2, seed=0)         # same demo, seed, attempt and tag -> same choice
    text = sentence(v)               # the sentences for the prompt
    key(v)                           # tuple of the 7 values
"""

import random

OUTDOOR_SHARE = 0.2
BLOCK = 50  # the place type shares are exact in every group of BLOCK demos

# Place type -> places. Used as "The scene is in {place}."
PLACES = {
    "laboratory": [
        "a robotics laboratory with equipment, cables and shelves in the background",
        "a university robotics lab with a whiteboard and desks behind",
        "a small research lab with monitors and a rack of electronics behind",
        "a bright lab with white walls and a row of computers behind",
        "a lab with black curtains hanging behind the table",
        "a lab with a plain gray wall and a fire extinguisher in the corner",
        "a lab with glass partitions and offices behind",
        "a lab with metal shelves full of boxes and parts behind",
        "a lab with a large window and a parking lot outside",
        "a lab with a second robot arm standing idle in the background",
        "a lab with a 3D printer and a soldering station behind",
        "a lab with a big screen showing a plot on the back wall",
        "a lab with cardboard boxes stacked against the wall",
        "a lab with a workbench and hand tools behind",
        "a lab with a blue safety fence around the work area",
        "a lab with a poster wall and a coffee machine behind",
        "a lab with a concrete floor and yellow floor markings",
        "a lab with white acoustic panels on the wall",
        "a lab with an open door to a corridor behind",
        "a lab with a camera tripod and lights standing behind",
        "a lab with a drone on a shelf and cables on the wall",
        "a lab with a fume hood and chemical bottles behind",
        "a lab with server racks with blinking lights behind",
        "a lab with a treadmill and motion capture cameras behind",
        "a lab with a ping-pong table pushed against the wall",
        "a lab with a mobile robot parked in the corner",
        "a lab with a tool cabinet and a vise behind",
        "a lab with a dark green chalkboard behind",
        "a lab with plants on the window sill",
        "a lab with a cluttered desk and a chair behind",
        "a clean room with smooth white panels and blue light",
        "a chemistry lab with glassware on the shelves behind",
        "a biology lab with microscopes and a fridge behind",
    ],
    "workshop": [
        "a workshop with tools hanging on a pegboard",
        "a wood workshop with sawdust and planks behind",
        "a metal workshop with a lathe and a drill press behind",
        "a car repair garage with a lifted car in the background",
        "a garage with a rolling door and shelves of boxes",
        "a bike repair shop with wheels hanging on the wall",
        "a makerspace with 3D printers and colorful filament spools",
        "an electronics repair shop with circuit boards on shelves",
        "a sewing workshop with fabric rolls behind",
        "a pottery studio with shelves of clay pots",
        "a subway workshop with train parts behind",
    ],
    "factory_warehouse": [
        "a factory hall with machines and safety railings far behind",
        "a factory with conveyor belts in the background",
        "a packaging plant with stacks of cardboard boxes",
        "a clean assembly line with white walls and bright light",
        "a textile factory with rolls of cloth behind",
        "a warehouse with tall pallet racks far behind",
        "a warehouse with a forklift parked in the background",
        "a logistics center with parcels on shelves",
        "a cold storage room with steel walls",
        "a loading dock with an open roller door",
        "an aircraft hangar with a plane far behind",
        "a shipyard hall with steel plates and cranes",
    ],
    "office": [
        "an office with desks, monitors and chairs behind",
        "a bright office room with a plain white wall",
        "an open-plan office with glass meeting rooms behind",
        "a meeting room with a long table and a screen behind",
        "a reception area with a sofa and plants",
        "an office corner with a filing cabinet and a printer",
        "a coworking space with brick walls and hanging lamps",
        "a call center with rows of desks behind",
        "a small startup office with sticky notes on the wall",
        "an office kitchen with a fridge and cabinets behind",
        "a conference room with flags and a podium",
        "a control room with many screens on the wall",
        "a security office with monitors behind",
        "a server room with racks and blue lights",
        "a data center corridor with cold air ducts",
        "a print shop with large printers and paper stacks",
    ],
    "school": [
        "a classroom with a chalkboard and rows of chairs",
        "a lecture hall with tiered seats behind",
        "a school science room with posters and sinks",
        "a library with tall bookshelves behind",
        "a study room with wooden desks and lamps",
        "a computer classroom with monitors on every desk",
        "a kindergarten room with small chairs and drawings on the wall",
        "a university hallway with lockers behind",
        "a student lounge with bean bags and a vending machine",
        "an art classroom with easels and paint stains",
        "a science fair hall with poster boards",
    ],
    "home": [
        "a home kitchen with tiled walls and cabinets",
        "a modern kitchen with white cabinets and a window",
        "a living room with a sofa and a bookshelf behind",
        "a living room with a TV and a plant in the corner",
        "a dining room with a wooden cabinet and framed pictures",
        "a bedroom with a bed and a wardrobe behind",
        "a home office with a desk, a lamp and books",
        "a laundry room with a washing machine and shelves",
        "a hallway of a house with coats hanging on hooks",
        "a small apartment with a balcony door behind",
        "a loft with exposed brick walls and large windows",
        "a basement with pipes on the ceiling and a concrete wall",
    ],
    "medical_care": [
        "a hospital room with a bed and medical equipment behind",
        "a clinic with white cabinets and a sink",
        "a nursing home lounge with armchairs",
        "a pharmacy with shelves of boxes behind",
        "a dental office with a chair and lamp behind",
        "a physiotherapy room with exercise balls and mats",
        "a veterinary clinic with cages and a scale",
        "a rehabilitation center with parallel bars and mats",
        "a hospital corridor with a gurney and handrails",
        "a medical storage room with boxes of supplies on shelves",
    ],
    "shop_restaurant_hotel": [
        "a small grocery store with shelves of goods",
        "a clothing store with racks of clothes behind",
        "a department store with folded clothes on tables",
        "a hardware store aisle with tools on hooks",
        "a furniture showroom with sofas and lamps",
        "an electronics store with TVs on the wall",
        "a flower shop with buckets of flowers",
        "a hotel room with a bed and curtains behind",
        "a hotel laundry with carts of folded linen",
        "a restaurant kitchen with steel counters and pots",
        "a cafe with a counter, cups and a chalk menu behind",
        "a bakery with racks of bread behind",
        "a cafeteria with tables and chairs far behind",
        "a bar with bottles on shelves behind",
    ],
    "studio_venue_sports": [
        "a photo studio with a white seamless backdrop",
        "a photo studio with a gray backdrop and softboxes",
        "a TV studio with cameras and lights behind",
        "a recording studio with foam panels on the wall",
        "a theater backstage with ropes and props",
        "a museum hall with glass cases behind",
        "an art gallery with white walls and paintings",
        "a science museum with an exhibit behind",
        "an exhibition booth with banners and a curtain behind",
        "a trade show hall with many booths far behind",
        "a robotics competition arena with a crowd far behind",
        "a gym with weights and mirrors behind",
        "a yoga studio with wooden floors and mats",
        "a locker room with metal lockers and benches",
        "a sports hall with a basketball hoop far behind",
    ],
    "other_indoor": [
        "a post office with parcels and counters",
        "a bank branch with counters and glass",
        "a community center with folding tables and chairs",
        "a shipping container converted into a workspace",
        "a greenhouse with plants and glass walls",
        "a modern hallway with glass walls and plants",
        "an elevator lobby with steel doors behind",
        "a laundromat with rows of washing machines",
        "a mailroom with sorting shelves",
        "a storage room with metal shelves and plastic bins",
        "a dry cleaner with clothes on a rail",
        "a tailor shop with a sewing machine and a mannequin",
        "a copy room with a photocopier and paper boxes",
        "a service center with devices on repair shelves",
        "a bicycle storage room with bikes on racks",
        "a janitor room with a sink and cleaning carts",
    ],
    "outdoor": [
        "a paved university courtyard",
        "a covered patio with a brick wall behind",
        "a rooftop terrace with a city skyline far behind",
        "a backyard with a lawn and a wooden fence",
        "a garden with flower beds and bushes",
        "a park with trees and a path behind",
        "a street market with stalls and awnings behind",
        "a sidewalk cafe terrace with chairs and umbrellas",
        "a farm yard with a tractor far behind",
        "a vegetable field with rows of plants",
        "an orchard with fruit trees",
        "a construction site with scaffolding far behind",
        "a container yard with stacked shipping containers",
        "a harbor quay with boats in the water",
        "a beach boardwalk with the sea behind",
        "a campsite with tents far behind",
        "a school playground with a climbing frame",
        "a sports field with goals far behind",
        "an outdoor research station with instruments on a mast",
        "a solar panel field",
        "a village square with old houses",
        "a mountain hut terrace with mountains behind",
        "a lakeside dock with trees across the water",
        "a desert test site with sand and rocks",
        "a snowy yard with snow on the ground",
        "a plant nursery with rows of potted plants",
        "a food stall under a canopy",
        "an industrial yard with pipes and tanks",
        "an apartment balcony with potted plants",
        "a driveway in front of a house with a garage door",
    ],
}

# The table under the towel. Used as "The table is {table}."
TABLES = [
    "a metal workbench", "an old wooden workbench with scratches", "a slightly scratched steel table",
    "a glossy white laminate table", "a gray plastic folding table", "a stainless steel cart",
    "a worn school desk with a scratched top", "a marble-look counter", "covered with a green cutting mat",
    "covered with a black rubber mat", "a light oak desk", "a dark walnut table", "a pine table with visible grain",
    "a bamboo table", "a birch plywood table", "a black painted wooden table", "a white painted wooden table",
    "a red painted workbench", "a blue metal table with a powder coat finish", "a green metal table with chipped paint",
    "a brushed aluminum table", "a polished chrome table", "a rusty steel table", "a concrete table",
    "a granite counter", "a white quartz counter", "a black slate table", "a glass table with a metal frame",
    "a frosted glass table", "a clear acrylic table", "a beige laminate office desk", "a gray laminate office desk",
    "a wood-look laminate table", "a table covered with a white tablecloth", "a table covered with a blue tablecloth",
    "a table covered with a red checkered cloth", "a table covered with a gray felt cloth",
    "a table covered with kraft paper", "a table covered with a thin foam sheet",
    "a table covered with a wooden cutting board", "a ceramic tiled table", "a mosaic tiled table", "a cork-top table",
    "a leather-top desk", "a carpet-covered platform", "a wooden picnic table", "a wooden kitchen island",
    "a butcher block counter", "a lab bench with a black resin top", "a lab bench with a white resin top",
    "a hospital bed table", "a folding camping table", "a wooden pallet on a stand",
    "an ironing board style padded table", "a table with a perforated steel top", "a table with a wire mesh top",
    "a table with an anti-static gray top", "a table with a yellow safety top",
    "a table with a wood top and black steel legs", "a table with a white top and wooden legs",
]

# Indoor lighting = source + color + level. Used as "The lighting is {source}, {color}, {level}."
INDOOR_LIGHT_SOURCES = [
    "daylight from a window on the left",
    "daylight from a window on the right",
    "fluorescent ceiling panels",
    "warm ceiling lamps",
    "a single desk lamp close to the table",
    "studio softbox lights from the front",
]
INDOOR_LIGHT_COLORS = [
    "with a neutral white tone",
    "with a warm yellow tone",
    "with a cool blue tone",
    "with a slight green tone",
]
INDOOR_LIGHT_LEVELS = [
    "bright",
    "medium bright",
    "dim, with soft shadows",
]
# Outdoor lighting. Used as "The lighting is {light}."
OUTDOOR_LIGHTS = [
    "direct midday sun with hard shadows",
    "morning sun from the left with long shadows",
    "morning sun from the right with long shadows",
    "warm evening sun from the left",
    "warm evening sun from the right",
    "golden hour light, low and warm",
    "a bright overcast sky with soft even light",
    "a dark overcast sky before rain, dim",
    "open shade under a roof, soft light",
    "tree shade with spots of sunlight",
    "hazy sun with a slight blue tone",
    "blue hour after sunset, dim and cool",
]

# The robot is the same white Franka arm in the same pose. Only the surface wear changes.
ROBOTS = [
    "The robot is brand new and spotless.",
    "The robot has light scuffs on the joints.",
    "The robot has some dust on the upper links.",
    "The robot has worn paint on the gripper fingers.",
    "The robot has a few fingerprints and smudges.",
]

# Used as "The towel is {color} {material}, {pattern}."
TOWEL_COLORS = [
    "white", "off-white", "ivory", "cream", "beige", "sand", "tan", "khaki", "light gray", "gray", "dark gray",
    "charcoal", "black", "silver gray", "warm gray", "cool gray", "taupe", "mushroom", "light brown", "brown",
    "dark brown", "chocolate", "coffee", "caramel", "rust", "terracotta", "brick red", "red", "dark red", "wine red",
    "burgundy", "cherry red", "coral", "salmon", "peach", "apricot", "orange", "burnt orange", "amber", "mustard",
    "yellow", "pale yellow", "lemon", "gold", "olive", "olive green", "moss green", "forest green", "dark green",
    "green", "bright green", "lime", "mint", "sage", "sea green", "teal", "dark teal", "turquoise", "aqua", "cyan",
    "sky blue", "light blue", "baby blue", "powder blue", "blue", "royal blue", "cobalt blue", "navy", "dark navy",
    "denim blue", "steel blue", "slate blue", "periwinkle", "lavender", "lilac", "violet", "purple", "dark purple",
    "plum", "grape", "magenta", "fuchsia", "hot pink", "pink", "light pink", "blush pink", "dusty rose", "rose",
    "mauve", "berry", "pale green", "pale blue", "pale pink", "pale lavender", "stone", "oatmeal", "linen white",
    "smoke gray", "ash gray", "pewter",
]
TOWEL_MATERIALS = [
    "terry cloth", "microfiber", "linen", "waffle-weave cotton", "fleece", "plain cotton", "bamboo fiber",
    "knit cotton",
]
TOWEL_PATTERNS = [
    "plain, one solid color",
    "with a darker border along the edges",
    "with thin stripes",
    "with a small check pattern",
    "with a two-tone design, one half a lighter shade",
    "with small dots",
]

FIELDS = ["place", "table", "lighting", "robot", "towel_color", "towel_material", "towel_pattern"]


def block_types() -> list:
    """The place types of one group of BLOCK demos: OUTDOOR_SHARE outdoor, the rest split equally over the indoor
    types."""
    indoor = [t for t in PLACES if t != "outdoor"]
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


def key(v: dict) -> tuple:
    """The 7 values that identify one combination."""
    return tuple(v[f] for f in FIELDS)


def draw(demo: int, seed: int, used: set | None = None, attempt: int = 0, tag: str = "") -> dict:
    """One variation for a demo: the 7 axes plus "place_type". Same demo, seed, attempt and tag give the same choice.
    The place type depends on demo, seed and tag only. A combination whose key() is in `used` is skipped, so two
    images never share a combination."""
    ptype = place_type(demo, seed, tag)
    rng = random.Random(f"{seed}:{tag}:{demo}:{attempt}")
    for _ in range(1000):
        if ptype == "outdoor":
            lighting = rng.choice(OUTDOOR_LIGHTS)
        else:
            source, color = rng.choice(INDOOR_LIGHT_SOURCES), rng.choice(INDOOR_LIGHT_COLORS)
            lighting = f"{source}, {color}, {rng.choice(INDOOR_LIGHT_LEVELS)}"
        v = {
            "place_type": ptype,
            "place": rng.choice(PLACES[ptype]),
            "table": rng.choice(TABLES),
            "lighting": lighting,
            "robot": rng.choice(ROBOTS),
            "towel_color": rng.choice(TOWEL_COLORS),
            "towel_material": rng.choice(TOWEL_MATERIALS),
            "towel_pattern": rng.choice(TOWEL_PATTERNS),
        }
        if used is None or key(v) not in used:
            return v
    raise RuntimeError("no unused combination found")


def sentence(v: dict) -> str:
    """The variation as text for the prompt."""
    return (f"The scene is in {v['place']}. The table is {v['table']}. The lighting is {v['lighting']}. {v['robot']} "
            f"The towel is {v['towel_color']} {v['towel_material']}, {v['towel_pattern']}.")


if __name__ == "__main__":
    n_places = {t: len(p) for t, p in PLACES.items()}
    print(f"places {sum(n_places.values())} {n_places}")
    n_indoor_light = len(INDOOR_LIGHT_SOURCES) * len(INDOOR_LIGHT_COLORS) * len(INDOOR_LIGHT_LEVELS)
    print(f"tables {len(TABLES)}, indoor lighting {n_indoor_light}, "
          f"outdoor lighting {len(OUTDOOR_LIGHTS)}, robots {len(ROBOTS)}, towel colors {len(TOWEL_COLORS)}, "
          f"materials {len(TOWEL_MATERIALS)}, patterns {len(TOWEL_PATTERNS)}")
    print(sentence(draw(2, 0)))
