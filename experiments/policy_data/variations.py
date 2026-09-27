"""Variation table for the reference images. All lists live here.

Every reference image gets one sentence per axis: environment, table, lighting, robot and towel. The lists follow
the axes of CRAFT (lighting, background, object color) plus the table and the robot state. The lists are large so
that 750 images can all differ, and `draw()` never returns a combination that is already in `used`.

    v = draw(demo=2, seed=0)         # same demo + seed -> same choice
    text = sentence(v)               # the sentences for the prompt
    key(v)                           # tuple that identifies the combination

Mode "wide" uses every environment. Mode "lab" uses only the laboratory-like environments.
"""

import random

# Where the robot stands. Indoor places where a robot arm can be. The first entries are laboratories.
ENVIRONMENTS = [
    "The scene is in a robotics laboratory with equipment, cables and shelves in the background.",
    "The scene is in a university robotics lab with a whiteboard and desks behind.",
    "The scene is in a small research lab with monitors and a rack of electronics behind.",
    "The scene is in a bright lab with white walls and a row of computers behind.",
    "The scene is in a lab with black curtains hanging behind the table.",
    "The scene is in a lab with a plain gray wall and a fire extinguisher in the corner.",
    "The scene is in a lab with glass partitions and offices behind.",
    "The scene is in a lab with metal shelves full of boxes and parts behind.",
    "The scene is in a lab with a large window and a parking lot outside.",
    "The scene is in a lab with a second robot arm standing idle in the background.",
    "The scene is in a lab with a 3D printer and a soldering station behind.",
    "The scene is in a lab with a big screen showing a plot on the back wall.",
    "The scene is in a lab with cardboard boxes stacked against the wall.",
    "The scene is in a lab with a workbench and hand tools behind.",
    "The scene is in a lab with a blue safety fence around the work area.",
    "The scene is in a lab with a poster wall and a coffee machine behind.",
    "The scene is in a lab with a concrete floor and yellow floor markings.",
    "The scene is in a lab with white acoustic panels on the wall.",
    "The scene is in a lab with an open door to a corridor behind.",
    "The scene is in a lab with a camera tripod and lights standing behind.",
    "The scene is in a lab with a drone on a shelf and cables on the wall.",
    "The scene is in a lab with a fume hood and chemical bottles behind.",
    "The scene is in a lab with server racks with blinking lights behind.",
    "The scene is in a lab with a treadmill and motion capture cameras behind.",
    "The scene is in a lab with a ping-pong table pushed against the wall.",
    "The scene is in a lab with a mobile robot parked in the corner.",
    "The scene is in a lab with a tool cabinet and a vise behind.",
    "The scene is in a lab with a dark green chalkboard behind.",
    "The scene is in a lab with plants on the window sill.",
    "The scene is in a lab with a cluttered desk and a chair behind.",
    "The scene is in a workshop with tools hanging on a pegboard.",
    "The scene is in a wood workshop with sawdust and planks behind.",
    "The scene is in a metal workshop with a lathe and a drill press behind.",
    "The scene is in a car repair garage with a lifted car in the background.",
    "The scene is in a garage with a rolling door and shelves of boxes.",
    "The scene is in a bike repair shop with wheels hanging on the wall.",
    "The scene is in a makerspace with 3D printers and colorful filament spools.",
    "The scene is in an electronics repair shop with circuit boards on shelves.",
    "The scene is in a sewing workshop with fabric rolls behind.",
    "The scene is in a pottery studio with shelves of clay pots.",
    "The scene is in a factory hall with machines and safety railings far behind.",
    "The scene is in a factory with conveyor belts in the background.",
    "The scene is in a packaging plant with stacks of cardboard boxes.",
    "The scene is in a clean assembly line with white walls and bright light.",
    "The scene is in a textile factory with rolls of cloth behind.",
    "The scene is in a warehouse with tall pallet racks far behind.",
    "The scene is in a warehouse with a forklift parked in the background.",
    "The scene is in a logistics center with parcels on shelves.",
    "The scene is in a cold storage room with steel walls.",
    "The scene is in a loading dock with an open roller door.",
    "The scene is in an office with desks, monitors and chairs behind.",
    "The scene is in a bright office room with a plain white wall.",
    "The scene is in an open-plan office with glass meeting rooms behind.",
    "The scene is in a meeting room with a long table and a screen behind.",
    "The scene is in a reception area with a sofa and plants.",
    "The scene is in an office corner with a filing cabinet and a printer.",
    "The scene is in a coworking space with brick walls and hanging lamps.",
    "The scene is in a call center with rows of desks behind.",
    "The scene is in a small startup office with sticky notes on the wall.",
    "The scene is in an office kitchen with a fridge and cabinets behind.",
    "The scene is in a classroom with a chalkboard and rows of chairs.",
    "The scene is in a lecture hall with tiered seats behind.",
    "The scene is in a school science room with posters and sinks.",
    "The scene is in a library with tall bookshelves behind.",
    "The scene is in a study room with wooden desks and lamps.",
    "The scene is in a computer classroom with monitors on every desk.",
    "The scene is in a kindergarten room with small chairs and drawings on the wall.",
    "The scene is in a university hallway with lockers behind.",
    "The scene is in a student lounge with bean bags and a vending machine.",
    "The scene is in an art classroom with easels and paint stains.",
    "The scene is in a home kitchen with tiled walls and cabinets.",
    "The scene is in a modern kitchen with white cabinets and a window.",
    "The scene is in a living room with a sofa and a bookshelf behind.",
    "The scene is in a living room with a TV and a plant in the corner.",
    "The scene is in a dining room with a wooden cabinet and framed pictures.",
    "The scene is in a bedroom with a bed and a wardrobe behind.",
    "The scene is in a home office with a desk, a lamp and books.",
    "The scene is in a laundry room with a washing machine and shelves.",
    "The scene is in a hallway of a house with coats hanging on hooks.",
    "The scene is in a small apartment with a balcony door behind.",
    "The scene is in a hospital room with a bed and medical equipment behind.",
    "The scene is in a clinic with white cabinets and a sink.",
    "The scene is in a nursing home lounge with armchairs.",
    "The scene is in a pharmacy with shelves of boxes behind.",
    "The scene is in a dental office with a chair and lamp behind.",
    "The scene is in a physiotherapy room with exercise balls and mats.",
    "The scene is in a hotel room with a bed and curtains behind.",
    "The scene is in a hotel laundry with carts of folded linen.",
    "The scene is in a restaurant kitchen with steel counters and pots.",
    "The scene is in a cafe with a counter, cups and a chalk menu behind.",
    "The scene is in a bakery with racks of bread behind.",
    "The scene is in a cafeteria with tables and chairs far behind.",
    "The scene is in a bar with bottles on shelves behind.",
    "The scene is in a small grocery store with shelves of goods.",
    "The scene is in a clothing store with racks of clothes behind.",
    "The scene is in a department store with folded clothes on tables.",
    "The scene is in a hardware store aisle with tools on hooks.",
    "The scene is in a furniture showroom with sofas and lamps.",
    "The scene is in an electronics store with TVs on the wall.",
    "The scene is in a flower shop with buckets of flowers.",
    "The scene is in a gym with weights and mirrors behind.",
    "The scene is in a yoga studio with wooden floors and mats.",
    "The scene is in a locker room with metal lockers and benches.",
    "The scene is in a swimming pool building with tiled walls.",
    "The scene is in a sports hall with a basketball hoop far behind.",
    "The scene is in a photo studio with a white seamless backdrop.",
    "The scene is in a photo studio with a gray backdrop and softboxes.",
    "The scene is in a TV studio with cameras and lights behind.",
    "The scene is in a recording studio with foam panels on the wall.",
    "The scene is in a theater backstage with ropes and props.",
    "The scene is in a museum hall with glass cases behind.",
    "The scene is in an art gallery with white walls and paintings.",
    "The scene is in a science museum with an exhibit behind.",
    "The scene is in an exhibition booth with banners and a curtain behind.",
    "The scene is in a trade show hall with many booths far behind.",
    "The scene is in a conference room with flags and a podium.",
    "The scene is in an airport lounge with large windows.",
    "The scene is in a train station hall with a departure board.",
    "The scene is in a bus depot with buses parked behind.",
    "The scene is in a ship cabin with round windows.",
    "The scene is in a server room with racks and blue lights.",
    "The scene is in a data center corridor with cold air ducts.",
    "The scene is in a control room with many screens on the wall.",
    "The scene is in a security office with monitors behind.",
    "The scene is in a print shop with large printers and paper stacks.",
    "The scene is in a post office with parcels and counters.",
    "The scene is in a bank branch with counters and glass.",
    "The scene is in a courtroom with wooden benches behind.",
    "The scene is in a church hall with wooden chairs and windows.",
    "The scene is in a community center with folding tables and chairs.",
    "The scene is in a basement with pipes on the ceiling and a concrete wall.",
    "The scene is in an attic with wooden beams and boxes.",
    "The scene is in a shipping container converted into a workspace.",
    "The scene is in a tent at an outdoor event with banners behind.",
    "The scene is in a greenhouse with plants and glass walls.",
    "The scene is in a barn with hay and wooden walls.",
    "The scene is in a veterinary clinic with cages and a scale.",
    "The scene is in a science fair hall with poster boards.",
    "The scene is in a robotics competition arena with a crowd far behind.",
    "The scene is in a loft with exposed brick walls and large windows.",
    "The scene is in a modern hallway with glass walls and plants.",
    "The scene is in a stairwell landing with a gray wall and a railing.",
    "The scene is in an elevator lobby with steel doors behind.",
    "The scene is in a parking garage with concrete pillars.",
    "The scene is in a subway workshop with train parts behind.",
    "The scene is in an aircraft hangar with a plane far behind.",
    "The scene is in a shipyard hall with steel plates and cranes.",
    "The scene is in a clean room with smooth white panels and blue light.",
    "The scene is in a chemistry lab with glassware on the shelves behind.",
    "The scene is in a biology lab with microscopes and a fridge behind.",
]

LAB_ENVIRONMENTS = [e for e in ENVIRONMENTS if "lab" in e.lower()]

# The table under the towel.
TABLES = [
    "The table is a metal workbench.",
    "The table is an old wooden workbench with scratches.",
    "The table is a slightly scratched steel table.",
    "The table is a glossy white laminate table.",
    "The table is a gray plastic folding table.",
    "The table is a stainless steel cart.",
    "The table is a worn school desk with a scratched top.",
    "The table is a marble-look counter.",
    "The table is covered with a green cutting mat.",
    "The table is covered with a black rubber mat.",
    "The table is a light oak desk.",
    "The table is a dark walnut table.",
    "The table is a pine table with visible grain.",
    "The table is a bamboo table.",
    "The table is a birch plywood table.",
    "The table is a black painted wooden table.",
    "The table is a white painted wooden table.",
    "The table is a red painted workbench.",
    "The table is a blue metal table with a powder coat finish.",
    "The table is a green metal table with chipped paint.",
    "The table is a brushed aluminum table.",
    "The table is a polished chrome table.",
    "The table is a rusty steel table.",
    "The table is a concrete table.",
    "The table is a granite counter.",
    "The table is a white quartz counter.",
    "The table is a black slate table.",
    "The table is a glass table with a metal frame.",
    "The table is a frosted glass table.",
    "The table is a clear acrylic table.",
    "The table is a beige laminate office desk.",
    "The table is a gray laminate office desk.",
    "The table is a wood-look laminate table.",
    "The table is a table covered with a white tablecloth.",
    "The table is a table covered with a blue tablecloth.",
    "The table is a table covered with a red checkered cloth.",
    "The table is a table covered with a gray felt cloth.",
    "The table is a table covered with kraft paper.",
    "The table is a table covered with a thin foam sheet.",
    "The table is a table covered with a wooden cutting board.",
    "The table is a ceramic tiled table.",
    "The table is a mosaic tiled table.",
    "The table is a cork-top table.",
    "The table is a leather-top desk.",
    "The table is a carpet-covered platform.",
    "The table is a wooden picnic table.",
    "The table is a wooden kitchen island.",
    "The table is a butcher block counter.",
    "The table is a lab bench with a black resin top.",
    "The table is a lab bench with a white resin top.",
    "The table is a hospital bed table.",
    "The table is a folding camping table.",
    "The table is a wooden pallet on a stand.",
    "The table is an ironing board style padded table.",
    "The table is a table with a perforated steel top.",
    "The table is a table with a wire mesh top.",
    "The table is a table with an anti-static gray top.",
    "The table is a table with a yellow safety top.",
    "The table is a table with a wood top and black steel legs.",
    "The table is a table with a white top and wooden legs.",
]

# Lighting = source + color + level, joined into one sentence.
LIGHT_SOURCES = [
    "daylight from a window on the left",
    "daylight from a window on the right",
    "fluorescent ceiling panels",
    "warm ceiling lamps",
    "a single desk lamp close to the table",
    "studio softbox lights from the front",
]
LIGHT_COLORS = [
    "with a neutral white tone",
    "with a warm yellow tone",
    "with a cool blue tone",
    "with a slight green tone",
]
LIGHT_LEVELS = [
    "bright",
    "medium bright",
    "dim, with soft shadows",
]

# The robot is the same white Franka arm. Only small wear changes.
ROBOTS = [
    "The robot is brand new and spotless.",
    "The robot has light scuffs on the joints.",
    "The robot has some dust on the upper links.",
    "The robot has worn paint on the gripper fingers.",
    "The robot has a few fingerprints and smudges.",
]

TOWEL_COLORS = [
    "white", "off-white", "ivory", "cream", "beige", "sand", "tan", "khaki", "light gray", "gray",
    "dark gray", "charcoal", "black", "silver gray", "warm gray", "cool gray", "taupe", "mushroom", "light brown", "brown",
    "dark brown", "chocolate", "coffee", "caramel", "rust", "terracotta", "brick red", "red", "dark red", "wine red",
    "burgundy", "cherry red", "coral", "salmon", "peach", "apricot", "orange", "burnt orange", "amber", "mustard",
    "yellow", "pale yellow", "lemon", "gold", "olive", "olive green", "moss green", "forest green", "dark green", "green",
    "bright green", "lime", "mint", "sage", "sea green", "teal", "dark teal", "turquoise", "aqua", "cyan",
    "sky blue", "light blue", "baby blue", "powder blue", "blue", "royal blue", "cobalt blue", "navy", "dark navy", "denim blue",
    "steel blue", "slate blue", "periwinkle", "lavender", "lilac", "violet", "purple", "dark purple", "plum", "grape",
    "magenta", "fuchsia", "hot pink", "pink", "light pink", "blush pink", "dusty rose", "rose", "mauve", "berry",
    "pale green", "pale blue", "pale pink", "pale lavender", "stone", "oatmeal", "linen white", "smoke gray", "ash gray", "pewter",
]

TOWEL_MATERIALS = [
    "terry cloth", "microfiber", "linen", "waffle-weave cotton", "fleece", "plain cotton", "bamboo fiber", "knit cotton",
]

TOWEL_PATTERNS = [
    "plain, one solid color",
    "with a darker border along the edges",
    "with thin stripes",
    "with a small check pattern",
    "with a two-tone design, one half a lighter shade",
    "with a small woven logo in one corner",
]

FIELDS = ["environment", "table", "lighting", "robot", "towel_color", "towel_material", "towel_pattern"]


def key(v: dict) -> tuple:
    """The tuple that identifies one combination."""
    return tuple(v[f] for f in FIELDS)


def draw(demo: int, seed: int, mode: str = "wide", used: set | None = None) -> dict:
    """One variation for a demo. Same demo, seed and mode give the same choice. A combination whose key() is in
    `used` is skipped, so two images never share a combination."""
    envs = LAB_ENVIRONMENTS if mode == "lab" else ENVIRONMENTS
    rng = random.Random(f"{seed}:{demo}")
    for _ in range(1000):
        v = {
            "environment": rng.choice(envs),
            "table": rng.choice(TABLES),
            "lighting": f"The lighting is {rng.choice(LIGHT_SOURCES)}, {rng.choice(LIGHT_COLORS)}, {rng.choice(LIGHT_LEVELS)}.",
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
    towel = f"The towel is {v['towel_color']} {v['towel_material']}, {v['towel_pattern']}."
    return " ".join([v["environment"], v["table"], v["lighting"], v["robot"], towel])


if __name__ == "__main__":
    print(f"environments {len(ENVIRONMENTS)} (lab {len(LAB_ENVIRONMENTS)}), tables {len(TABLES)}, "
          f"lighting {len(LIGHT_SOURCES) * len(LIGHT_COLORS) * len(LIGHT_LEVELS)}, robots {len(ROBOTS)}, "
          f"towels {len(TOWEL_COLORS) * len(TOWEL_MATERIALS) * len(TOWEL_PATTERNS)}")
    print(sentence(draw(2, 0)))
