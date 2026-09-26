"""Folder rules of a run folder. Every script imports this file, so the rules live in one place.

Per-demo files, reference images and Cosmos videos are grouped in folders of CHUNK demos (000-049, 050-099, ...),
so no folder holds hundreds of files. The rule is the same for 2 demos and for 750 demos.

    <run>/demos/000-049/demo_002/demo_002_source.mp4 ...            make_videos.py (videos of one demo)
    <run>/ref_sim/000-049/demo_002_ref_sim.png                      make_videos.py (frame 0, for reference images)
    <run>/refs/000-049/demo_002_01.png                              your reference images
    <run>/cosmos/<checkpoint>_<date>_<time>/000-049/demo_002_01.mp4  run_cosmos.py
"""

CHUNK = 50


def chunk(idx: int) -> str:
    """Folder name of the group that holds demo idx: 0..49 -> "000-049", 50..99 -> "050-099", ..."""
    start = idx // CHUNK * CHUNK
    return f"{start:03d}-{start + CHUNK - 1:03d}"


def demo_name(idx: int) -> str:
    return f"demo_{idx:03d}"


def demo_index(name: str) -> int:
    """"demo_002" or "demo_002_01" -> 2."""
    return int(name.split("_")[1])


def demo_dir(run: str, idx: int) -> str:
    """Folder of the videos of one demo: <run>/demos/000-049/demo_002."""
    return f"{run}/demos/{chunk(idx)}/{demo_name(idx)}"


def ref_sim_dir(run: str, idx: int) -> str:
    """Folder of the frame-0 images (ref_sim) of a group of demos: <run>/ref_sim/000-049."""
    return f"{run}/ref_sim/{chunk(idx)}"


def ref_dir(refs: str, idx: int) -> str:
    """Folder of the reference images of one demo: <refs>/000-049."""
    return f"{refs}/{chunk(idx)}"
