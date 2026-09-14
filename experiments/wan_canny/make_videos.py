"""Stages 2 + 3 (softmimicgen env): source video + reference image, RGB Canny, shaded Canny, geometry edges,
shaded Canny + depth, union (shaded Canny OR geometry edges) and, when the hdf5 was made with --inspect, the
shaded segmentation and geometry-input (depth, normals, instance) videos.

  python experiments/wan_canny/make_videos.py <run_dir> [--hdf5 H5] [--prefix P] [--task T] [--demo demo_0]

The learned edges (make_learned_edges.py, rgb_edge env) and the Wan videos (make_wan.py) are separate steps;
docker/gen.sh chains everything. Each step is also runnable on its own."""
import make_geo_edge
import make_geo_inputs
import make_rgb_canny
import make_shaded
import make_shaded_canny
import make_shaded_canny_depth
import make_source
import make_union
from common import parse_args


def main():
    rd = parse_args(__doc__)
    make_source.run(rd)
    make_rgb_canny.run(rd)
    make_shaded_canny.run(rd)
    make_geo_edge.run(rd)
    make_shaded_canny_depth.run(rd)
    make_union.run(rd)
    make_shaded.run(rd)
    make_geo_inputs.run(rd)
    print(f"done -> {rd.run_dir}/")


if __name__ == "__main__":
    main()
