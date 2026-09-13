"""Build every Wan input from one generated hdf5: source video + reference image, RGB Canny video, shaded Canny video.

Usage: python experiments/wan_canny/make_videos.py <hdf5> [--out_dir DIR] [--demo demo_0]
Each step is also runnable on its own (make_source.py, make_rgb_canny.py, make_shaded_canny.py)."""
import make_rgb_canny
import make_shaded_canny
import make_source
from common import parse_args


def main():
    args = parse_args(__doc__)
    make_source.run(args.hdf5, args.out_dir, args.demo)
    make_rgb_canny.run(args.hdf5, args.out_dir, args.demo)
    make_shaded_canny.run(args.hdf5, args.out_dir, args.demo)
    print(f"done -> {args.out_dir}/")


if __name__ == "__main__":
    main()
