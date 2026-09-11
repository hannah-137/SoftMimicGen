"""towel_512.hdf5 → Canny control video for Wan (81 frames, 16fps).
CRAFT-style post-processing: edge thickening (dilation) + gap bridging (closing)."""
import h5py, cv2, numpy as np, os
import imageio

import sys, os
HDF5 = sys.argv[1] if len(sys.argv) > 1 else "datasets/generated_dataset/towel_512.hdf5"
PREFIX = os.path.splitext(os.path.basename(HDF5))[0]
OUT_DIR = "experiments/wan_canny/inputs"

N_FRAMES = 81
FPS = 16
CANNY_LO, CANNY_HI = 80, 180
DILATE_K = 2   # edge thickening kernel
CLOSE_K = 3    # gap bridging kernel

os.makedirs(OUT_DIR, exist_ok=True)
with h5py.File(HDF5) as f:
    imgs = f["data/demo_0/obs/agentview_image"][:]   # (133, 512, 512, 3)

# 133 -> 81 uniform sampling
idx = np.linspace(0, len(imgs) - 1, N_FRAMES).astype(int)
imgs = imgs[idx]

h, w = imgs.shape[1:3]
dilate_kernel = np.ones((DILATE_K, DILATE_K), np.uint8)
close_kernel = np.ones((CLOSE_K, CLOSE_K), np.uint8)

canny_frames = []
for im in imgs:
    # per-channel Canny, then combine — catches color edges (blue towel on dark table)
    edges = np.zeros(im.shape[:2], np.uint8)
    for c in range(3):
        edges = cv2.bitwise_or(edges, cv2.Canny(im[:, :, c], CANNY_LO, CANNY_HI))
    edges = cv2.dilate(edges, dilate_kernel)                        # thicken
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)  # bridge gaps
    canny_frames.append(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

imageio.mimsave(f"{OUT_DIR}/{PREFIX}_source.mp4", list(imgs), fps=FPS)
imageio.mimsave(f"{OUT_DIR}/{PREFIX}_canny.mp4", canny_frames, fps=FPS)

# save first source frame as candidate reference image
cv2.imwrite(f"{OUT_DIR}/{PREFIX}_ref_sim.png", cv2.cvtColor(imgs[0], cv2.COLOR_RGB2BGR))

print(f"done: {N_FRAMES} frames, {w}x{h}, {FPS}fps -> {OUT_DIR}/")
print("saved: towel_source.mp4, towel_canny.mp4, towel_ref_sim.png")