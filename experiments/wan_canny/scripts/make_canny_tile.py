"""Two-view tiled Canny control video for Wan (CRAFT-style multi-view).
Tiles agentview + wrist side by side (1024x512), THEN extracts Canny —
tiling before edge extraction, per CRAFT Sec. IV-C."""
import h5py, cv2, numpy as np, os, sys
import imageio

HDF5 = sys.argv[1] if len(sys.argv) > 1 else "datasets/generated_dataset/towel_512x2.hdf5"
PREFIX = os.path.splitext(os.path.basename(HDF5))[0] + "_tile"
OUT_DIR = "experiments/wan_canny/inputs"
N_FRAMES = 81
FPS = 16
CANNY_LO, CANNY_HI = 20, 80
DILATE_K = 2
CLOSE_K = 3

os.makedirs(OUT_DIR, exist_ok=True)
with h5py.File(HDF5) as f:
    obs = f["data/demo_0/obs"]
    agent = obs["agentview_image"][:]          # (T, 512, 512, 3)
    wrist = obs["robot0_eye_in_hand_image"][:] # (T, 512, 512, 3)

idx = np.linspace(0, len(agent) - 1, N_FRAMES).astype(int)
agent, wrist = agent[idx], wrist[idx]

# tile first: side by side -> (T, 512, 1024, 3)
tiled = np.concatenate([agent, wrist], axis=2)

dilate_kernel = np.ones((DILATE_K, DILATE_K), np.uint8)
close_kernel = np.ones((CLOSE_K, CLOSE_K), np.uint8)

canny_frames = []
for im in tiled:
    blurred = cv2.GaussianBlur(im, (5, 5), 0)
    edges = np.zeros(im.shape[:2], np.uint8)
    for c in range(3):
        edges = cv2.bitwise_or(edges, cv2.Canny(blurred[:, :, c], CANNY_LO, CANNY_HI))
    edges = cv2.dilate(edges, dilate_kernel)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)
    canny_frames.append(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

imageio.mimsave(f"{OUT_DIR}/{PREFIX}_source.mp4", list(tiled), fps=FPS)
imageio.mimsave(f"{OUT_DIR}/{PREFIX}_canny.mp4", canny_frames, fps=FPS)
cv2.imwrite(f"{OUT_DIR}/{PREFIX}_ref_sim.png", cv2.cvtColor(tiled[0], cv2.COLOR_RGB2BGR))

print(f"done: {N_FRAMES} frames, {tiled.shape[2]}x{tiled.shape[1]} (tiled), {FPS}fps -> {OUT_DIR}/")
print(f"saved: {PREFIX}_source.mp4, {PREFIX}_canny.mp4, {PREFIX}_ref_sim.png")