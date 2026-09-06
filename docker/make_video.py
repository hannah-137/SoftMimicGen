# hdf5의 카메라 이미지를 mp4로 변환
# 사용법: python /workspace/make_video.py <hdf5경로> [출력폴더]

import sys, os
import h5py
import numpy as np
import imageio.v3 as iio

hdf5_path = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else "/workspace/videos"
os.makedirs(out_dir, exist_ok=True)

FPS = 20          # 환경 스텝 0.05초 → 20fps = 실시간
SCALE = 4         # 128px는 너무 작아서 4배 확대 (512px)

with h5py.File(hdf5_path, "r") as f:
    for demo in f["data"].keys():
        obs = f["data"][demo]["obs"]
        agent = obs["agentview_image"][:]            # (T, 128, 128, 3)
        hand = obs["robot0_eye_in_hand_image"][:]    # (T, 128, 128, 3)

        # 두 카메라를 좌우로 붙임 → (T, 128, 256, 3)
        frames = np.concatenate([agent, hand], axis=2)

        # 확대 (nearest neighbor, 픽셀 뭉개지지 않게)
        frames = frames.repeat(SCALE, axis=1).repeat(SCALE, axis=2)

        out_path = os.path.join(out_dir, f"{demo}.mp4")
        iio.imwrite(out_path, frames, fps=FPS, codec="libx264", plugin="pyav" if False else None)
        print(f"{demo}: {frames.shape[0]} frames → {out_path}")

print("done")