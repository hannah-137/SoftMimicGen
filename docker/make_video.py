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

        # 이미지 관측 전부 찾기: (T, H, W, 3) uint8
        # 1채널 엣지 채널(<cam>_shadedcanny 등)은 제외, RGB 카메라만
        cams = [k for k in obs.keys() if obs[k].ndim == 4 and obs[k].dtype == np.uint8 and obs[k].shape[-1] == 3]
        if not cams:
            print(f"{demo}: no image obs, skip"); continue
        imgs = [obs[k][:] for k in cams]

        # 높이를 첫 카메라에 맞춤 (태스크마다 해상도 다름)
        import cv2
        h = imgs[0].shape[1]
        for i in range(1, len(imgs)):
            if imgs[i].shape[1] != h:
                w = int(imgs[i].shape[2] * h / imgs[i].shape[1])
                imgs[i] = np.stack([cv2.resize(fr, (w, h), interpolation=cv2.INTER_NEAREST) for fr in imgs[i]])

        # 카메라들을 좌우로 붙임 (하나면 그대로)
        frames = np.concatenate(imgs, axis=2)
        print(f"{demo}: cameras {cams}")

        # 확대 (nearest neighbor, 픽셀 뭉개지지 않게)
        scale = max(1, SCALE * 128 // h)  # 작은 카메라만 확대 (512px 카메라는 그대로)
        frames = frames.repeat(scale, axis=1).repeat(scale, axis=2)

        out_path = os.path.join(out_dir, f"{demo}.mp4")
        iio.imwrite(out_path, frames, fps=FPS, codec="libx264", plugin="pyav" if False else None)
        print(f"{demo}: {frames.shape[0]} frames → {out_path}")

print("done")