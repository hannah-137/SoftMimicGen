"""Start cosmos_framework inference with two in-process fixes. torchrun runs this file, run_cosmos.py builds the command.

1. The 1024x512 tiled video is detected as aspect "16,9". That size slot is set to (1024, 512) so the output keeps
   the input size. Nothing in the framework is edited.
2. The cuSOLVER handle is created now, while the GPU is empty. UniPC calls torch.linalg.solve during sampling. With
   a nearly full GPU the handle creation fails (CUSOLVER_STATUS_INTERNAL_ERROR).
"""

import os
import runpy
import sys

import torch

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
_a = torch.eye(4, device="cuda")
torch.linalg.solve(_a, _a)
torch.cuda.synchronize()
del _a

from cosmos_framework.data.generator import utils  # noqa: E402

utils.VIDEO_RES_SIZE_INFO["480"]["16,9"] = (1024, 512)
print("[cosmos_launch] size slot 480/16,9 =", utils.VIDEO_RES_SIZE_INFO["480"]["16,9"], flush=True)

sys.argv = ["cosmos_framework.scripts.inference"] + sys.argv[1:]
runpy.run_module("cosmos_framework.scripts.inference", run_name="__main__")
