"""Start cosmos_framework inference with in-process fixes. torchrun runs this file, run_cosmos.py builds the command.

1. The video goes into the "16,9" size slot of the 480 tier. That slot is set to the video size so the output keeps
   the input size: 1024x512 (the Franka tile), or COSMOS_SIZE=<W>x<H> from run_cosmos.py (the Cosmos size of the
   spec, yam_towel/v1: 640x240). Nothing in the framework is edited.
2. The cuSOLVER handle is created now, while the GPU is empty. UniPC calls torch.linalg.solve during sampling. With
   a nearly full GPU the handle creation fails (CUSOLVER_STATUS_INTERNAL_ERROR).
3. Two options lower the GPU memory. run_cosmos.py sets them (--no_text_head, --free_gather_early). The framework
   splits the weights over the GPUs with fully_shard (FSDP). Both options change that step.
   COSMOS_NO_TEXT_HEAD=1 removes the text output layer (lm_head, 1484 MiB in Cosmos3-Super). Video generation does
   not call it. At the first forward each of 2 GPUs holds 2.5 times its size: the shard, the gathered copy and the
   gather buffer.
   COSMOS_FREE_GATHER_EARLY=1 frees every gather buffer right after use. By default FSDP keeps the buffer of one
   unit until it gathers the next unit.
4. Every rank prints its peak GPU memory at the end. PyTorch measures it.
"""

import os
import runpy
import sys

import torch
import torch.distributed.fsdp as fsdp

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
_a = torch.eye(4, device="cuda")
torch.linalg.solve(_a, _a)
torch.cuda.synchronize()
del _a

_NO_TEXT_HEAD = os.environ.get("COSMOS_NO_TEXT_HEAD") == "1"
_FREE_GATHER_EARLY = os.environ.get("COSMOS_FREE_GATHER_EARLY") == "1"
_fully_shard = fsdp.fully_shard


def _fully_shard_with_savings(module, *args, **kwargs):
    """fully_shard for the framework. Only the whole network has language_model. Its call is the last one."""
    lm = getattr(module, "language_model", None)
    if _NO_TEXT_HEAD and lm is not None and getattr(lm, "lm_head", None) is not None:
        lm.lm_head = None
        print("[cosmos_launch] removed the text output layer (lm_head)", flush=True)
    out = _fully_shard(module, *args, **kwargs)
    if _FREE_GATHER_EARLY and lm is not None:
        out._set_unshard_async_op(True)
        print("[cosmos_launch] gather buffers are freed right after use", flush=True)
    return out


if _NO_TEXT_HEAD or _FREE_GATHER_EARLY:
    fsdp.fully_shard = _fully_shard_with_savings  # before the framework imports fully_shard

from cosmos_framework.data.generator import utils  # noqa: E402

_w, _h = (int(x) for x in os.environ.get("COSMOS_SIZE", "1024x512").split("x"))
utils.VIDEO_RES_SIZE_INFO["480"]["16,9"] = (_w, _h)
print("[cosmos_launch] size slot 480/16,9 =", utils.VIDEO_RES_SIZE_INFO["480"]["16,9"], flush=True)

sys.argv = ["cosmos_framework.scripts.inference"] + sys.argv[1:]
try:
    runpy.run_module("cosmos_framework.scripts.inference", run_name="__main__")
finally:
    print(f"[cosmos_launch] rank {os.environ.get('LOCAL_RANK', 0)} peak GPU memory: "
          f"{torch.cuda.max_memory_allocated() // 2**20} MiB allocated, "
          f"{torch.cuda.max_memory_reserved() // 2**20} MiB reserved", flush=True)
