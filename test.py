import torch
import numpy as np
from PIL import Image
from oemer_ng.inference.pipeline import OMRPipeline
from oemer_ng.models.omr_model import OMRModel

# 检查 checkpoints 目录下哪个模型权重没有 NaN
import os, glob

def check_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict):
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        sd = ckpt
    nan_keys = [k for k, v in sd.items() if torch.isnan(v).any()]
    total = len(sd)
    print(f"{os.path.basename(path)}: {len(nan_keys)}/{total} 层含 NaN  {'❌ 坏' if nan_keys else '✅ 正常'}")
    if nan_keys:
        print(f"  含 NaN 的层: {nan_keys[:5]}{'...' if len(nan_keys)>5 else ''}")
    return len(nan_keys) == 0

candidates = sorted(glob.glob("checkpoints/*.pth")) + ["final_model.pth"]
good = []
for p in candidates:
    if os.path.exists(p):
        if check_checkpoint(p):
            good.append(p)

print(f"\n可用的正常模型: {good}")

print("\n--- 诊断信息 ---")
print(f"各类概率均值: bg={probs[0].mean():.4f}, staff={probs[1].mean():.4f}, symbol={probs[2].mean():.4f}")
print(f"各类概率最大值: bg={probs[0].max():.4f}, staff={probs[1].max():.4f}, symbol={probs[2].max():.4f}")
print(f"置信度 mean={conf.mean():.4f}, min={conf.min():.4f}, max={conf.max():.4f}")

# 检查输入 tensor
tensor = pipeline.preprocess_image("qcwq.png", enhance=True)
print(f"\n输入 tensor shape: {tensor.shape}")
print(f"输入 tensor range: [{tensor.min():.4f}, {tensor.max():.4f}], mean={tensor.mean():.4f}")

# 保存可视化结果
from PIL import Image as PILImage
color_map = np.array([[0,0,0],[0,255,0],[255,0,0]], dtype=np.uint8)
vis = color_map[pred]
PILImage.fromarray(vis).save("result.png")
print("\n结果已保存到 result.png")
print("结果已保存到 result.png")