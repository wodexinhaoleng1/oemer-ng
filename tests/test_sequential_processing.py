#!/usr/bin/env python3
"""
测试：使用集成了ONNX清理功能的ImagePreprocessor处理图片
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add src directory to Python path
# This is a common pattern in tests to ensure the source code is importable
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

from oemer_ng.utils.preprocessing import ImagePreprocessor


def main():
    """主函数：使用集成的预处理器进行一步处理"""
    print("=" * 70)
    print(" 测试: 使用集成的 ImagePreprocessor 进行一步处理")
    print(" (ONNX 清理 + Resize + 归一化)")
    print("=" * 70)

    # --- Configuration ---
    INPUT_IMAGE = "qcwq.png"
    ONNX_MODEL_PATH = "src/resize_model/best.onnx"
    FINAL_OUTPUT = "qcwq_final_processed.png"

    # Preprocessing parameters
    TARGET_SIZE = (512, 512)
    ONNX_IMAGE_SIZE = 640

    try:
        # --- File Checks ---
        if not Path(INPUT_IMAGE).exists():
            print(f"\n❌ Error: Input image not found: {INPUT_IMAGE}")
            return

        if not Path(ONNX_MODEL_PATH).exists():
            print(f"\n❌ Error: ONNX model not found: {ONNX_MODEL_PATH}")
            return

        # ============================================================
        # Step 1: Initialize the Integrated ImagePreprocessor
        # ============================================================
        print(f"\n{'='*70}")
        print("步骤 1/2: 初始化 ImagePreprocessor")
        print(f"{'='*70}")
        print(f"\n[1.1] 创建集成的预处理器...")

        preprocessor = ImagePreprocessor(
            target_size=TARGET_SIZE,
            normalize=True,
            onnx_model_path=ONNX_MODEL_PATH,
            onnx_image_size=ONNX_IMAGE_SIZE,
        )

        print(f"      - 目标尺寸: {TARGET_SIZE}")
        print(f"      - ONNX 模型: {ONNX_MODEL_PATH}")
        print("      - 预处理器创建成功.")

        # ============================================================
        # Step 2: Process the Image in a Single Step
        # ============================================================
        print(f"\n{'='*70}")
        print("步骤 2/2: 执行一步预处理")
        print(f"{'='*70}")

        print(f"\n[2.1] 从文件加载并处理图像: {INPUT_IMAGE}")

        # The preprocess method now handles everything:
        # 1. Loads the image.
        # 2. Cleans it using the ONNX model.
        # 3. Resizes and pads it.
        # 4. Normalizes it.
        preprocessed_image = preprocessor.preprocess(INPUT_IMAGE, return_tensor=False)

        print(f"      - 预处理完成!")
        print(f"      - 输出形状: {preprocessed_image.shape}")
        print(f"      - 输出类型: {preprocessed_image.dtype}")
        print(f"      - 输出范围: [{preprocessed_image.min():.2f}, {preprocessed_image.max():.2f}]")

        # ============================================================
        # Save the Final Result
        # ============================================================
        print("\n[2.2] 保存最终处理结果...")

        # Denormalize for visualization
        # Note: The denormalization logic depends on the mean/std used.
        # This is a simplified version for visual inspection.
        if preprocessed_image.dtype in [np.float32, np.float64]:
            # Assuming mean=0.5, std=0.5 for simple reversal. For exact reversal, use preprocessor.mean/std.
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            # Check if image is in C, H, W format (tensor format) or H, W, C format (numpy format)
            if preprocessed_image.shape[0] == 3:  # C, H, W format
                img_to_save = np.transpose(preprocessed_image, (1, 2, 0))
            else:  # H, W, C format
                img_to_save = preprocessed_image

            # Denormalize
            img_to_save = (img_to_save * std) + mean
            img_to_save = (img_to_save * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img_to_save = preprocessed_image

        cv2.imwrite(FINAL_OUTPUT, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
        print(f"      - 已保存为: {FINAL_OUTPUT}")

        # ============================================================
        # Summary
        # ============================================================
        original_shape = cv2.imread(INPUT_IMAGE).shape
        final_shape = img_to_save.shape

        print(f"\n{'='*70}")
        print("处理流程总结")
        print(f"{'='*70}")
        print(f"  - 输入图像: {INPUT_IMAGE} (尺寸: {original_shape})")
        print(f"  - 处理步骤: ONNX Cleanup -> Resize -> Normalize")
        print(f"  - 输出图像: {FINAL_OUTPUT} (尺寸: {final_shape})")
        print(f"{'='*70}")

    except Exception as e:
        print(f"\n❌ 严重错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
