import time
import torch
import numpy as np
import os
from PIL import Image
from oemer_ng.inference.pipeline import OMRPipeline


def create_dummy_images(num_images, size=(1024, 1024)):
    print(f"Creating {num_images} dummy images of size {size}...")
    images = []
    for _ in range(num_images):
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
        images.append(img)
    return images


def benchmark_predict_batch():
    num_images = 16
    batch_size = 4
    images = create_dummy_images(num_images)

    print("Initializing pipeline...")
    pipeline = OMRPipeline(num_classes=128)

    # Warmup
    print("Warmup...")
    _ = pipeline.predict_batch(images[:1], batch_size=1)

    print(f"Running benchmark with {num_images} images, batch_size={batch_size}...")
    start_time = time.time()
    predictions = pipeline.predict_batch(images, batch_size=batch_size)
    end_time = time.time()

    duration = end_time - start_time
    print(f"\nResults:")
    print(f"Total time: {duration:.4f} seconds")
    print(f"Average time per image: {duration/num_images:.4f} seconds")
    print(f"Images per second: {num_images/duration:.4f}")


if __name__ == "__main__":
    try:
        benchmark_predict_batch()
    except Exception as e:
        print(f"\nError running benchmark: {e}")
        print("This is likely due to missing dependencies in the environment.")
