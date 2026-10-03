from pathlib import Path
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda, int32

folder = Path(__file__).resolve().parent

image = np.ascontiguousarray(
    mpimg.imread(str(folder / "2021-Road-Race-Grand-Championship.jpg"))
)
imageHeight, imageWidth, _ = image.shape

gaussianFilter = np.array([
    [0, 0, 1, 2, 1, 0, 0],
    [0, 3, 13, 22, 13, 3, 0],
    [1, 13, 59, 97, 59, 13, 1],
    [2, 22, 97, 159, 97, 22, 2],
    [1, 13, 59, 97, 59, 13, 1],
    [0, 3, 13, 22, 13, 3, 0],
    [0, 0, 1, 2, 1, 0, 0],
], dtype=np.int32)


@cuda.jit
def gaussian_blur(src, dst, weights):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if tidy < src.shape[0] and tidx < src.shape[1]:
        r = 0
        g = 0
        b = 0
        for j in range(7):
            y = tidy + j - 3
            if y < 0:
                y = 0
            elif y >= src.shape[0]:
                y = src.shape[0] - 1
            for i in range(7):
                x = tidx + i - 3
                if x < 0:
                    x = 0
                elif x >= src.shape[1]:
                    x = src.shape[1] - 1
                weight = weights[j, i]
                r += int(src[y, x, 0]) * weight
                g += int(src[y, x, 1]) * weight
                b += int(src[y, x, 2]) * weight
        dst[tidy, tidx, 0] = np.uint8(r // 1003)
        dst[tidy, tidx, 1] = np.uint8(g // 1003)
        dst[tidy, tidx, 2] = np.uint8(b // 1003)


@cuda.jit
def gaussian_blur_shared(src, dst, weights):
    sharedFilter = cuda.shared.array((7, 7), dtype=int32)
    tid = cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x
    threadsPerBlock = cuda.blockDim.x * cuda.blockDim.y
    for k in range(tid, 49, threadsPerBlock):
        sharedFilter[k // 7, k % 7] = weights[k // 7, k % 7]

    cuda.syncthreads()

    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if tidy < src.shape[0] and tidx < src.shape[1]:
        r = 0
        g = 0
        b = 0
        for j in range(7):
            y = tidy + j - 3
            if y < 0:
                y = 0
            elif y >= src.shape[0]:
                y = src.shape[0] - 1
            for i in range(7):
                x = tidx + i - 3
                if x < 0:
                    x = 0
                elif x >= src.shape[1]:
                    x = src.shape[1] - 1
                weight = sharedFilter[j, i]
                r += int(src[y, x, 0]) * weight
                g += int(src[y, x, 1]) * weight
                b += int(src[y, x, 2]) * weight
        dst[tidy, tidx, 0] = np.uint8(r // 1003)
        dst[tidy, tidx, 1] = np.uint8(g // 1003)
        dst[tidy, tidx, 2] = np.uint8(b // 1003)


blockSize = (16, 16)
gridSize = (
    (imageWidth + blockSize[0] - 1) // blockSize[0],
    (imageHeight + blockSize[1] - 1) // blockSize[1],
)

devInput = cuda.to_device(image)
devFilter = cuda.to_device(gaussianFilter)
devOutput = cuda.device_array(image.shape, np.uint8)
devOutputShared = cuda.device_array(image.shape, np.uint8)

gaussian_blur[gridSize, blockSize](devInput, devOutput, devFilter)
gaussian_blur_shared[gridSize, blockSize](devInput, devOutputShared, devFilter)
cuda.synchronize()

start = time.time()
gaussian_blur[gridSize, blockSize](devInput, devOutput, devFilter)
cuda.synchronize()
gpuTime = time.time() - start

start = time.time()
gaussian_blur_shared[gridSize, blockSize](devInput, devOutputShared, devFilter)
cuda.synchronize()
sharedTime = time.time() - start

hostOutput = devOutput.copy_to_host()
hostOutputShared = devOutputShared.copy_to_host()

print(f"GPU without shared memory: {gpuTime:.6f} s")
print(f"GPU with shared memory: {sharedTime:.6f} s")
print(f"Speedup: {gpuTime / sharedTime:.2f}x")
print(f"Similar result: {np.array_equal(hostOutput, hostOutputShared)}")

plt.imsave(str(folder / "blur_gpu.png"), hostOutput)
plt.imsave(str(folder / "blur_gpu_shared.png"), hostOutputShared)
