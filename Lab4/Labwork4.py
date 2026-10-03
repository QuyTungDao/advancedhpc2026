from pathlib import Path
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda

folder = Path(__file__).resolve().parent

image = mpimg.imread(str(folder / "2021-Road-Race-Grand-Championship.jpg"))
imageHeight, imageWidth, _ = image.shape
pixelCount = imageWidth * imageHeight
flatSrc = np.ascontiguousarray(image.reshape(pixelCount, 3))

hostCPU = np.zeros((pixelCount, 3), np.uint8)
start = time.time()
for i in range(pixelCount):
    r = int(flatSrc[i, 0])
    g = int(flatSrc[i, 1])
    b = int(flatSrc[i, 2])
    gray = np.uint8((r + g + b) // 3)
    hostCPU[i, 0] = hostCPU[i, 1] = hostCPU[i, 2] = gray
cpuTime = time.time() - start
hostCPU = hostCPU.reshape(imageHeight, imageWidth, 3)


@cuda.jit
def grayscale(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if tidy < src.shape[0] and tidx < src.shape[1]:
        r = int(src[tidy, tidx, 0])
        g = int(src[tidy, tidx, 1])
        b = int(src[tidy, tidx, 2])
        gray = np.uint8((r + g + b) // 3)
        dst[tidy, tidx, 0] = gray
        dst[tidy, tidx, 1] = gray
        dst[tidy, tidx, 2] = gray


blockSize = (16, 16)
gridSize = (
    (imageWidth + blockSize[0] - 1) // blockSize[0],
    (imageHeight + blockSize[1] - 1) // blockSize[1],
)

devInput = cuda.to_device(image)
devOutput = cuda.device_array(image.shape, np.uint8)
grayscale[gridSize, blockSize](devInput, devOutput)
cuda.synchronize()

start = time.time()
devInput = cuda.to_device(image)
devOutput = cuda.device_array(image.shape, np.uint8)
grayscale[gridSize, blockSize](devInput, devOutput)
hostOutput = devOutput.copy_to_host()
gpuTime = time.time() - start

print(f"CPU: {cpuTime:.6f} s")
print(f"GPU (including data transfer): {gpuTime:.6f} s")
print(f"Speedup: {cpuTime / gpuTime:.2f}x")
print(f"Similar result: {np.array_equal(hostCPU, hostOutput)}")

plt.imsave(str(folder / "gray_cpu.png"), hostCPU)
plt.imsave(str(folder / "gray_gpu.png"), hostOutput)
