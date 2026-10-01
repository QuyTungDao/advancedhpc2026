from pathlib import Path
import statistics
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

@cuda.jit
def grayscale(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tidx < src.shape[0]:
        r = int(src[tidx, 0])
        g = int(src[tidx, 1])
        b = int(src[tidx, 2])
        gray = np.uint8((r + g + b) // 3)
        dst[tidx, 0] = gray
        dst[tidx, 1] = gray
        dst[tidx, 2] = gray

blockSize = 64
gridSize = (pixelCount + blockSize - 1) // blockSize

devInput = cuda.to_device(flatSrc)
devOutput = cuda.device_array((pixelCount, 3), np.uint8)
grayscale[gridSize, blockSize](devInput, devOutput)
cuda.synchronize()

start = time.time()
devInput = cuda.to_device(flatSrc)
devOutput = cuda.device_array((pixelCount, 3), np.uint8)
grayscale[gridSize, blockSize](devInput, devOutput)
hostOutput = devOutput.copy_to_host()
gpuTime = time.time() - start

print(f"CPU: {cpuTime:.6f} s")
print(f"GPU (Data transfer including): {gpuTime:.6f} s")
print(f"Speedup: {cpuTime / gpuTime:.2f}x")
print(f"Similar result: {np.array_equal(hostCPU, hostOutput)}")

plt.imsave(str(folder / "gray_cpu.png"),
           hostCPU.reshape(imageHeight, imageWidth, 3))
plt.imsave(str(folder / "gray_gpu.png"),
           hostOutput.reshape(imageHeight, imageWidth, 3))

blockSizes = [32, 64, 128, 256, 512]
timesMs = []

for blockSize in blockSizes:
    gridSize = (pixelCount + blockSize - 1) // blockSize
    samples = []

    for _ in range(30):
        start = time.time()
        grayscale[gridSize, blockSize](devInput, devOutput)
        cuda.synchronize()
        samples.append((time.time() - start) * 1000)

    timesMs.append(statistics.median(samples))
