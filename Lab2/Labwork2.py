import numba.cuda

device = numba.cuda.select_device(0)
free, total = numba.cuda.current_context().get_memory_info()

print(f"Device's name: {device.name}")
print(f"Multiprocessor: {device.MULTIPROCESSOR_COUNT}")
print(f"Number of cores: {device.MULTIPROCESSOR_COUNT * 128}")
print(f"Total memory: {total / 1024**3:.2f} GiB")
print(f"Free memory: {free / 1024**3:.2f} GiB")
