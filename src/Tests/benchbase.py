import torch
import time
mat = torch.rand(4096, 4096, dtype=torch.float32)
vec = torch.rand(4096, dtype=torch.float32)

for _ in range(100):
    _ = mat @ vec

iters = 1000
start = time.perf_counter()
for _ in range(iters):
    _ = mat @ vec
end = time.perf_counter()

avg_ms = (end - start) / iters * 1000
print(f"pytorch matvec 4096x4096: {avg_ms:.4f} ms avg ({iters} iters)")

print(f"threads: {torch.get_num_threads()}")
print(f"blas: {torch.__config__.parallel_info()}")
