import torch
import time
import tiktoken

# mat = torch.rand(4096, 4096, dtype=torch.float32)
# vec = torch.rand(4096, dtype=torch.float32)
#
# for _ in range(100):
#     _ = mat @ vec
#
# iters = 1000
# start = time.perf_counter()
# for _ in range(iters):
#     _ = mat @ vec
# end = time.perf_counter()
#
# avg_ms = (end - start) / iters * 1000
# print(f"pytorch matvec 4096x4096: {avg_ms:.4f} ms avg ({iters} iters)")
#
# print(f"threads: {torch.get_num_threads()}")
# print(f"blas: {torch.__config__.parallel_info()}")

enc = tiktoken.get_encoding("cl100k_base")

short_text = "abcdefghijklmnopqrs"
medium_text = "a" * 1000
long_text = "a" * 5000
roundtrip_text = "a" * 2000

short_tokens = enc.encode("hello world 12345")
long_tokens = enc.encode("a" * 2000)

for _ in range(100):
    enc.encode(short_text)
    enc.encode(medium_text)
    enc.encode(long_text)
    enc.decode(short_tokens)
    enc.decode(long_tokens)
    enc.decode(enc.encode(roundtrip_text))

start = time.perf_counter()
for _ in range(10000):
    enc.encode(short_text)
end = time.perf_counter()
print(f"encode 19 chars: {(end - start) / 10000 * 1000:.9f} ms avg (10000 iters)")

start = time.perf_counter()
for _ in range(1000):
    enc.encode(medium_text)
end = time.perf_counter()
print(f"encode 1000 chars: {(end - start) / 1000 * 1000:.9f} ms avg (1000 iters)")

start = time.perf_counter()
for _ in range(200):
    enc.encode(long_text)
end = time.perf_counter()
print(f"encode 5000 chars: {(end - start) / 200 * 1000:.9f} ms avg (200 iters)")

start = time.perf_counter()
for _ in range(50000):
    enc.decode(short_tokens)
end = time.perf_counter()
print(f"decode 20 tokens: {(end - start) / 50000 * 1000:.9f} ms avg (50000 iters)")

start = time.perf_counter()
for _ in range(5000):
    enc.decode(long_tokens)
end = time.perf_counter()
print(f"decode 2000 tokens: {(end - start) / 5000 * 1000:.9f} ms avg (5000 iters)")

start = time.perf_counter()
for _ in range(500):
    enc.decode(enc.encode(roundtrip_text))
end = time.perf_counter()
print(f"roundtrip 2000 chars: {(end - start) / 500 * 1000:.9f} ms avg (500 iters)")
