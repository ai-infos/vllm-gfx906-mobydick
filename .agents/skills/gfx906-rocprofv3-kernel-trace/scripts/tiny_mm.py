import os
import torch, time
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(3): a @ b
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10): a @ b
torch.cuda.synchronize()
print(f"MM done {time.time()-t0:.3f}s", flush=True)
if os.environ.get("FLUSH_WAIT"):
    time.sleep(float(os.environ["FLUSH_WAIT"]))
import os
