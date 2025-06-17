import os
import sys

def get(fname):
    local_dir = os.path.join(os.path.dirname(__file__), 'fineweb10B')
    if not os.path.exists(os.path.join(local_dir, fname)):
	print(f"curl -o {os.path.join(local_dir, fname)} https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main/{fname}")
get("fineweb_val_%06d.bin" % 0)
num_chunks = 103 # full fineweb10B. Each chunk is 100M tokens
if len(sys.argv) >= 2: # we can pass an argument to download less
    num_chunks = int(sys.argv[1])
for i in range(1, num_chunks+1):
    get("fineweb_train_%06d.bin" % i)
