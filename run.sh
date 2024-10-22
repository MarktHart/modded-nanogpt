OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) train.py 2> /dev/null
