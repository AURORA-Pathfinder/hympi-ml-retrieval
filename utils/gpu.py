import subprocess as sb
import os

def set_gpus(count: int = 1, min_free: float = 0.99, verbose: bool = False):
    '''
    Sets the environment variable "CUDA_VISIBLE_DEVICES" for tensorflow GPU 
    training.

    Calculated by finding the GPUs with the most available memory. With a given 
    count value that determines the number of GPUs to use (default is 1).

    In order to be set as a gpu, the GPU must have a free / total ratio greater
    than the 'min-free' parameter (default is 0.99).
    '''

    cmd = "nvidia-smi --query-gpu=memory.free,memory.total --format=csv"
    lines = sb.check_output(cmd.split()).decode("ascii").splitlines()[1:]

    free_mems = []

    for i in range(len(lines)):
        splits = lines[i].split()

        free = float(splits[0])
        total = float(splits[2])

        mem = free / total

        if mem > min_free:
            free_mems.append((i, mem))

    sorted(free_mems, key=lambda x: x[1])

    indices = [free_mem[0] for free_mem in free_mems[:count]]

    env_str = ""
    for index in indices:
        env_str += f"{index},"

    os.environ["CUDA_VISIBLE_DEVICES"] = env_str
    os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"

    if (verbose):
        print(f"Set environment variable, CUDA_VISIBLE_DEVICES, to {env_str}.")
