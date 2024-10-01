import subprocess as sb
import os


def set_gpus(count: int = 1, min_free: float = 0.99, verbose: bool = False):
    """
    Sets the environment variable "CUDA_VISIBLE_DEVICES" for tensorflow GPU training.

    Calculated by finding the GPUs with the most available memory first.

    Also sets "TF_GPU_ALLOCATOR" to "cuda_malloc_async" which makes it so that GPU
    memory is not completely taken over by Tensorflow.

    Args:
        count (int, optional): The number of GPUs to make visible. Defaults to 1.
        min_free (float, optional):
            The minimum free / total memory ratio required for a GPU to be selected.
            Defaults to 0.99.
        verbose (bool, optional):
            Whether to print the list of visible GPUs to the console. Defaults to False.
    """
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

    sorted(free_mems, key=lambda x: -x[1])

    indices = [free_mem[0] for free_mem in free_mems[:count]]

    env_str = ""
    for index in indices:
        env_str += f"{index},"

    os.environ["CUDA_VISIBLE_DEVICES"] = env_str
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    if verbose:
        print(f"Set environment variable, CUDA_VISIBLE_DEVICES, to {env_str}.")
