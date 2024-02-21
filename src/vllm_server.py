import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
from vllm import LLM, SamplingParams
import torch

# import time
from constants import MAX_TOKENS


tensor_parallel_size = len(os.environ.get("CUDA_VISIBLE_DEVICES").split(","))

sampling_params = SamplingParams(max_tokens=MAX_TOKENS)

llm = LLM(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    tensor_parallel_size=tensor_parallel_size,
    dtype=torch.float16,
    gpu_memory_utilization=1.0,
)
