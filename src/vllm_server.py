
from vllm import LLM, SamplingParams
import torch
#import time
#import os
from constants import MAX_TOKENS


#tensor_parallel_size = int(os.environ.get("DEVICES", "1"))

sampling_params = SamplingParams(max_tokens=MAX_TOKENS)

llm = LLM(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    #tensor_parallel_size=tensor_parallel_size,
    dtype=torch.bfloat16,
    gpu_memory_utilization=1.0
)