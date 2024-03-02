import torch

print(
    max(range(torch.cuda.device_count()), key=lambda x: torch.cuda.mem_get_info(x)[0])
)
