import torch
from gpt.model import CharGPT, config_from_dict
ckpt = torch.load("gpt_checkpoints/yoga_word_gpt/best.pt", map_location="cpu", weights_only=False)
config = config_from_dict(ckpt["config"])
model = CharGPT(config)
model.load_state_dict(ckpt["model"], strict=False)
print("Hyperparams: ", config)                 # hyperparams
print("Layer tree: ", model)                  # layer tree
n_params = model.count_parameters()
size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
print(f"Total params:  {n_params:,}")
print(f"Model size:    {size_mb:.1f} MB")
for name, p in model.named_parameters():
    print(f"{p.numel():10,}  {tuple(p.shape)!s:20}  {name}")