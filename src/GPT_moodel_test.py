import torch
import tiktoken
from GPT_model import GPTModel
from gpt_config import GPT_CONFIG_124M

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
tokenizer = tiktoken.get_encoding("gpt2")

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

print(batch)
print("Input batch:\n", batch)

out = model(batch)

print("\nOutput shape: ", out.shape)
print(out)
