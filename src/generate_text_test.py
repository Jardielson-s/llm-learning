import torch
import tiktoken
from gpt_config import GPT_CONFIG_124M
from GPT_model import GPTModel
from generate_text import generate_text_simple

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

tokenizer = tiktoken.get_encoding("gpt2")

start_context = "Hello, I am"

encoded = tokenizer.encode(start_context)
print("Encoded: ", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("Encoded_tensor.shape: ", encoded_tensor.shape)

model.eval()
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"],
)

print("Output: ", out)
print("Output length: ", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
