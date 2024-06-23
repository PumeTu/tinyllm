import time
import torch
import torch.nn.functional as F
import tiktoken
from tinyllm.gpt import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")

torch.set_float32_matmul_precision('high')

seed = 1337
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        enc = tiktoken.get_encoding('gpt2')
        with open('data/shakespeare/input.txt', 'r') as f:
            text = f.read()
        self.tokens = torch.tensor(enc.encode(text))
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        x, y = buf[:-1].view(B, T), buf[1:].view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

train_loader = DataLoader(B=4, T=1024)

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    tok_per_sec = (train_loader.B * train_loader.T) / (t1-t0)
    print(f"step {i}, loss: {loss.item()}, dt: {(t1-t0)*1000:.2f} ms, tok/sec: {tok_per_sec:2f}")

import sys; sys.exit(0)

num_return_sequence = 5
max_length = 30

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequence, 1)
x = tokens.to('cuda')

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits, _ = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequence):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)