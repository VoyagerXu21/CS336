import torch
from transformer_lm import TransformerLM

model = TransformerLM(
    vocab_size=1000,
    context_length=128,
    num_layers=2,
    d_model=256,
    num_heads=8,
    d_ff=512,
    dropout=0.0,
    use_rope=True,
)

x = torch.randint(0, 1000, (4, 32), dtype=torch.long)
logits = model(x)
print(logits.shape)  # (4, 32, 1000)

# loss（可选）
targets = torch.randint(0, 1000, (4, 32), dtype=torch.long)
logits, loss = model(x, targets=targets, return_loss=True)
print(loss.item())

