import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Optional

from error_propagation_modeler import ErrorPropagationModeler, create_low_precision_modeler


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = q @ k.transpose(-2, -1) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = self.softmax(scores)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff2(self.gelu(self.ff1(self.ln2(x))))
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 3, d_ff: int = 512,
                 max_seq_len: int = 64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(positions)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_final(x)
        return self.head(x)


def train_example():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    vocab_size = 1000
    batch_size = 4
    seq_len = 32
    d_model = 128
    
    model = SimpleTransformer(vocab_size, d_model=d_model).to(device)
    
    print("\n=== 无补偿训练 (基准) ===")
    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    model.train()
    total_time_base = 0.0
    for step in range(10):
        t0 = time.perf_counter()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        dt = time.perf_counter() - t0
        total_time_base += dt
        if step % 5 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}, time={dt*1000:.2f}ms")
    
    avg_time_base = total_time_base / 10
    print(f"  平均前向+反向时间: {avg_time_base*1000:.2f}ms")
    
    print("\n=== 低精度误差传播建模器训练 ===")
    model2 = SimpleTransformer(vocab_size, d_model=d_model).to(device)
    
    modeler = create_low_precision_modeler(
        precision="fp32",
        max_overhead_ratio=0.05
    )
    model_wrapped = modeler.wrap_model(model2)
    model_wrapped = model_wrapped.to(device)
    
    optimizer2 = torch.optim.AdamW(model_wrapped.parameters(), lr=1e-3)
    model_wrapped.train()
    
    total_time_comp = 0.0
    for step in range(20):
        t0 = time.perf_counter()
        with modeler.track_forward():
            logits = model_wrapped(x)
        
        loss = F.cross_entropy(logits.float().view(-1, vocab_size), y.view(-1))
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        
        dt = time.perf_counter() - t0
        total_time_comp += dt
        
        if step % 5 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}, time={dt*1000:.2f}ms")
            print(f"    当前开销比例: {modeler.get_overhead_ratio()*100:.2f}%")
        
        if step == 15:
            modeler.print_stats()
    
    avg_time_comp = total_time_comp / 20
    print(f"\n  平均前向+反向时间: {avg_time_comp*1000:.2f}ms")
    print(f"  相对基准速度: {avg_time_comp/avg_time_base:.2f}x")
    
    final_stats = modeler.get_cancellation_stats()
    print(f"\n最终开销比例: {final_stats['overhead_ratio']*100:.2f}%")
    print(f"是否满足<5%要求: {'是' if final_stats['overhead_ratio'] < 0.05 else '否'}")
    
    print("\n=== 训练完成 ===")


if __name__ == "__main__":
    train_example()
