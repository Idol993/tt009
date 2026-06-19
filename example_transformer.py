import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Optional

from error_propagation_modeler import ErrorPropagationModeler, create_low_precision_modeler
from error_features import ModuleType
from fourier_analyzer import FourierModeAnalyzer
from inverse_solver import InverseErrorPropagationSolver


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


def verify_cancellation_identity():
    print("=" * 60)
    print("验证 恒等传播残差验证 (Identity Propagation Verification)")
    print("=" * 60)
    
    device = torch.device("cpu")
    dim = 128
    
    analyzer = FourierModeAnalyzer(n_modes=32)
    solver = InverseErrorPropagationSolver(analyzer, max_iter=50, lr=0.1)
    
    from error_features import ErrorFeatureMatrix
    
    efm = ErrorFeatureMatrix(
        name="test_layer",
        module_type=ModuleType.LINEAR,
        input_shape=(1, dim),
        output_shape=(1, dim),
    )
    efm.initialize_tensors(device)
    
    torch.manual_seed(42)
    
    test_error = torch.randn(2, dim, device=device) * 0.01
    bias = torch.randn(dim, device=device) * 0.002
    test_error = test_error + bias.unsqueeze(0)
    
    error_fft = torch.fft.rfft(test_error.mean(dim=0), n=dim)
    efm.error_spectrum = error_fft.abs().pow(2).detach()
    efm.error_mean = test_error.mean(dim=0).detach()
    efm.update_count = 10
    
    subsequent_modules = []
    
    print("\n--- 恒等传播 (0层后续模块，即误差=抵消目标) ---")
    
    cancelation_fast = solver.solve_spectral_domain(
        efm, subsequent_modules, output_shape=test_error.shape,
        observed_error=test_error
    )
    result_fast = solver.verify_cancellation(test_error, cancelation_fast, subsequent_modules)
    
    print(f"  快速路径:")
    print(f"    原始误差范数: {result_fast['error_norm']:.6e}")
    print(f"    抵消后残差范数: {result_fast['combined_norm']:.6e}")
    print(f"    残差降低比例: {result_fast['reduction_ratio']*100:.2f}%")
    print(f"    相位余弦相似度: {result_fast['cos_similarity']:.4f} (负为反相)")
    print(f"    验证通过: {'是' if result_fast['is_improved'] else '否'}")
    
    cancelation_full = solver.solve_cancelation_signal(
        efm, subsequent_modules,
        observed_error=test_error,
        output_shape=test_error.shape
    )
    result_full = solver.verify_cancellation(test_error, cancelation_full, subsequent_modules)
    
    print(f"\n  精确路径:")
    print(f"    原始误差范数: {result_full['error_norm']:.6e}")
    print(f"    抵消后残差范数: {result_full['combined_norm']:.6e}")
    print(f"    残差降低比例: {result_full['reduction_ratio']*100:.2f}%")
    print(f"    相位余弦相似度: {result_full['cos_similarity']:.4f} (负为反相)")
    print(f"    验证通过: {'是' if result_full['is_improved'] else '否'}")
    
    print("\n--- 单层线性传播 (1层后续Linear模块) ---")
    
    subsequent_linear = [(efm, ModuleType.LINEAR)]
    
    cancelation_linear = solver.solve_spectral_domain(
        efm, subsequent_linear, output_shape=test_error.shape,
        observed_error=test_error
    )
    result_linear = solver.verify_cancellation(test_error, cancelation_linear, subsequent_linear)
    
    print(f"  快速路径 + Linear 传播:")
    print(f"    传播后误差范数: {result_linear['error_norm']:.6e}")
    print(f"    抵消后残差范数: {result_linear['combined_norm']:.6e}")
    print(f"    残差降低比例: {result_linear['reduction_ratio']*100:.2f}%")
    print(f"    相位余弦相似度: {result_linear['cos_similarity']:.4f}")
    print(f"    验证通过: {'是' if result_linear['is_improved'] else '否'}")
    
    print("\n--- 单层GELU传播 (非线性放大验证) ---")
    
    subsequent_gelu = [(efm, ModuleType.GELU)]
    
    cancelation_gelu = solver.solve_spectral_domain(
        efm, subsequent_gelu, output_shape=test_error.shape,
        observed_error=test_error
    )
    result_gelu = solver.verify_cancellation(test_error, cancelation_gelu, subsequent_gelu)
    
    print(f"  快速路径 + GELU 传播:")
    print(f"    传播后误差范数: {result_gelu['error_norm']:.6e}")
    print(f"    抵消后残差范数: {result_gelu['combined_norm']:.6e}")
    print(f"    残差降低比例: {result_gelu['reduction_ratio']*100:.2f}%")
    print(f"    相位余弦相似度: {result_gelu['cos_similarity']:.4f}")
    print(f"    验证通过: {'是' if result_gelu['is_improved'] else '否'}")
    
    cancelation_gelu_full = solver.solve_cancelation_signal(
        efm, subsequent_gelu,
        observed_error=test_error,
        output_shape=test_error.shape
    )
    result_gelu_full = solver.verify_cancellation(test_error, cancelation_gelu_full, subsequent_gelu)
    
    print(f"\n  精确路径 + GELU 传播 (非线性迭代优化):")
    print(f"    传播后误差范数: {result_gelu_full['error_norm']:.6e}")
    print(f"    抵消后残差范数: {result_gelu_full['combined_norm']:.6e}")
    print(f"    残差降低比例: {result_gelu_full['reduction_ratio']*100:.2f}%")
    print(f"    相位余弦相似度: {result_gelu_full['cos_similarity']:.4f}")
    print(f"    比快速路径更优: {'是' if result_gelu_full['reduction_ratio'] > result_gelu['reduction_ratio'] else '否'}")
    print(f"    验证通过: {'是' if result_gelu_full['is_improved'] else '否'}")
    
    print("\n" + "=" * 60)
    all_pass = (
        result_fast['is_improved'] and result_fast['reduction_ratio'] > 0.2 and result_fast['cos_similarity'] < -0.3 and
        result_full['is_improved'] and result_full['cos_similarity'] < -0.3 and
        result_linear['is_improved'] and result_linear['cos_similarity'] < -0.3 and
        result_gelu['is_improved'] and result_gelu['cos_similarity'] < -0.3 and
        result_gelu_full['is_improved'] and result_gelu_full['cos_similarity'] < -0.3
    )
    print(f"残差抵消验证 {'全部通过' if all_pass else '存在失败项'}")
    print("=" * 60)
    
    return all_pass


def train_with_precision(precision: str, vocab_size: int, d_model: int,
                      x: torch.Tensor, y: torch.Tensor,
                      device: torch.device, n_steps: int = 60) -> dict:
    print(f"\n{'='*60}")
    print(f"  低精度训练测试 - {precision.upper()} 模式")
    print(f"{'='*60}")
    
    model = SimpleTransformer(vocab_size, d_model=d_model, n_layers=2).to(device)
    
    modeler = create_low_precision_modeler(
        precision=precision,
        max_overhead_ratio=0.05
    )
    model_wrapped = modeler.wrap_model(model)
    model_wrapped = model_wrapped.to(device)
    
    optimizer = torch.optim.AdamW(model_wrapped.parameters(), lr=1e-3)
    model_wrapped.train()
    
    print(f"精度: {modeler.precision_sim.precision_label()}")
    print(f"目标开销上限: 5%")
    
    losses = []
    times = []
    
    for step in range(n_steps):
        t0 = time.perf_counter()
        with modeler.track_forward():
            logits = model_wrapped(x)
        
        loss = F.cross_entropy(logits.float().view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        dt = time.perf_counter() - t0
        losses.append(loss.item())
        times.append(dt)
        
        if step % 5 == 0:
            oh = modeler.get_overhead_ratio() * 100
            print(f"  Step {step:2d}: loss={loss.item():.4f}, time={dt*1000:.1f}ms, 开销={oh:.2f}%")
    
    avg_time = sum(times) / len(times)
    final_stats = modeler.get_cancellation_stats()
    
    print(f"\n平均前向+反向时间: {avg_time*1000:.1f}ms")
    print(f"最终开销比例: {final_stats['overhead_ratio']*100:.2f}%")
    print(f"是否满足<5%: {'是' if final_stats['overhead_ratio'] < 0.05 else '否'}")
    print(f"已注册模块数: {final_stats['registered_modules']}")
    
    if final_stats['module_stats']:
        print("\n模块观测误差统计 (前3个):")
        for i, (name, mstats) in enumerate(final_stats['module_stats'].items()):
            if i >= 3:
                break
            if mstats['error_max_observed'] > 0:
                print(f"  {name}: 观测最大误差={mstats['error_max_observed']:.2e}")
    
    return {
        "precision": precision,
        "losses": losses,
        "avg_time_ms": avg_time * 1000,
        "overhead_ratio": final_stats['overhead_ratio'],
        "n_modules": final_stats['registered_modules'],
        "modeler": modeler,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    verify_cancellation_identity()
    
    vocab_size = 1000
    batch_size = 4
    seq_len = 32
    d_model = 128
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    results = {}
    
    for prec in ["fp16", "fp8_e4m3"]:
        results[prec] = train_with_precision(
            prec, vocab_size, d_model, x, y, device, n_steps=80
        )
    
    print(f"\n{'='*60}")
    print(f"  汇总对比")
    print(f"{'='*60}")
    print(f"{'精度':<12} {'平均耗时(ms)':<14} {'开销比例':<12} {'模块数':<8}")
    print(f"{'-'*50}")
    for prec, res in results.items():
        oh_pct = f"{res['overhead_ratio']*100:.2f}%"
        print(f"{prec:<12} {res['avg_time_ms']:<14.1f} {oh_pct:<12} {res['n_modules']:<8}")
    
    all_ok = all(r['overhead_ratio'] < 0.05 for r in results.values())
    print(f"\n开销全部<5%: {'是' if all_ok else '否'}")
    print("\n=== 全部测试完成 ===")


if __name__ == "__main__":
    main()
