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
    print("验证 残差抵消验证 (Cancellation Verification)")
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
    
    def verify_scenario(name, test_error, subsequent_modules, require_exact_improvement=False):
        error_fft = torch.fft.rfft(test_error.mean(dim=0), n=dim)
        efm.error_spectrum = error_fft.abs().pow(2).detach()
        efm.error_mean = test_error.mean(dim=0).detach()
        efm.update_count = 10
        
        zero_cancel = torch.zeros_like(test_error)
        result_none = solver.verify_cancellation(test_error, zero_cancel, subsequent_modules)
        
        cancelation_fast = solver.solve_spectral_domain(
            efm, subsequent_modules, output_shape=test_error.shape,
            observed_error=test_error
        )
        result_fast = solver.verify_cancellation(test_error, cancelation_fast, subsequent_modules)
        
        cancelation_full = solver.solve_cancelation_signal(
            efm, subsequent_modules,
            observed_error=test_error,
            output_shape=test_error.shape
        )
        result_full = solver.verify_cancellation(test_error, cancelation_full, subsequent_modules)
        
        fast_diff = (cancelation_fast - cancelation_full).abs().max().item()
        fast_equals_full = fast_diff < 1e-8
        
        fast_better = result_fast['combined_norm'] < result_none['combined_norm'] * 0.99
        full_better_than_none = result_full['combined_norm'] < result_none['combined_norm'] * 0.99
        
        is_linear = not require_exact_improvement
        
        print(f"\n--- {name} ---")
        print(f"  {'不开抵消':<15}: 残差范数={result_none['error_norm']:.6e}")
        print(f"  {'快速路径':<15}: 残差范数={result_fast['combined_norm']:.6e}, "
              f"降低={result_fast['reduction_ratio']*100:6.2f}%, "
              f"相位cos={result_fast['cos_similarity']:.4f}")
        print(f"  {'精确路径':<15}: 残差范数={result_full['combined_norm']:.6e}, "
              f"降低={result_full['reduction_ratio']*100:6.2f}%, "
              f"相位cos={result_full['cos_similarity']:.4f}")
        
        if is_linear and fast_equals_full:
            print(f"  [信息] 线性场景精确路径复用快速路径最优解（已达理论最优）")
            full_pass = True
        elif fast_equals_full:
            print(f"  [失败] 精确路径复用了快速路径结果，未做实际迭代优化")
            full_pass = False
        else:
            full_pass = True
            if require_exact_improvement:
                print(f"  [信息] 非线性场景精确路径完成迭代优化（与快速路径差异={fast_diff:.2e}）")
        
        all_pass = True
        if not fast_better:
            print(f"  [失败] 快速路径残差未明显降低")
            all_pass = False
        if not full_better_than_none:
            print(f"  [失败] 精确路径残差未明显降低")
            all_pass = False
        if result_fast['cos_similarity'] >= -0.1:
            print(f"  [失败] 快速路径未反相（cos >= -0.1）")
            all_pass = False
        if result_full['cos_similarity'] >= -0.1:
            print(f"  [失败] 精确路径未反相（cos >= -0.1）")
            all_pass = False
        
        if all_pass and full_pass:
            print(f"  [通过] 三组对比验证全部通过")
        else:
            print(f"  [失败] 验证未通过")
            all_pass = False
        
        return all_pass
    
    torch.manual_seed(42)
    
    base_error = torch.randn(2, dim, device=device) * 0.01
    bias = torch.randn(dim, device=device) * 0.002
    test_error_bias = base_error + bias.unsqueeze(0)
    
    all_results = []
    
    all_results.append(verify_scenario(
        "恒等传播 (0层后续模块)",
        test_error_bias, [],
        require_exact_improvement=False
    ))
    
    all_results.append(verify_scenario(
        "单层Linear传播 (1层后续模块)",
        test_error_bias, [(efm, ModuleType.LINEAR)],
        require_exact_improvement=False
    ))
    
    all_results.append(verify_scenario(
        "单层GELU传播 (非线性模块)",
        test_error_bias, [(efm, ModuleType.GELU)],
        require_exact_improvement=True
    ))
    
    print("\n" + "=" * 60)
    print("验证 批次相位验证 (Batch Phase Verification)")
    print("=" * 60)
    print("\n构造正负误差相互抵消的批次（批均值接近0）：")
    
    torch.manual_seed(123)
    batch_error = torch.randn(8, dim, device=device) * 0.01
    batch_error[0::2] = batch_error[0::2] * 1.0
    batch_error[1::2] = -batch_error[0::2] * 0.9 + torch.randn(4, dim, device=device) * 0.0001
    
    batch_mean = batch_error.mean(dim=0).abs().mean().item()
    sample_max = batch_error.abs().max(dim=1)[0].mean().item()
    print(f"  批均值绝对值: {batch_mean:.2e}")
    print(f"  样本平均最大误差: {sample_max:.2e}")
    print(f"  批均值/样本误差比: {batch_mean/max(sample_max, 1e-20):.2e} (<<1说明正负抵消)")
    
    error_fft = torch.fft.rfft(batch_error.mean(dim=0), n=dim)
    efm.error_spectrum = error_fft.abs().pow(2).detach()
    efm.error_mean = batch_error.mean(dim=0).detach()
    efm.update_count = 10
    
    subsequent = []
    cancelation = solver.solve_spectral_domain(
        efm, subsequent, output_shape=batch_error.shape,
        observed_error=batch_error
    )
    result = solver.verify_cancellation(batch_error, cancelation, subsequent)
    
    per_sample_cos = []
    for i in range(batch_error.size(0)):
        e = batch_error[i:i+1]
        c = cancelation[i:i+1]
        dot = (e * c).sum().item()
        cos = dot / max(e.norm().item() * c.norm().item(), 1e-20)
        per_sample_cos.append(cos)
    
    avg_cos = sum(per_sample_cos) / len(per_sample_cos)
    min_cos = min(per_sample_cos)
    all_negative = all(c < -0.1 for c in per_sample_cos)
    nonzero_cancel = cancelation.abs().mean().item() > 1e-10
    
    print(f"\n  快速路径抵消结果:")
    print(f"    整体降低: {result['reduction_ratio']*100:.2f}%, 整体cos: {result['cos_similarity']:.4f}")
    print(f"    单样本cos均值: {avg_cos:.4f}, 最差cos: {min_cos:.4f}")
    print(f"    所有样本反相: {'是' if all_negative else '否'}")
    print(f"    非空抵消信号: {'是' if nonzero_cancel else '否'}")
    
    batch_pass = all_negative and nonzero_cancel and result['reduction_ratio'] > 0.1
    
    if batch_pass:
        print(f"  [通过] 批均值近0时仍能对每个样本生成正确反相扰动")
    else:
        print(f"  [失败] 批次相位验证未通过")
    
    all_results.append(batch_pass)
    
    print("\n" + "=" * 60)
    all_pass = all(all_results)
    if all_pass:
        print("残差抵消验证 全部通过")
    else:
        print(f"残差抵消验证 存在失败项 ({sum(all_results)}/{len(all_results)} 通过)")
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
    print(f"总前向步数: {final_stats['forward_count']}")
    
    if 'segment_stats' in final_stats:
        seg_stats = final_stats['segment_stats']
        total_training_ms = avg_time * n_steps * 1000
        print("\n=== 分项开销统计 (累计耗时) ===")
        print(f"{'项目':<22} {'累计(ms)':<12} {'占总训练%':<12} {'占前向%':<12}")
        print(f"{'-'*60}")
        segment_labels = {
            'reference_computation': '高精度参考计算',
            'spectrum_analysis': '误差谱分析',
            'inverse_solver': '逆传播求解',
            'noise_injection': '噪声注入',
        }
        for seg_key, label in segment_labels.items():
            if seg_key in seg_stats:
                time_ms = seg_stats[seg_key]['time_ms']
                pct_of_fwd = seg_stats[seg_key]['pct']
                pct_of_total = (time_ms / max(total_training_ms, 1e-10)) * 100
                print(f"{label:<22} {time_ms:<12.1f} {pct_of_total:<12.2f} {pct_of_fwd:<12.2f}")
        
        total_seg_ms = sum(seg_stats[k]['time_ms'] for k in segment_labels if k in seg_stats)
        total_seg_pct = sum(seg_stats[k]['pct'] for k in segment_labels if k in seg_stats)
        total_pct_of_training = (total_seg_ms / max(total_training_ms, 1e-10)) * 100
        print(f"{'-'*60}")
        print(f"{'合计':<22} {total_seg_ms:<12.1f} {total_pct_of_training:<12.2f} {total_seg_pct:<12.2f}")
        print(f"总训练耗时: {total_training_ms:.1f}ms, 总前向耗时: {final_stats.get('total_forward_ms', 0):.1f}ms")
    
    if final_stats['module_stats']:
        print("\n模块观测误差统计 (前3个):")
        for i, (name, mstats) in enumerate(final_stats['module_stats'].items()):
            if i >= 3:
                break
            if mstats['error_max_observed'] > 0:
                print(f"  {name}: 观测最大误差={mstats['error_max_observed']:.2e}, 更新次数={mstats['update_count']}")
    
    return {
        "precision": precision,
        "losses": losses,
        "avg_time_ms": avg_time * 1000,
        "overhead_ratio": final_stats['overhead_ratio'],
        "n_modules": final_stats['registered_modules'],
        "modeler": modeler,
    }


def main():
    import sys
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    verify_cancellation_identity()
    
    vocab_size = 1000
    batch_size = 4
    seq_len = 32
    d_model = 128
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    if len(sys.argv) > 1:
        precision_arg = sys.argv[1].lower()
        if precision_arg in ["fp16", "fp8", "fp32"]:
            precisions_to_run = [precision_arg]
        else:
            print(f"未知精度: {precision_arg}，支持: fp16, fp8")
            precisions_to_run = ["fp16", "fp8"]
    else:
        precisions_to_run = ["fp16", "fp8"]
    
    results = {}
    
    for prec in precisions_to_run:
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
