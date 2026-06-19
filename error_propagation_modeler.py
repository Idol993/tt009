import torch
import torch.nn as nn
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any

from error_features import ModuleErrorTracker
from fourier_analyzer import FourierModeAnalyzer
from inverse_solver import InverseErrorPropagationSolver
from error_canceler import ErrorCancellationInjector
from module_wrappers import (
    BudgetController, _TimingHelper, wrap_model
)
from low_precision_sim import LowPrecisionSimulator


class ErrorPropagationModeler:
    def __init__(self,
                 precision: str = "fp16",
                 max_overhead_ratio: float = 0.05,
                 n_fourier_modes: int = 64,
                 max_cancellation_strength: float = 0.3,
                 solver_max_iter: int = 30,
                 use_fast_path: bool = True):
        self.original_precision = precision.lower()
        p = self.original_precision
        if p == "fp8":
            p = "fp8_e4m3"
        self.precision = p
        self.precision_sim = LowPrecisionSimulator(self.original_precision)
        
        self.tracker = ModuleErrorTracker()
        self.fourier_analyzer = FourierModeAnalyzer(n_modes=n_fourier_modes)
        self.inverse_solver = InverseErrorPropagationSolver(
            self.fourier_analyzer,
            max_iter=solver_max_iter
        )
        self.canceler = ErrorCancellationInjector(
            self.tracker,
            self.fourier_analyzer,
            self.inverse_solver,
            max_cancellation_strength=max_cancellation_strength,
            use_fast_path=use_fast_path
        )
        self.budget_controller = BudgetController(
            max_overhead_ratio=max_overhead_ratio
        )
        self.timing_helper = _TimingHelper(self.budget_controller)
        
        self._is_wrapped = False
        self._model: Optional[nn.Module] = None
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        if self._is_wrapped:
            return model
        
        device = next(model.parameters()).device
        self.tracker.set_device(device)
        
        wrapped = wrap_model(
            model, self.tracker, self.canceler,
            self.budget_controller, self.precision_sim,
            self.timing_helper
        )
        self._model = wrapped
        self._is_wrapped = True
        
        return wrapped
    
    @contextmanager
    def track_forward(self):
        self.timing_helper.start_forward()
        try:
            yield
        finally:
            self.timing_helper.end_forward()
    
    def get_overhead_ratio(self) -> float:
        return self.timing_helper.get_real_overhead_ratio()
    
    def get_cancellation_stats(self) -> Dict[str, Any]:
        segment_stats = self.timing_helper.get_segment_stats()
        total_segment_ms = sum(v['time_ms'] for v in segment_stats.values())
        
        return {
            "precision": self.precision,
            "precision_label": self.precision_sim.precision_label(),
            "overhead_ratio": self.get_overhead_ratio(),
            "module_stats": self.canceler.get_cancellation_stats(),
            "registered_modules": len(self.tracker.trackers),
            "segment_stats": segment_stats,
            "total_overhead_ms": total_segment_ms,
            "total_forward_ms": self.timing_helper._total_forward_time * 1000.0,
            "forward_count": self.timing_helper._forward_count,
        }
    
    def print_stats(self):
        stats = self.get_cancellation_stats()
        print(f"=== 误差传播建模器状态 ===")
        print(f"精度模式: {stats['precision_label']}")
        print(f"总开销比例: {stats['overhead_ratio']:.4f} ({stats['overhead_ratio']*100:.2f}%)")
        print(f"已注册模块数: {stats['registered_modules']}")
        if stats['module_stats']:
            print("各模块抵消信号统计 (前5个):")
            for i, (name, mstats) in enumerate(stats['module_stats'].items()):
                if i >= 5:
                    print(f"  ... 还有 {len(stats['module_stats']) - 5} 个模块")
                    break
                print(f"  {name}:")
                print(f"    抵消均值={mstats['cancelation_mean']:.2e}")
                print(f"    抵消最大={mstats['cancelation_max']:.2e}")
                print(f"    观测误差最大={mstats['error_max_observed']:.2e}")
                print(f"    更新次数={mstats['update_count']}")
        print("=" * 40)
    
    def run_cancellation_verification(self, module_name: str,
                                       subsequent_count: int = 1) -> Optional[dict]:
        efm = self.tracker.get_tracker(module_name)
        if efm is None or efm.error_spectrum is None:
            return None
        
        subsequent = self.canceler._get_subsequent_modules(module_name)
        subsequent = subsequent[:subsequent_count]
        
        if len(subsequent) == 0:
            return None
        
        dim = efm.error_mean.size(-1)
        device = efm.error_mean.device
        
        test_error = torch.randn(1, dim, device=device) * 0.01
        if efm.error_spectrum is not None:
            noise_fft = torch.fft.rfft(test_error, n=dim)
            weights = torch.sqrt(efm.error_spectrum + 1e-20)
            noise_fft = noise_fft * weights.unsqueeze(0)
            test_error = torch.fft.irfft(noise_fft, n=dim, dim=-1)
        
        cancelation = self.inverse_solver.solve_spectral_domain(
            efm, subsequent, output_shape=(1, dim),
            observed_error=test_error
        )
        
        result = self.inverse_solver.verify_cancellation(
            test_error, cancelation, subsequent
        )
        
        return result
    
    def reset(self):
        self.canceler.reset()
        self.budget_controller.reset()
    
    def set_device(self, device: torch.device):
        self.tracker.set_device(device)
    
    def set_enabled(self, enabled: bool):
        self.budget_controller.set_enabled(enabled)


def create_low_precision_modeler(
    precision: str = "fp16",
    **kwargs
) -> ErrorPropagationModeler:
    if precision == "fp8":
        kwargs.setdefault("max_cancellation_strength", 0.4)
        kwargs.setdefault("solver_max_iter", 40)
    elif precision == "fp16":
        kwargs.setdefault("max_cancellation_strength", 0.25)
        kwargs.setdefault("solver_max_iter", 30)
    else:
        kwargs.setdefault("max_cancellation_strength", 0.15)
        kwargs.setdefault("solver_max_iter", 20)
    
    return ErrorPropagationModeler(precision=precision, **kwargs)
