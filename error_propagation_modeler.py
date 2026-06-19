import torch
import torch.nn as nn
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any

from error_features import ModuleErrorTracker
from fourier_analyzer import FourierModeAnalyzer
from inverse_solver import InverseErrorPropagationSolver
from error_canceler import ErrorCancellationInjector
from module_wrappers import BudgetController, wrap_model


class ErrorPropagationModeler:
    def __init__(self,
                 max_overhead_ratio: float = 0.05,
                 n_fourier_modes: int = 64,
                 max_cancellation_strength: float = 0.3,
                 solver_max_iter: int = 30,
                 use_fast_path: bool = True):
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
        
        self._is_wrapped = False
        self._model: Optional[nn.Module] = None
        
        self._forward_start: float = 0.0
        self._overhead_accumulator: float = 0.0
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        if self._is_wrapped:
            return model
        
        device = next(model.parameters()).device
        self.tracker.set_device(device)
        
        wrapped = wrap_model(model, self.tracker, self.canceler, self.budget_controller)
        self._model = wrapped
        self._is_wrapped = True
        
        return wrapped
    
    @contextmanager
    def track_forward(self):
        self._overhead_accumulator = 0.0
        self._forward_start = time.perf_counter()
        overhead_start = time.perf_counter()
        
        try:
            yield
            self._overhead_accumulator += time.perf_counter() - overhead_start
        finally:
            total_forward = time.perf_counter() - self._forward_start
            self.budget_controller.record_forward_time(total_forward)
            self.budget_controller.record_overhead_time(self._overhead_accumulator)
    
    def record_overhead_segment(self, duration: float):
        self._overhead_accumulator += duration
    
    def get_overhead_ratio(self) -> float:
        return self.budget_controller.get_current_overhead_ratio()
    
    def get_cancellation_stats(self) -> Dict[str, Any]:
        return {
            "overhead_ratio": self.get_overhead_ratio(),
            "module_stats": self.canceler.get_cancellation_stats(),
            "registered_modules": len(self.tracker.trackers),
        }
    
    def print_stats(self):
        stats = self.get_cancellation_stats()
        print(f"=== 误差传播建模器状态 ===")
        print(f"总开销比例: {stats['overhead_ratio']:.4f} ({stats['overhead_ratio']*100:.2f}%)")
        print(f"已注册模块数: {stats['registered_modules']}")
        if stats['module_stats']:
            print("各模块抵消信号统计:")
            for name, mstats in stats['module_stats'].items():
                print(f"  {name}:")
                print(f"    抵消均值={mstats['cancelation_mean']:.2e}")
                print(f"    抵消最大={mstats['cancelation_max']:.2e}")
                print(f"    观测误差最大={mstats['error_max_observed']:.2e}")
                print(f"    更新次数={mstats['update_count']}")
        print("=" * 40)
    
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
    
    return ErrorPropagationModeler(**kwargs)
