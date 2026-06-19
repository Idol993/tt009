import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from error_features import ModuleErrorTracker, ModuleType
from error_canceler import ErrorCancellationInjector


class BudgetController:
    def __init__(self, max_overhead_ratio: float = 0.05,
                 initial_skip: int = 20):
        self.max_overhead_ratio = max_overhead_ratio
        
        self._call_counter: int = 0
        self._compensation_run_count: int = 0
        self._total_calls: int = 0
        
        self._current_skip: int = initial_skip
        self._enabled: bool = True
        
        self._actual_overhead_ratio: float = 0.0
    
    def record_forward_time(self, t: float):
        pass
    
    def record_overhead_time(self, t: float):
        pass
    
    def should_run_cancellation(self) -> bool:
        if not self._enabled:
            return False
        
        self._call_counter += 1
        self._total_calls += 1
        
        if self._call_counter > self._current_skip:
            self._call_counter = 0
            self._compensation_run_count += 1
            self._actual_overhead_ratio = self._compensation_run_count / max(self._total_calls, 1)
            
            if self._actual_overhead_ratio > self.max_overhead_ratio * 1.2:
                self._current_skip = min(self._current_skip + 2, 100)
            elif self._actual_overhead_ratio < self.max_overhead_ratio * 0.5:
                self._current_skip = max(2, self._current_skip - 1)
            
            return True
        return False
    
    def get_current_overhead_ratio(self) -> float:
        if self._total_calls == 0:
            return 0.0
        return self._compensation_run_count / self._total_calls
    
    def reset(self):
        self._call_counter = 0
        self._compensation_run_count = 0
        self._total_calls = 0
    
    def set_enabled(self, enabled: bool):
        self._enabled = enabled


class ErrorCompensatedLinear(nn.Module):
    def __init__(self, linear: nn.Linear,
                 name: str,
                 tracker: ModuleErrorTracker,
                 canceler: ErrorCancellationInjector,
                 budget_controller: BudgetController,
                 enable_reference_fp32: bool = True,
                 reference_interval: int = 10):
        super().__init__()
        self.linear = linear
        self.name = name
        self.tracker = tracker
        self.canceler = canceler
        self.budget_controller = budget_controller
        self.enable_reference_fp32 = enable_reference_fp32
        self.reference_interval = reference_interval
        
        self._is_registered = False
        self._call_count = 0
    
    def _register(self, input_shape):
        output_features = self.linear.out_features
        output_shape = list(input_shape[:-1]) + [output_features]
        self.tracker.register_module(
            self.name, ModuleType.LINEAR,
            tuple(input_shape), tuple(output_shape)
        )
        self._is_registered = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_registered:
            self._register(x.shape)
        
        self._call_count += 1
        
        if not self.budget_controller.should_run_cancellation():
            return self.linear(x)
        
        output_low = self.linear(x)
        
        output_high = None
        should_compute_ref = (
            self.enable_reference_fp32 and self.training
            and self._call_count % self.reference_interval == 0
        )
        if should_compute_ref:
            with torch.no_grad():
                weight_fp32 = self.linear.weight.float()
                bias_fp32 = self.linear.bias.float() if self.linear.bias is not None else None
                input_fp32 = x.float()
                output_high = F.linear(input_fp32, weight_fp32, bias_fp32)
        
        output_compensated = self.canceler.compute_and_inject(
            self.name, output_low, output_high
        )
        
        return output_compensated


class ErrorCompensatedLayerNorm(nn.Module):
    def __init__(self, norm: nn.LayerNorm,
                 name: str,
                 tracker: ModuleErrorTracker,
                 canceler: ErrorCancellationInjector,
                 budget_controller: BudgetController,
                 enable_reference_fp32: bool = True,
                 reference_interval: int = 10):
        super().__init__()
        self.norm = norm
        self.name = name
        self.tracker = tracker
        self.canceler = canceler
        self.budget_controller = budget_controller
        self.enable_reference_fp32 = enable_reference_fp32
        self.reference_interval = reference_interval
        
        self._is_registered = False
        self._call_count = 0
    
    def _register(self, input_shape):
        self.tracker.register_module(
            self.name, ModuleType.LAYERNORM,
            tuple(input_shape), tuple(input_shape)
        )
        self._is_registered = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_registered:
            self._register(x.shape)
        
        self._call_count += 1
        
        if not self.budget_controller.should_run_cancellation():
            return self.norm(x)
        
        output_low = self.norm(x)
        
        output_high = None
        should_compute_ref = (
            self.enable_reference_fp32 and self.training
            and self._call_count % self.reference_interval == 0
        )
        if should_compute_ref:
            with torch.no_grad():
                output_high = F.layer_norm(
                    x.float(), self.norm.normalized_shape,
                    self.norm.weight.float() if self.norm.weight is not None else None,
                    self.norm.bias.float() if self.norm.bias is not None else None,
                    self.norm.eps
                )
        
        output_compensated = self.canceler.compute_and_inject(
            self.name, output_low, output_high
        )
        
        return output_compensated


class ErrorCompensatedGELU(nn.Module):
    def __init__(self, name: str,
                 tracker: ModuleErrorTracker,
                 canceler: ErrorCancellationInjector,
                 budget_controller: BudgetController,
                 approximate: str = 'none',
                 reference_interval: int = 10):
        super().__init__()
        self.name = name
        self.tracker = tracker
        self.canceler = canceler
        self.budget_controller = budget_controller
        self.approximate = approximate
        self.reference_interval = reference_interval
        
        self._is_registered = False
        self._call_count = 0
    
    def _register(self, input_shape):
        self.tracker.register_module(
            self.name, ModuleType.GELU,
            tuple(input_shape), tuple(input_shape)
        )
        self._is_registered = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_registered:
            self._register(x.shape)
        
        self._call_count += 1
        
        if not self.budget_controller.should_run_cancellation():
            return F.gelu(x, approximate=self.approximate)
        
        output_low = F.gelu(x, approximate=self.approximate)
        
        output_high = None
        should_compute_ref = (
            self.training
            and self._call_count % self.reference_interval == 0
        )
        if should_compute_ref:
            with torch.no_grad():
                output_high = F.gelu(x.float(), approximate='none')
        
        output_compensated = self.canceler.compute_and_inject(
            self.name, output_low, output_high
        )
        
        return output_compensated


class ErrorCompensatedSoftmax(nn.Module):
    def __init__(self, name: str,
                 tracker: ModuleErrorTracker,
                 canceler: ErrorCancellationInjector,
                 budget_controller: BudgetController,
                 dim: int = -1,
                 reference_interval: int = 10):
        super().__init__()
        self.name = name
        self.tracker = tracker
        self.canceler = canceler
        self.budget_controller = budget_controller
        self.dim = dim
        self.reference_interval = reference_interval
        
        self._is_registered = False
        self._call_count = 0
    
    def _register(self, input_shape):
        self.tracker.register_module(
            self.name, ModuleType.SOFTMAX,
            tuple(input_shape), tuple(input_shape)
        )
        self._is_registered = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_registered:
            self._register(x.shape)
        
        self._call_count += 1
        
        if not self.budget_controller.should_run_cancellation():
            return F.softmax(x, dim=self.dim)
        
        output_low = F.softmax(x, dim=self.dim)
        
        output_high = None
        should_compute_ref = (
            self.training
            and self._call_count % self.reference_interval == 0
        )
        if should_compute_ref:
            with torch.no_grad():
                output_high = F.softmax(x.float(), dim=self.dim)
        
        output_compensated = self.canceler.compute_and_inject(
            self.name, output_low, output_high
        )
        
        return output_compensated


def wrap_module(module: nn.Module,
                name: str,
                tracker: ModuleErrorTracker,
                canceler: ErrorCancellationInjector,
                budget_controller: BudgetController,
                **kwargs) -> nn.Module:
    if isinstance(module, nn.Linear):
        return ErrorCompensatedLinear(module, name, tracker, canceler, budget_controller, **kwargs)
    elif isinstance(module, nn.LayerNorm):
        return ErrorCompensatedLayerNorm(module, name, tracker, canceler, budget_controller, **kwargs)
    elif isinstance(module, nn.GELU):
        return ErrorCompensatedGELU(name, tracker, canceler, budget_controller, **kwargs)
    elif isinstance(module, nn.Softmax):
        return ErrorCompensatedSoftmax(name, tracker, canceler, budget_controller, dim=module.dim, **kwargs)
    else:
        return module


def wrap_model(model: nn.Module,
               tracker: ModuleErrorTracker,
               canceler: ErrorCancellationInjector,
               budget_controller: BudgetController,
               prefix: str = "") -> nn.Module:
    for name, child in list(model.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        
        if isinstance(child, (nn.Linear, nn.LayerNorm, nn.GELU, nn.Softmax)):
            setattr(model, name, wrap_module(child, full_name, tracker, canceler, budget_controller))
        else:
            wrap_model(child, tracker, canceler, budget_controller, full_name)
    
    return model
