import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Optional, Callable
from error_features import ModuleErrorTracker, ModuleType
from error_canceler import ErrorCancellationInjector
from low_precision_sim import LowPrecisionSimulator


class BudgetController:
    def __init__(self, max_overhead_ratio: float = 0.05,
                 window_size: int = 10,
                 initial_skip: int = 9):
        self.max_overhead_ratio = max_overhead_ratio
        self.window_size = window_size
        
        self._baseline_forward_times: list = []
        self._active_forward_times: list = []
        self._active_overhead_times: list = []
        
        self._enabled: bool = True
        self._forward_counter: int = 0
        self._skip_counter: int = 0
        self._current_skip: int = initial_skip
        self._warmup_remaining: int = 5
        
        self._last_overhead_ratio: float = 0.0
        self._is_current_forward_active: bool = False
    
    def start_forward(self) -> bool:
        if not self._enabled:
            self._is_current_forward_active = False
            return False
        
        self._forward_counter += 1
        
        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            self._is_current_forward_active = False
            return False
        
        if self._skip_counter > 0:
            self._skip_counter -= 1
            self._is_current_forward_active = False
        else:
            self._is_current_forward_active = True
        
        return self._is_current_forward_active
    
    def end_forward(self, forward_time: float, overhead_time: float):
        if not self._enabled:
            return
        
        if self._is_current_forward_active:
            self._active_forward_times.append(forward_time)
            self._active_overhead_times.append(overhead_time)
            
            if len(self._active_forward_times) > self.window_size:
                self._active_forward_times.pop(0)
                self._active_overhead_times.pop(0)
        else:
            self._baseline_forward_times.append(forward_time)
            if len(self._baseline_forward_times) > self.window_size:
                self._baseline_forward_times.pop(0)
        
        if len(self._baseline_forward_times) >= 2 and len(self._active_forward_times) >= 1:
            avg_baseline = sum(self._baseline_forward_times) / len(self._baseline_forward_times)
            avg_active = sum(self._active_forward_times) / len(self._active_forward_times)
            
            if avg_baseline > 0:
                single_overhead_ratio = (avg_active - avg_baseline) / avg_baseline
                
                active_frequency = 1.0 / (self._current_skip + 1)
                self._last_overhead_ratio = single_overhead_ratio * active_frequency
                
                target_skip = max(1, int(single_overhead_ratio / self.max_overhead_ratio * 1.2) - 1)
                
                if self._current_skip < target_skip:
                    self._current_skip = min(self._current_skip * 2, target_skip + 20)
                elif self._current_skip > target_skip * 3:
                    self._current_skip = max(1, self._current_skip - 2)
                
                self._skip_counter = self._current_skip
    
    def should_run_cancellation(self) -> bool:
        return self._is_current_forward_active and self._enabled
    
    def get_current_overhead_ratio(self) -> float:
        return min(self._last_overhead_ratio, 1.0)
    
    def reset(self):
        self._baseline_forward_times = []
        self._active_forward_times = []
        self._active_overhead_times = []
        self._forward_counter = 0
        self._skip_counter = 0
        self._current_skip = 19
        self._last_overhead_ratio = 0.0
        self._is_current_forward_active = False
    
    def set_enabled(self, enabled: bool):
        self._enabled = enabled


class _TimingHelper:
    def __init__(self, budget_controller: BudgetController):
        self.bc = budget_controller
        self._fwd_start: float = 0.0
        self._oh_accum: float = 0.0
        self._oh_start: float = 0.0
        self._is_active: bool = False
    
    def start_forward(self):
        self._is_active = self.bc.start_forward()
        self._fwd_start = time.perf_counter()
        self._oh_accum = 0.0
    
    def start_overhead_segment(self):
        self._oh_start = time.perf_counter()
    
    def end_overhead_segment(self):
        self._oh_accum += time.perf_counter() - self._oh_start
    
    def end_forward(self):
        total_fwd = time.perf_counter() - self._fwd_start
        self.bc.end_forward(total_fwd, self._oh_accum)


class ErrorCompensatedLinear(nn.Module):
    def __init__(self, linear: nn.Linear,
                 name: str,
                 tracker: ModuleErrorTracker,
                 canceler: ErrorCancellationInjector,
                 budget_controller: BudgetController,
                 precision_sim: LowPrecisionSimulator,
                 timing_helper: _TimingHelper):
        super().__init__()
        self.linear = linear
        self.name = name
        self.tracker = tracker
        self.canceler = canceler
        self.budget_controller = budget_controller
        self.precision_sim = precision_sim
        self.timing_helper = timing_helper
        
        self._is_registered = False
    
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
        
        if not self.budget_controller.should_run_cancellation():
            return self.linear(x)
        
        self.timing_helper.start_overhead_segment()
        
        use_native = self.precision_sim.is_native_low_precision()
        
        if use_native:
            dtype = self.precision_sim.get_dtype()
            x_low = x.to(dtype)
            w_low = self.linear.weight.to(dtype)
            b_low = self.linear.bias.to(dtype) if self.linear.bias is not None else None
            output_low = F.linear(x_low, w_low, b_low).float()
        else:
            output_fp32 = F.linear(x.float(), self.linear.weight.float(),
                                     self.linear.bias.float() if self.linear.bias is not None else None)
            output_low = self.precision_sim.quantize(output_fp32)
        
        with torch.no_grad():
            output_high = F.linear(
                x.float(),
                self.linear.weight.float(),
                self.linear.bias.float() if self.linear.bias is not None else None
            )
            observed_error = output_low - output_high
        
        output_compensated = self.canceler.compute_and_inject(
            self.name, output_low, output_high
        )
        
        self.timing_helper.end_overhead_segment()
        
        return output_compensated


class ErrorCompensatedLayerNorm(nn.Module):
    def __init__(self, norm: nn.LayerNorm,
                 name: str,
                 tracker: ModuleErrorTracker,
                 canceler: ErrorCancellationInjector,
                 budget_controller: BudgetController,
                 precision_sim: LowPrecisionSimulator,
                 timing_helper: _TimingHelper):
        super().__init__()
        self.norm = norm
        self.name = name
        self.tracker = tracker
        self.canceler = canceler
        self.budget_controller = budget_controller
        self.precision_sim = precision_sim
        self.timing_helper = timing_helper
        
        self._is_registered = False
    
    def _register(self, input_shape):
        self.tracker.register_module(
            self.name, ModuleType.LAYERNORM,
            tuple(input_shape), tuple(input_shape)
        )
        self._is_registered = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_registered:
            self._register(x.shape)
        
        if not self.budget_controller.should_run_cancellation():
            return self.norm(x)
        
        self.timing_helper.start_overhead_segment()
        
        use_native = self.precision_sim.is_native_low_precision()
        
        if use_native:
            dtype = self.precision_sim.get_dtype()
            x_low = x.to(dtype)
            w_low = self.norm.weight.to(dtype) if self.norm.weight is not None else None
            b_low = self.norm.bias.to(dtype) if self.norm.bias is not None else None
            output_low = F.layer_norm(
                x_low, self.norm.normalized_shape, w_low, b_low, self.norm.eps
            ).float()
        else:
            output_fp32 = F.layer_norm(
                x.float(), self.norm.normalized_shape,
                self.norm.weight.float() if self.norm.weight is not None else None,
                self.norm.bias.float() if self.norm.bias is not None else None,
                self.norm.eps
            )
            output_low = self.precision_sim.quantize(output_fp32)
        
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
        
        self.timing_helper.end_overhead_segment()
        
        return output_compensated


class ErrorCompensatedGELU(nn.Module):
    def __init__(self, name: str,
                 tracker: ModuleErrorTracker,
                 canceler: ErrorCancellationInjector,
                 budget_controller: BudgetController,
                 precision_sim: LowPrecisionSimulator,
                 timing_helper: _TimingHelper,
                 approximate: str = 'none'):
        super().__init__()
        self.name = name
        self.tracker = tracker
        self.canceler = canceler
        self.budget_controller = budget_controller
        self.precision_sim = precision_sim
        self.timing_helper = timing_helper
        self.approximate = approximate
        
        self._is_registered = False
    
    def _register(self, input_shape):
        self.tracker.register_module(
            self.name, ModuleType.GELU,
            tuple(input_shape), tuple(input_shape)
        )
        self._is_registered = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_registered:
            self._register(x.shape)
        
        if not self.budget_controller.should_run_cancellation():
            return F.gelu(x, approximate=self.approximate)
        
        self.timing_helper.start_overhead_segment()
        
        use_native = self.precision_sim.is_native_low_precision()
        
        if use_native:
            dtype = self.precision_sim.get_dtype()
            output_low = F.gelu(x.to(dtype), approximate=self.approximate).float()
        else:
            output_fp32 = F.gelu(x.float(), approximate='none')
            output_low = self.precision_sim.quantize(output_fp32)
        
        with torch.no_grad():
            output_high = F.gelu(x.float(), approximate='none')
        
        output_compensated = self.canceler.compute_and_inject(
            self.name, output_low, output_high
        )
        
        self.timing_helper.end_overhead_segment()
        
        return output_compensated


class ErrorCompensatedSoftmax(nn.Module):
    def __init__(self, name: str,
                 tracker: ModuleErrorTracker,
                 canceler: ErrorCancellationInjector,
                 budget_controller: BudgetController,
                 precision_sim: LowPrecisionSimulator,
                 timing_helper: _TimingHelper,
                 dim: int = -1):
        super().__init__()
        self.name = name
        self.tracker = tracker
        self.canceler = canceler
        self.budget_controller = budget_controller
        self.precision_sim = precision_sim
        self.timing_helper = timing_helper
        self.dim = dim
        
        self._is_registered = False
    
    def _register(self, input_shape):
        self.tracker.register_module(
            self.name, ModuleType.SOFTMAX,
            tuple(input_shape), tuple(input_shape)
        )
        self._is_registered = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_registered:
            self._register(x.shape)
        
        if not self.budget_controller.should_run_cancellation():
            return F.softmax(x, dim=self.dim)
        
        self.timing_helper.start_overhead_segment()
        
        use_native = self.precision_sim.is_native_low_precision()
        
        if use_native:
            dtype = self.precision_sim.get_dtype()
            output_low = F.softmax(x.to(dtype), dim=self.dim).float()
        else:
            output_fp32 = F.softmax(x.float(), dim=self.dim)
            output_low = self.precision_sim.quantize(output_fp32)
        
        with torch.no_grad():
            output_high = F.softmax(x.float(), dim=self.dim)
        
        output_compensated = self.canceler.compute_and_inject(
            self.name, output_low, output_high
        )
        
        self.timing_helper.end_overhead_segment()
        
        return output_compensated


def wrap_module(module: nn.Module,
                name: str,
                tracker: ModuleErrorTracker,
                canceler: ErrorCancellationInjector,
                budget_controller: BudgetController,
                precision_sim: LowPrecisionSimulator,
                timing_helper: _TimingHelper,
                **kwargs) -> nn.Module:
    if isinstance(module, nn.Linear):
        return ErrorCompensatedLinear(
            module, name, tracker, canceler, budget_controller,
            precision_sim, timing_helper, **kwargs
        )
    elif isinstance(module, nn.LayerNorm):
        return ErrorCompensatedLayerNorm(
            module, name, tracker, canceler, budget_controller,
            precision_sim, timing_helper, **kwargs
        )
    elif isinstance(module, nn.GELU):
        return ErrorCompensatedGELU(
            name, tracker, canceler, budget_controller,
            precision_sim, timing_helper, approximate=kwargs.get('approximate', 'none'), **kwargs
        )
    elif isinstance(module, nn.Softmax):
        return ErrorCompensatedSoftmax(
            name, tracker, canceler, budget_controller,
            precision_sim, timing_helper, dim=module.dim, **kwargs
        )
    else:
        return module


def wrap_model(model: nn.Module,
               tracker: ModuleErrorTracker,
               canceler: ErrorCancellationInjector,
               budget_controller: BudgetController,
               precision_sim: LowPrecisionSimulator,
               timing_helper: _TimingHelper,
               prefix: str = "") -> nn.Module:
    for name, child in list(model.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        
        if isinstance(child, (nn.Linear, nn.LayerNorm, nn.GELU, nn.Softmax)):
            setattr(model, name, wrap_module(
                child, full_name, tracker, canceler,
                budget_controller, precision_sim, timing_helper
            ))
        else:
            wrap_model(child, tracker, canceler, budget_controller,
                       precision_sim, timing_helper, full_name)
    
    return model
