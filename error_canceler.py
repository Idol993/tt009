import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from error_features import ErrorFeatureMatrix, ModuleErrorTracker, ModuleType
from fourier_analyzer import FourierModeAnalyzer
from inverse_solver import InverseErrorPropagationSolver


class ErrorCancellationInjector:
    def __init__(self,
                 tracker: ModuleErrorTracker,
                 fourier_analyzer: FourierModeAnalyzer,
                 inverse_solver: InverseErrorPropagationSolver,
                 max_cancellation_strength: float = 0.3,
                 adaptive_strength: bool = True,
                 use_fast_path: bool = True):
        self.tracker = tracker
        self.fourier_analyzer = fourier_analyzer
        self.inverse_solver = inverse_solver
        self.max_cancellation_strength = max_cancellation_strength
        self.adaptive_strength = adaptive_strength
        self.use_fast_path = use_fast_path
        
        self._module_execution_order: List[str] = []
        self._strength_history: dict = {}
    
    def register_execution_order(self, module_name: str):
        if module_name not in self._module_execution_order:
            self._module_execution_order.append(module_name)
    
    def _get_subsequent_modules(self, current_name: str) -> List[Tuple[ErrorFeatureMatrix, ModuleType]]:
        try:
            current_idx = self._module_execution_order.index(current_name)
        except ValueError:
            return []
        
        subsequent = []
        for name in self._module_execution_order[current_idx + 1:]:
            efm = self.tracker.get_tracker(name)
            if efm is not None:
                subsequent.append((efm, efm.module_type))
        return subsequent
    
    def _estimate_rounding_error(self,
                                  x_low: torch.Tensor,
                                  x_high: torch.Tensor,
                                  efm: ErrorFeatureMatrix) -> torch.Tensor:
        if x_high is None or x_low is None:
            return torch.zeros_like(x_low) if x_low is not None else torch.zeros(1)
        
        error = x_high.float() - x_low.float()
        return error.detach()
    
    def _compute_adaptive_strength(self, efm: ErrorFeatureMatrix,
                                     subsequent: List[Tuple[ErrorFeatureMatrix, ModuleType]]) -> float:
        if not self.adaptive_strength:
            return self.max_cancellation_strength
        
        total_amplification = 1.0
        for sub_efm, mtype in subsequent:
            if sub_efm.amplification_factors is not None:
                total_amplification *= sub_efm.amplification_factors.mean().item()
            else:
                total_amplification *= 1.5
        
        strength = min(self.max_cancellation_strength, 1.0 / (1.0 + total_amplification * 0.1))
        strength = max(0.01, strength)
        
        return strength
    
    def compute_and_inject(self,
                            module_name: str,
                            output_low: torch.Tensor,
                            output_high: Optional[torch.Tensor] = None,
                            force_full_solve: bool = False) -> torch.Tensor:
        efm = self.tracker.get_tracker(module_name)
        if efm is None:
            return output_low
        
        self.register_execution_order(module_name)
        
        if output_high is not None:
            estimated_error = self._estimate_rounding_error(output_low, output_high, efm)
            self.fourier_analyzer.analyze_error_spectrum(estimated_error, efm)
            efm.update_error_stats(estimated_error)
        
        subsequent = self._get_subsequent_modules(module_name)
        
        if len(subsequent) == 0:
            return output_low
        
        for sub_efm, mtype in subsequent[:1]:
            self.fourier_analyzer.predict_amplification(efm, mtype)
        
        strength = self._compute_adaptive_strength(efm, subsequent)
        
        if self.use_fast_path and not force_full_solve:
            cancelation = self.inverse_solver.solve_spectral_domain(
                efm, subsequent, output_low.shape
            )
        else:
            cancelation = self.inverse_solver.solve_cancelation_signal(
                efm, subsequent,
                observed_error=estimated_error if output_high is not None else None,
                output_shape=output_low.shape
            )
        
        output_scale = output_low.abs().mean().item() + 1e-8
        cancelation_scale = cancelation.abs().mean().item() + 1e-8
        max_allowed = output_scale * self.max_cancellation_strength
        scale_factor = min(1.0, max_allowed / cancelation_scale) * strength
        
        final_cancelation = cancelation * scale_factor
        efm.cancelation_signal = final_cancelation.detach()
        
        if output_low.dtype in [torch.float16, torch.bfloat16]:
            final_cancelation = final_cancelation.to(output_low.dtype)
        
        return output_low + final_cancelation
    
    def get_cancellation_stats(self) -> dict:
        stats = {}
        for name, efm in self.tracker.get_all_trackers():
            if efm.cancelation_signal is not None:
                stats[name] = {
                    "cancelation_mean": efm.cancelation_signal.abs().mean().item(),
                    "cancelation_max": efm.cancelation_signal.abs().max().item(),
                    "error_max_observed": efm.running_stats.max_abs,
                    "update_count": efm.update_count,
                }
        return stats
    
    def reset(self):
        self._module_execution_order = []
        self._strength_history = {}
