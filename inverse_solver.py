import torch
from typing import List, Optional, Tuple
from error_features import ErrorFeatureMatrix, ModuleType
from fourier_analyzer import FourierModeAnalyzer


class InverseErrorPropagationSolver:
    def __init__(self, fourier_analyzer: FourierModeAnalyzer,
                 max_iter: int = 30,
                 convergence_tol: float = 1e-5,
                 damping: float = 0.3,
                 regularization: float = 1e-4,
                 lr: float = 0.05):
        self.fourier_analyzer = fourier_analyzer
        self.max_iter = max_iter
        self.convergence_tol = convergence_tol
        self.damping = damping
        self.regularization = regularization
        self.lr = lr
    
    def forward_propagate(self, initial_signal: torch.Tensor,
                          module_chain: List[Tuple[ErrorFeatureMatrix, ModuleType]]) -> torch.Tensor:
        current = initial_signal.clone()
        dim = current.size(-1)
        device = current.device
        
        for _, next_type in module_chain:
            modes = torch.fft.rfftfreq(dim, device=device)
            gain_fn = self.fourier_analyzer._module_gain_profiles.get(
                next_type, self.fourier_analyzer._default_gain_profile
            )
            gains = gain_fn(modes)
            
            signal_fft = torch.fft.rfft(current, dim=-1)
            amplified_fft = signal_fft * gains.unsqueeze(0)
            current = torch.fft.irfft(amplified_fft, n=dim, dim=-1)
            
            if next_type in [ModuleType.GELU, ModuleType.SOFTMAX, ModuleType.ATTENTION]:
                current = current * (1.0 + 0.3 * current.abs())
        
        return current
    
    def _back_propagate_gradient(self, grad_output: torch.Tensor,
                                  module_chain: List[Tuple[ErrorFeatureMatrix, ModuleType]],
                                  current_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        grad = grad_output.clone()
        dim = grad.size(-1)
        device = grad.device
        
        for i in range(len(module_chain) - 1, -1, -1):
            _, next_type = module_chain[i]
            
            if next_type in [ModuleType.GELU, ModuleType.SOFTMAX, ModuleType.ATTENTION]:
                if current_signal is not None:
                    propagated_i = self._propagate_partial(current_signal, module_chain[:i+1])
                    grad = grad * (1.0 + 0.6 * propagated_i.abs())
                else:
                    grad = grad * 1.1
            
            modes = torch.fft.rfftfreq(dim, device=device)
            gain_fn = self.fourier_analyzer._module_gain_profiles.get(
                next_type, self.fourier_analyzer._default_gain_profile
            )
            gains = gain_fn(modes)
            
            grad_fft = torch.fft.rfft(grad, dim=-1)
            grad_fft = grad_fft * gains.unsqueeze(0)
            grad = torch.fft.irfft(grad_fft, n=dim, dim=-1)
        
        return grad
    
    def _propagate_partial(self, signal, module_chain):
        if len(module_chain) == 0:
            return signal.clone()
        return self.forward_propagate(signal, module_chain)
    
    def solve_spectral_domain(self,
                               current_efm: ErrorFeatureMatrix,
                               subsequent_modules: List[Tuple[ErrorFeatureMatrix, ModuleType]],
                               output_shape: Optional[Tuple[int, ...]] = None,
                               observed_error: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = current_efm.error_mean.device
        dim = current_efm.error_mean.size(-1)
        
        if output_shape is None:
            output_shape = current_efm.output_shape
        
        n_modes = dim // 2 + 1
        modes = torch.fft.rfftfreq(dim, device=device)
        
        cumulative_gain = torch.ones(n_modes, device=device)
        for _, next_type in subsequent_modules:
            gain_fn = self.fourier_analyzer._module_gain_profiles.get(
                next_type, self.fourier_analyzer._default_gain_profile
            )
            mode_gain = gain_fn(modes)
            cumulative_gain = cumulative_gain * mode_gain
        
        cumulative_gain = torch.clamp(cumulative_gain, min=0.01, max=100.0)
        
        if observed_error is not None and observed_error.abs().sum() > 1e-20:
            flat_error = observed_error.detach().view(-1, dim)
            
            batch_size = flat_error.size(0)
            error_fft_per_sample = torch.fft.rfft(flat_error, n=dim, dim=-1)
            
            target_magnitude_per_sample = error_fft_per_sample.abs()
            target_phase_per_sample = torch.angle(error_fft_per_sample)
            
            cancelation_magnitude_per_sample = target_magnitude_per_sample / (cumulative_gain.unsqueeze(0) + self.regularization)
            cancelation_phase_per_sample = target_phase_per_sample + 3.141592653589793
            
            cancelation_fft_per_sample = cancelation_magnitude_per_sample * torch.exp(1j * cancelation_phase_per_sample)
            cancelation_signal_per_sample = torch.fft.irfft(cancelation_fft_per_sample, n=dim, dim=-1)
            
            batch_dims = output_shape[:-1]
            cancelation_signal = cancelation_signal_per_sample.view(*batch_dims + (dim,))
            
        else:
            if current_efm.error_spectrum is not None and current_efm.error_spectrum.sum() > 1e-20:
                target_power = current_efm.error_spectrum
            else:
                return torch.zeros(output_shape, device=device)
            
            cancelation_magnitude = torch.sqrt(target_power + 1e-20) / (cumulative_gain + self.regularization)
            cancelation_magnitude = torch.clamp(cancelation_magnitude, max=1e6)
            
            has_bias = current_efm.error_mean.abs().sum() > 1e-10
            if has_bias:
                mean_fft = torch.fft.rfft(current_efm.error_mean, n=dim)
                base_phase = torch.angle(mean_fft) + 3.141592653589793
            else:
                return torch.zeros(output_shape, device=device)
            
            cancelation_fft_1d = cancelation_magnitude * torch.exp(1j * base_phase)
            
            batch_dims = output_shape[:-1]
            cancelation_fft = cancelation_fft_1d.unsqueeze(0).expand(*batch_dims, -1).clone()
            cancelation_signal = torch.fft.irfft(cancelation_fft, n=dim, dim=-1)
        
        cancelation_signal = torch.nan_to_num(cancelation_signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        return cancelation_signal.detach()
    
    def solve_cancelation_signal(self,
                                  current_efm: ErrorFeatureMatrix,
                                  subsequent_modules: List[Tuple[ErrorFeatureMatrix, ModuleType]],
                                  observed_error: Optional[torch.Tensor] = None,
                                  output_shape: Optional[Tuple[int, ...]] = None,
                                  force_full_solve: bool = False) -> torch.Tensor:
        device = current_efm.error_mean.device
        
        if observed_error is not None and observed_error.numel() > 0:
            target_error = observed_error.detach()
        else:
            target_error = self._synthesize_target_error(current_efm)
        
        if output_shape is None:
            output_shape = target_error.shape
        
        has_nonlinear = any(
            mt in [ModuleType.GELU, ModuleType.SOFTMAX, ModuleType.ATTENTION]
            for _, mt in subsequent_modules
        )
        
        cancelation_fast = self.solve_spectral_domain(
            current_efm, subsequent_modules,
            output_shape=output_shape,
            observed_error=target_error
        )
        
        if not has_nonlinear and not force_full_solve:
            return cancelation_fast.detach()
        
        target_flat = target_error.detach().view(-1, target_error.size(-1))
        
        if force_full_solve:
            cancelation_flat = torch.zeros_like(target_flat)
        else:
            cancelation_flat = cancelation_fast.detach().view(-1, cancelation_fast.size(-1))
        
        best_residual = self._compute_residual(cancelation_flat, target_flat, subsequent_modules)
        best_cancelation_flat = cancelation_flat.clone()
        
        velocity = torch.zeros_like(cancelation_flat)
        current_lr = self.lr
        
        effective_iter = self.max_iter * 3 if force_full_solve else self.max_iter
        
        for iteration in range(effective_iter):
            propagated = self.forward_propagate(cancelation_flat, subsequent_modules)
            residual = propagated + target_flat
            
            grad = residual.clone()
            grad = self._back_propagate_gradient(grad, subsequent_modules, cancelation_flat)
            
            grad_norm = grad.norm()
            if grad_norm < self.regularization * 0.1:
                break
            
            velocity = self.damping * velocity + (1.0 - self.damping) * grad
            step_size = current_lr / (1.0 + iteration * 0.005)
            
            new_cancelation = cancelation_flat - step_size * velocity
            
            current_residual = self._compute_residual(new_cancelation, target_flat, subsequent_modules)
            
            if current_residual < best_residual:
                best_residual = current_residual
                best_cancelation_flat = new_cancelation.clone()
                cancelation_flat = new_cancelation
            else:
                current_lr *= 0.5
                if current_lr < 1e-7:
                    break
                cancelation_flat = best_cancelation_flat.clone()
                velocity.zero_()
            
            if iteration > 10 and best_residual < self.convergence_tol * 0.1:
                break
        
        final_cancelation = best_cancelation_flat.view(*output_shape).clone()
        
        return final_cancelation.detach()
    
    def _compute_residual(self, cancelation, target_error, module_chain) -> float:
        propagated = self.forward_propagate(cancelation, module_chain)
        return (propagated + target_error).norm().item()
    
    def _synthesize_target_error(self, efm: ErrorFeatureMatrix) -> torch.Tensor:
        device = efm.error_mean.device
        dim = efm.error_mean.size(-1)
        
        error = torch.zeros((1, dim), device=device)
        
        if efm.error_spectrum is not None and efm.error_spectrum.sum() > 1e-20:
            noise = torch.randn((1, dim), device=device)
            noise_fft = torch.fft.rfft(noise, n=dim)
            power_weights = torch.sqrt(efm.error_spectrum + 1e-20)
            noise_fft = noise_fft * power_weights.unsqueeze(0)
            error = torch.fft.irfft(noise_fft, n=dim, dim=-1)
        
        error = error + efm.error_mean.unsqueeze(0)
        
        return error.detach()
    
    def verify_cancellation(self,
                             initial_error: torch.Tensor,
                             cancelation_signal: torch.Tensor,
                             module_chain: List[Tuple[ErrorFeatureMatrix, ModuleType]]) -> dict:
        with torch.no_grad():
            propagated_error = self.forward_propagate(initial_error, module_chain)
            propagated_cancel = self.forward_propagate(cancelation_signal, module_chain)
            combined = propagated_error + propagated_cancel
            
            error_norm = propagated_error.norm().item()
            combined_norm = combined.norm().item()
            reduction_ratio = (1.0 - combined_norm / max(error_norm, 1e-20))
            
            dot_product = (propagated_error * propagated_cancel).sum().item()
            cos_similarity = dot_product / max(error_norm * propagated_cancel.norm().item(), 1e-20)
        
        return {
            "error_norm": error_norm,
            "cancel_norm_after_prop": propagated_cancel.norm().item(),
            "combined_norm": combined_norm,
            "reduction_ratio": reduction_ratio,
            "cos_similarity": cos_similarity,
            "is_opposite_phase": cos_similarity < -0.1,
            "is_improved": reduction_ratio > 0.0,
        }
