import torch
from typing import List, Optional, Tuple
from error_features import ErrorFeatureMatrix, ModuleType
from fourier_analyzer import FourierModeAnalyzer


class InverseErrorPropagationSolver:
    def __init__(self, fourier_analyzer: FourierModeAnalyzer,
                 max_iter: int = 50,
                 convergence_tol: float = 1e-6,
                 damping: float = 0.1,
                 regularization: float = 1e-4):
        self.fourier_analyzer = fourier_analyzer
        self.max_iter = max_iter
        self.convergence_tol = convergence_tol
        self.damping = damping
        self.regularization = regularization
    
    def _forward_propagate_error(self, initial_error: torch.Tensor,
                                 module_chain: List[Tuple[ErrorFeatureMatrix, ModuleType]]) -> torch.Tensor:
        current_error = initial_error.clone()
        
        for efm, next_type in module_chain:
            if efm.error_spectrum is None:
                continue
            
            dim = current_error.size(-1)
            device = current_error.device
            
            error_fft = torch.fft.rfft(current_error, dim=-1)
            
            modes = torch.fft.rfftfreq(dim, device=device)
            gain_fn = self.fourier_analyzer._module_gain_profiles.get(
                next_type, self.fourier_analyzer._default_gain_profile
            )
            gains = gain_fn(modes)
            
            amplified_fft = error_fft * gains.unsqueeze(0)
            current_error = torch.fft.irfft(amplified_fft, n=dim, dim=-1)
            
            if next_type in [ModuleType.GELU, ModuleType.SOFTMAX, ModuleType.ATTENTION]:
                nonlinear_factor = 1.0 + 0.5 * current_error.abs()
                current_error = current_error * nonlinear_factor
        
        return current_error
    
    def _compute_adjoint_gradient(self, target_error: torch.Tensor,
                                   cancelation_signal: torch.Tensor,
                                   module_chain: List[Tuple[ErrorFeatureMatrix, ModuleType]]) -> torch.Tensor:
        propagated = self._forward_propagate_error(cancelation_signal, module_chain)
        
        residual = propagated + target_error
        loss = 0.5 * (residual ** 2).sum()
        
        grad_output = residual
        
        for i in range(len(module_chain) - 1, -1, -1):
            efm, next_type = module_chain[i]
            
            if efm.error_spectrum is None:
                continue
            
            dim = grad_output.size(-1)
            device = grad_output.device
            
            modes = torch.fft.rfftfreq(dim, device=device)
            gain_fn = self.fourier_analyzer._module_gain_profiles.get(
                next_type, self.fourier_analyzer._default_gain_profile
            )
            gains = gain_fn(modes)
            
            if next_type in [ModuleType.GELU, ModuleType.SOFTMAX, ModuleType.ATTENTION]:
                prev_signal = self._forward_propagate_error(cancelation_signal, module_chain[:i])
                nonlinear_grad = 0.5 * torch.sign(prev_signal)
                grad_output = grad_output * (1.0 + nonlinear_grad.abs())
            
            grad_fft = torch.fft.rfft(grad_output, dim=-1)
            grad_fft = grad_fft * gains.unsqueeze(0)
            grad_output = torch.fft.irfft(grad_fft, n=dim, dim=-1)
        
        return grad_output
    
    def solve_cancelation_signal(self,
                                  current_efm: ErrorFeatureMatrix,
                                  subsequent_modules: List[Tuple[ErrorFeatureMatrix, ModuleType]],
                                  observed_error: Optional[torch.Tensor] = None,
                                  output_shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        device = current_efm.error_mean.device
        
        if observed_error is not None:
            target_error = observed_error
        else:
            target_error = self._synthesize_target_error(current_efm)
        
        if output_shape is None:
            output_shape = target_error.shape
        
        cancelation = torch.zeros(output_shape, device=device)
        momentum = torch.zeros_like(cancelation)
        
        best_cancelation = cancelation.clone()
        best_residual = float('inf')
        
        for iteration in range(self.max_iter):
            grad = self._compute_adjoint_gradient(target_error, cancelation, subsequent_modules)
            
            grad_norm = grad.norm() + self.regularization
            normalized_grad = grad / grad_norm
            
            momentum = self.damping * momentum + (1 - self.damping) * normalized_grad
            step_size = 0.1 / (1.0 + iteration * 0.05)
            
            cancelation = cancelation - step_size * momentum
            
            propagated = self._forward_propagate_error(cancelation, subsequent_modules)
            residual = (propagated + target_error).norm().item()
            
            if residual < best_residual:
                best_residual = residual
                best_cancelation = cancelation.clone()
            
            if iteration > 5 and abs(best_residual - residual) < self.convergence_tol:
                break
        
        return -best_cancelation.detach()
    
    def _synthesize_target_error(self, efm: ErrorFeatureMatrix) -> torch.Tensor:
        device = efm.error_mean.device
        dim = efm.error_mean.size(-1)
        
        base_shape = list(efm.output_shape)
        if len(base_shape) < 2:
            base_shape = [1] + base_shape
        
        error = torch.randn(base_shape, device=device) * 0.01
        
        if efm.error_spectrum is not None:
            error_fft = torch.fft.rfft(error, dim=-1)
            spectral_weights = torch.sqrt(efm.error_spectrum + 1e-10)
            spectral_weights = spectral_weights / (spectral_weights.sum() + 1e-10)
            error_fft = error_fft * spectral_weights.unsqueeze(0)
            error = torch.fft.irfft(error_fft, n=dim, dim=-1)
        
        error = error + efm.error_mean.unsqueeze(0)
        
        return error.detach()
    
    def solve_spectral_domain(self,
                               current_efm: ErrorFeatureMatrix,
                               subsequent_modules: List[Tuple[ErrorFeatureMatrix, ModuleType]],
                               output_shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        device = current_efm.error_mean.device
        dim = current_efm.error_mean.size(-1)
        
        if output_shape is None:
            output_shape = current_efm.output_shape
        
        n_modes = dim // 2 + 1
        modes = torch.fft.rfftfreq(dim, device=device)
        
        cumulative_gain = torch.ones(n_modes, device=device)
        for efm, next_type in subsequent_modules:
            gain_fn = self.fourier_analyzer._module_gain_profiles.get(
                next_type, self.fourier_analyzer._default_gain_profile
            )
            mode_gain = gain_fn(modes)
            cumulative_gain = cumulative_gain * mode_gain
        
        cumulative_gain = torch.clamp(cumulative_gain, min=0.01, max=100.0)
        
        if current_efm.error_spectrum is not None and current_efm.error_spectrum.sum() > 1e-20:
            target_spectrum = current_efm.error_spectrum
        else:
            target_spectrum = torch.ones(n_modes, device=device) * 1e-6
        
        target_spectrum = torch.nan_to_num(target_spectrum, nan=1e-6, posinf=1e-6, neginf=1e-6)
        
        cancelation_magnitude = target_spectrum / (cumulative_gain + self.regularization)
        cancelation_magnitude = torch.clamp(cancelation_magnitude, max=1e6)
        
        cancelation_phase = torch.zeros(n_modes, dtype=torch.float32, device=device)
        if current_efm.error_mean.abs().sum() > 1e-20:
            mean_fft = torch.fft.rfft(current_efm.error_mean)
            cancelation_phase = torch.angle(mean_fft) + 3.141592653589793
        
        cancelation_spectrum = cancelation_magnitude * torch.exp(1j * cancelation_phase)
        cancelation_spectrum = torch.nan_to_num(cancelation_spectrum, nan=0.0, posinf=0.0, neginf=0.0)
        
        batch_dims = output_shape[:-1]
        cancelation_fft = cancelation_spectrum.unsqueeze(0).expand(*batch_dims, -1).clone()
        cancelation_signal = torch.fft.irfft(cancelation_fft, n=dim, dim=-1)
        
        cancelation_signal = torch.nan_to_num(cancelation_signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        target_energy = target_spectrum.sum().clamp(min=1e-20)
        cancel_energy = cancelation_signal.pow(2).mean().clamp(min=1e-20)
        scale_factor = torch.sqrt(target_energy / cancel_energy) * 0.3
        scale_factor = torch.clamp(scale_factor, max=1e3)
        
        cancelation_signal = cancelation_signal * scale_factor
        
        return cancelation_signal.detach()
