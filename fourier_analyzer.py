import torch
from typing import Dict, Tuple, Optional
from error_features import ErrorFeatureMatrix, ModuleType


class FourierModeAnalyzer:
    def __init__(self, n_modes: int = 64, epsilon: float = 1e-10):
        self.n_modes = n_modes
        self.epsilon = epsilon
        self._module_gain_profiles = {
            ModuleType.LINEAR: self._linear_gain_profile,
            ModuleType.LAYERNORM: self._layernorm_gain_profile,
            ModuleType.GELU: self._gelu_gain_profile,
            ModuleType.SOFTMAX: self._softmax_gain_profile,
            ModuleType.MATMUL: self._matmul_gain_profile,
            ModuleType.ATTENTION: self._attention_gain_profile,
            ModuleType.OTHER: self._default_gain_profile,
        }
    
    def analyze_error_spectrum(self, error_tensor: torch.Tensor,
                               efm: ErrorFeatureMatrix) -> Tuple[torch.Tensor, torch.Tensor]:
        if error_tensor.dim() > 2:
            flat = error_tensor.view(-1, error_tensor.size(-1))
        else:
            flat = error_tensor
        
        mean_sub = flat - flat.mean(dim=0, keepdim=True)
        spectrum = torch.fft.rfft(mean_sub, dim=-1)
        power_spectrum = spectrum.abs().pow(2).mean(dim=0)
        
        modes = torch.fft.rfftfreq(error_tensor.size(-1), device=error_tensor.device)
        efm.error_spectrum = power_spectrum.detach()
        
        return power_spectrum, modes
    
    def _linear_gain_profile(self, modes: torch.Tensor,
                             input_scale: float = 1.0) -> torch.Tensor:
        base_gain = torch.ones_like(modes)
        freq_weight = 1.0 + 0.3 * torch.abs(modes)
        return base_gain * freq_weight * input_scale
    
    def _layernorm_gain_profile(self, modes: torch.Tensor,
                                input_scale: float = 1.0) -> torch.Tensor:
        low_freq_boost = 1.0 + 2.0 * torch.exp(-torch.abs(modes) * 5.0)
        high_freq_atten = 1.0 / (1.0 + 0.5 * torch.abs(modes))
        return low_freq_boost * high_freq_atten * input_scale
    
    def _gelu_gain_profile(self, modes: torch.Tensor,
                           input_scale: float = 1.0) -> torch.Tensor:
        nonlinear_boost = 1.0 + 1.5 * torch.abs(torch.sin(3.14159 * modes))
        return nonlinear_boost * input_scale
    
    def _softmax_gain_profile(self, modes: torch.Tensor,
                              input_scale: float = 1.0) -> torch.Tensor:
        freq_exponential = torch.exp(2.0 * torch.abs(modes))
        return freq_exponential * input_scale
    
    def _matmul_gain_profile(self, modes: torch.Tensor,
                             input_scale: float = 1.0) -> torch.Tensor:
        dim_factor = torch.sqrt(torch.tensor(modes.size(-1), dtype=torch.float32, device=modes.device))
        return (1.0 + 0.8 * torch.abs(modes)) * dim_factor * input_scale
    
    def _attention_gain_profile(self, modes: torch.Tensor,
                                input_scale: float = 1.0) -> torch.Tensor:
        low_freq = 1.0 + 3.0 * torch.exp(-torch.abs(modes) * 3.0)
        high_freq = torch.exp(1.5 * torch.abs(modes))
        return (low_freq + 0.5 * high_freq) * input_scale
    
    def _default_gain_profile(self, modes: torch.Tensor,
                              input_scale: float = 1.0) -> torch.Tensor:
        return torch.ones_like(modes) * input_scale
    
    def predict_amplification(self, efm: ErrorFeatureMatrix,
                              next_module_type: ModuleType,
                              input_scale: float = 1.0) -> torch.Tensor:
        if efm.error_spectrum is None:
            return torch.ones(self.n_modes, device=efm.error_mean.device)
        
        original_dim = efm.error_mean.size(-1)
        modes = torch.fft.rfftfreq(original_dim,
                                   device=efm.error_spectrum.device)
        
        gain_fn = self._module_gain_profiles.get(next_module_type, self._default_gain_profile)
        gain_profile = gain_fn(modes, input_scale)
        
        spectral_density = efm.error_spectrum / (efm.error_spectrum.sum() + self.epsilon)
        weighted_amplification = gain_profile * (1.0 + spectral_density)
        
        efm.amplification_factors = weighted_amplification.detach()
        return weighted_amplification
    
    def compute_error_propagation_matrix(self, efm: ErrorFeatureMatrix,
                                         next_module_type: ModuleType) -> torch.Tensor:
        dim = efm.error_covariance.size(-1)
        device = efm.error_covariance.device
        
        modes = torch.fft.fftfreq(dim, device=device)
        gain_fn = self._module_gain_profiles.get(next_module_type, self._default_gain_profile)
        freq_gains = gain_fn(modes)
        
        prop_matrix = torch.diag_embed(freq_gains)
        cov_sqrt = torch.linalg.cholesky(efm.error_covariance + torch.eye(dim, device=device) * self.epsilon)
        prop_matrix = cov_sqrt @ prop_matrix @ cov_sqrt.T.conj()
        
        return prop_matrix
    
    def identify_dominant_modes(self, efm: ErrorFeatureMatrix,
                                top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        if efm.error_spectrum is None:
            return torch.tensor([], device=efm.error_mean.device), torch.tensor([], device=efm.error_mean.device)
        
        spectrum = efm.error_spectrum
        values, indices = torch.topk(spectrum, k=min(top_k, spectrum.size(-1)))
        original_dim = efm.error_mean.size(-1)
        modes = torch.fft.rfftfreq(original_dim, device=spectrum.device)
        
        return modes[indices], values
