import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ModuleType(Enum):
    LINEAR = "linear"
    LAYERNORM = "layernorm"
    GELU = "gelu"
    SOFTMAX = "softmax"
    MATMUL = "matmul"
    ATTENTION = "attention"
    OTHER = "other"


@dataclass
class RoundingErrorStats:
    mean: float = 0.0
    variance: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 3.0
    max_abs: float = 0.0


@dataclass
class ErrorFeatureMatrix:
    name: str
    module_type: ModuleType
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    
    error_mean: torch.Tensor = None
    error_covariance: torch.Tensor = None
    error_spectrum: torch.Tensor = None
    
    fourier_modes: torch.Tensor = None
    amplification_factors: torch.Tensor = None
    
    cancelation_signal: torch.Tensor = None
    
    update_count: int = 0
    running_stats: RoundingErrorStats = field(default_factory=RoundingErrorStats)
    
    def initialize_tensors(self, device: torch.device):
        out_dim = self.output_shape[-1]
        self.error_mean = torch.zeros(out_dim, device=device)
        self.error_covariance = torch.eye(out_dim, device=device) * 1e-8
        self.fourier_modes = torch.fft.fftfreq(out_dim, device=device)
        self.amplification_factors = torch.ones(out_dim, device=device)
        self.cancelation_signal = torch.zeros(self.output_shape, device=device)
    
    def update_error_stats(self, error_tensor: torch.Tensor):
        flat_error = error_tensor.detach().view(-1, error_tensor.size(-1))
        batch_mean = flat_error.mean(dim=0)
        batch_cov = flat_error.T @ flat_error / max(flat_error.size(0), 1)
        
        alpha = 1.0 / (self.update_count + 1)
        self.error_mean = (1 - alpha) * self.error_mean + alpha * batch_mean
        
        diff = batch_mean - self.error_mean
        self.error_covariance = (1 - alpha) * self.error_covariance + alpha * (batch_cov + diff.outer(diff))
        
        abs_err = error_tensor.abs().mean().item()
        self.running_stats.max_abs = max(self.running_stats.max_abs, abs_err)
        self.update_count += 1


class ModuleErrorTracker:
    def __init__(self):
        self.trackers: Dict[str, ErrorFeatureMatrix] = {}
        self.global_device: torch.device = torch.device("cpu")
    
    def register_module(self, name: str, module_type: ModuleType,
                        input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]):
        if name not in self.trackers:
            efm = ErrorFeatureMatrix(
                name=name,
                module_type=module_type,
                input_shape=input_shape,
                output_shape=output_shape,
            )
            efm.initialize_tensors(self.global_device)
            self.trackers[name] = efm
        return self.trackers[name]
    
    def get_tracker(self, name: str) -> Optional[ErrorFeatureMatrix]:
        return self.trackers.get(name)
    
    def set_device(self, device: torch.device):
        self.global_device = device
        for efm in self.trackers.values():
            efm.initialize_tensors(device)
    
    def get_all_trackers(self) -> List[Tuple[str, ErrorFeatureMatrix]]:
        return list(self.trackers.items())
