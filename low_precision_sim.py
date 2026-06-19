import torch
from typing import Optional
from enum import Enum


class PrecisionMode(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"


def get_fp8_ranges(mode: str = "e4m3"):
    if mode == "e4m3":
        max_val = 448.0
        min_val = -max_val
        eps = 2 ** -7
    else:
        max_val = 57344.0
        min_val = -max_val
        eps = 2 ** -14
    return min_val, max_val, eps


def quantize_to_fp8(x: torch.Tensor, mode: str = "e4m3") -> torch.Tensor:
    min_val, max_val, eps = get_fp8_ranges(mode)
    
    x_clamped = torch.clamp(x, min_val, max_val)
    
    sign = torch.sign(x_clamped)
    abs_x = torch.abs(x_clamped)
    
    exponent = torch.floor(torch.log2(abs_x + eps))
    exponent = torch.clamp(exponent, min=-4 if mode == "e4m3" else -14,
                           max=7 if mode == "e4m3" else 15)
    
    mantissa_bits = 3 if mode == "e4m3" else 2
    mantissa_scale = 2 ** mantissa_bits
    
    mantissa = abs_x / (2 ** exponent) - 1.0
    mantissa = torch.round(mantissa * mantissa_scale) / mantissa_scale
    mantissa = torch.clamp(mantissa, 0.0, 1.0 - 1.0 / mantissa_scale)
    
    quantized = sign * (1.0 + mantissa) * (2 ** exponent)
    
    quantized = torch.where(abs_x < eps, torch.zeros_like(x_clamped), quantized)
    quantized = torch.clamp(quantized, min_val, max_val)
    
    return quantized


class LowPrecisionSimulator:
    def __init__(self, precision: str = "fp16"):
        self.original_precision = precision.lower()
        p = self.original_precision
        if p == "fp8":
            p = "fp8_e4m3"
        self.precision = p
        self.mode = PrecisionMode(self.precision)
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == PrecisionMode.FP32:
            return x
        elif self.mode == PrecisionMode.FP16:
            return x.half().float()
        elif self.mode == PrecisionMode.BF16:
            return x.bfloat16().float()
        elif self.mode == PrecisionMode.FP8_E4M3:
            return quantize_to_fp8(x, "e4m3")
        elif self.mode == PrecisionMode.FP8_E5M2:
            return quantize_to_fp8(x, "e5m2")
        else:
            return x
    
    def compute_rounding_error(self, x_high: torch.Tensor) -> torch.Tensor:
        x_low = self.quantize(x_high)
        return x_low - x_high
    
    def get_dtype(self):
        if self.mode == PrecisionMode.FP32:
            return torch.float32
        elif self.mode == PrecisionMode.FP16:
            return torch.float16
        elif self.mode == PrecisionMode.BF16:
            return torch.bfloat16
        else:
            return torch.float32
    
    def is_native_low_precision(self) -> bool:
        return self.mode in [PrecisionMode.FP16, PrecisionMode.BF16]
    
    def precision_label(self) -> str:
        if self.original_precision == "fp8":
            return "FP8 (8位浮点，E4M3模拟量化)"
        labels = {
            PrecisionMode.FP32: "FP32 (单精度)",
            PrecisionMode.FP16: "FP16 (半精度)",
            PrecisionMode.BF16: "BF16 (脑浮点)",
            PrecisionMode.FP8_E4M3: "FP8-E4M3 (模拟量化)",
            PrecisionMode.FP8_E5M2: "FP8-E5M2 (模拟量化)",
        }
        return labels.get(self.mode, "Unknown")
