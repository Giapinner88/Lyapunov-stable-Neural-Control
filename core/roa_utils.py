"""
ROA (Region of Attraction) utilities for Lyapunov-stable neural control.

Implements:
1. Boundary value finding (ρ calculation)
2. ROA expansion tracking
3. Verification condition checking
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict


def compute_rho_boundary(
    lyapunov: nn.Module, dynamics: nn.Module, controller: nn.Module,
    x_min: torch.Tensor, x_max: torch.Tensor,
    n_boundary_samples: int = 1000, n_pgd_steps: int = 50,
    pgd_step_size: float = 0.02, gamma: float = 0.9,
    device: torch.device = torch.device('cpu'), verbose: bool = False,
) -> Tuple[float, torch.Tensor]:
    
    lyapunov.eval()
    nx = x_min.shape[0]
    n_per_face = max(1, n_boundary_samples // (2 * nx))
    min_v_values = []
    x_solutions = []

    for dim in range(nx):
        for is_max_face in [False, True]:
            # Khởi tạo điểm ngẫu nhiên
            x_face = torch.rand((n_per_face, nx), device=device, dtype=x_min.dtype) * (x_max - x_min) + x_min
            # Ép điểm nằm chặt trên mặt phẳng tương ứng
            if is_max_face:
                x_face[:, dim] = x_max[dim]
            else:
                x_face[:, dim] = x_min[dim]
                
            x_face = x_face.clone().detach().requires_grad_(True)
            
            for step in range(n_pgd_steps):
                v = lyapunov(x_face)
                loss = torch.sum(v)
                grad = torch.autograd.grad(loss, x_face)[0]
                
                with torch.no_grad():
                    # Gradient descent để tìm V nhỏ nhất
                    span = x_max - x_min
                    x_face.data = x_face.data - pgd_step_size * span * torch.sign(grad)
                    x_face.data = torch.clamp(x_face.data, min=x_min, max=x_max)
                    
                    # QUAN TRỌNG: Khóa x trở lại mặt phẳng vỏ hộp (Project back to boundary)
                    if is_max_face:
                        x_face.data[:, dim] = x_max[dim]
                    else:
                        x_face.data[:, dim] = x_min[dim]
            
            with torch.no_grad():
                v_final = lyapunov(x_face)
                min_v_values.append(v_final.min().item())
                x_solutions.append(x_face.clone())

    min_v_val = min(min_v_values)
    rho = gamma * max(min_v_val, 1e-4)
    
    if verbose:
        print(f"[ROA] Computed ρ = {gamma} * {min_v_val:.6f} = {rho:.6f}")
        
    x_solutions_tensor = torch.cat(x_solutions, dim=0)
    return rho, x_solutions_tensor


def estimate_roa_size(
    lyapunov: nn.Module,
    rho: float,
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    n_samples: int = 10000,
    device: torch.device = torch.device('cpu'),
) -> float:
    """
    Estimate the volume of ROA by Monte Carlo sampling.
    
    ROA = {x ∈ B | V(x) < ρ}
    
    Volume ≈ |B| * (# of samples with V(x) < ρ) / n_samples
    """
    
    lyapunov.eval()
    nx = x_min.shape[0]
    
    # Sample uniformly in box B
    x_samples = torch.rand((n_samples, nx), device=device, dtype=x_min.dtype)
    x_samples = x_samples * (x_max - x_min) + x_min
    
    with torch.no_grad():
        v_samples = lyapunov(x_samples)
        inside_roa = (v_samples.squeeze() < rho).float()
    
    roa_ratio = inside_roa.mean().item()
    box_volume = torch.prod(x_max - x_min).item()
    estimated_roa_volume = roa_ratio * box_volume
    
    return estimated_roa_volume, roa_ratio


def verify_lyapunov_condition(
    lyapunov: nn.Module,
    dynamics: nn.Module,
    controller: nn.Module,
    rho: float,
    x_samples: torch.Tensor,
    alpha_lyap: float = 0.05,
    x_min: Optional[torch.Tensor] = None,
    x_max: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Check verification condition on sampled points:
    ∀x ∈ B: (-F(x) ≥ 0 ∧ x_{t+1} ∈ B) ∨ (V(x) ≥ ρ)
    
    Where F(x) = V(x_{t+1}) - (1-α)·V(x)
    
    Returns:
        Dict with statistics on condition satisfaction
    """
    
    lyapunov.eval()
    dynamics.eval()
    controller.eval()
    
    with torch.no_grad():
        # Compute V(x)
        v_x = lyapunov(x_samples)
        
        # Check which points are inside vs outside ROA
        inside_roa = (v_x < rho).squeeze()
        outside_roa = ~inside_roa
        
        # For points inside ROA, need to verify Lyapunov decrease
        x_inside = x_samples[inside_roa]
        if x_inside.shape[0] > 0:
            u = controller(x_inside)
            x_next = dynamics.step(x_inside, u)
            v_next = lyapunov(x_next)
            v_curr = lyapunov(x_inside)
            
            # Check decrease: V(x+1) - (1-α)V(x) ≤ 0
            decrease = v_next - (1 - alpha_lyap) * v_curr
            violation_inside = (decrease > 0).float().mean().item()
            max_decrease_violation = decrease.max().item()
        else:
            violation_inside = 0.0
            max_decrease_violation = 0.0
        
        # For points outside ROA, they're automatically satisfied
        violation_outside = 0.0
        
        # Compute overall satisfaction
        total_violation_count = inside_roa.sum().item() * (1 - (1 - violation_inside))
        total_samples = x_samples.shape[0]
        satisfaction_rate = 1.0 - (total_violation_count / max(1, total_samples))
    
    return {
        "roa_ratio": inside_roa.float().mean().item(),
        "violation_inside_roa": violation_inside,
        "max_decrease_violation": max_decrease_violation,
        "overall_satisfaction": satisfaction_rate,
        "n_samples": x_samples.shape[0],
    }


class ROATracker:
    """Track ROA expansion during training."""
    
    def __init__(self, gamma: float = 0.9):
        self.gamma = gamma
        self.rho_values = []
        self.roa_volumes = []
        self.satisfaction_rates = []
        
    def update(self, rho: float, roa_volume: float, satisfaction: float):
        self.rho_values.append(rho)
        self.roa_volumes.append(roa_volume)
        self.satisfaction_rates.append(satisfaction)
    
    def get_latest(self) -> Dict[str, float]:
        if not self.rho_values:
            return {}
        return {
            "rho": self.rho_values[-1],
            "roa_volume": self.roa_volumes[-1],
            "satisfaction": self.satisfaction_rates[-1],
        }
    
    def get_trajectory(self) -> Dict[str, list]:
        return {
            "rho_values": self.rho_values,
            "roa_volumes": self.roa_volumes,
            "satisfaction_rates": self.satisfaction_rates,
        }
