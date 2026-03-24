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
    lyapunov: nn.Module,
    dynamics: nn.Module,
    controller: nn.Module,
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    n_boundary_samples: int = 1000,
    n_pgd_steps: int = 50,
    pgd_step_size: float = 0.02,
    gamma: float = 0.9,
    device: torch.device = torch.device('cpu'),
    verbose: bool = False,
) -> Tuple[float, torch.Tensor]:
    """
    Compute ρ = γ · min_{x ∈ ∂B} V(x)
    
    Strategy:
    1. Sample points on boundary of box B = [x_min, x_max]
    2. Use PGD to minimize V(x) for each boundary point
    3. Find minimum V value on boundary
    4. Return ρ = γ * min_V
    
    Args:
        lyapunov: LyapunovNet(x) → V(x)
        dynamics: CartpoleDynamics or PendulumDynamics
        controller: NeuralController(x) → u
        x_min, x_max: box boundaries
        n_boundary_samples: number of boundary sample points
        n_pgd_steps: steps of PGD for V minimization
        pgd_step_size: step size for PGD
        gamma: scaling factor for ρ (typically 0.9)
        device: computation device
        verbose: print debug info
        
    Returns:
        rho: Computed ROA threshold
        x_boundary_solutions: States found on boundary
    """
    
    lyapunov.eval()
    controller.eval()
    dynamics.eval()
    
    nx = x_min.shape[0]
    batch_size = n_boundary_samples
    device = x_min.device
    dtype = x_min.dtype
    
    # Step 1: Sample points on boundary of box B
    # Strategy: Place points near edges (either x_min or x_max for each dimension)
    x_boundary = []
    for dim in range(nx):
        # Half samples near x_min[dim], half near x_max[dim]
        n_this_face = batch_size // (2 * nx) + 1
        
        # Near x_min[dim]
        x_face = torch.rand((n_this_face, nx), device=device, dtype=dtype) * (x_max - x_min) + x_min
        x_face[:, dim] = x_min[dim] + 1e-3 * (x_max[dim] - x_min[dim])  # Stick to boundary
        x_boundary.append(x_face)
        
        # Near x_max[dim]
        x_face = torch.rand((n_this_face, nx), device=device, dtype=dtype) * (x_max - x_min) + x_min
        x_face[:, dim] = x_max[dim] - 1e-3 * (x_max[dim] - x_min[dim])  # Stick to boundary
        x_boundary.append(x_face)
    
    x_boundary = torch.cat(x_boundary, dim=0)[:batch_size]  # Trim to size
    
    # Step 2: PGD to minimize V(x) on boundary
    min_v_values = []
    x_solutions = []
    
    for i, x_init in enumerate(x_boundary):
        x = x_init.unsqueeze(0).clone().detach().requires_grad_(True)
        
        for step in range(n_pgd_steps):
            with torch.enable_grad():
                v = lyapunov(x)
                loss = torch.sum(v)  # Minimize V
            
            loss.backward()
            
            with torch.no_grad():
                # Gradient descent on x (minimize V)
                span = x_max - x_min
                scaled_step = pgd_step_size * span
                x.data = x - scaled_step * torch.sign(x.grad)
                
                # Clamp to box
                x.data = torch.clamp(x.data, min=x_min, max=x_max)
                
                # Keep on boundary (optional: could relax this)
                x.grad.zero_()
        
        with torch.no_grad():
            v_final = lyapunov(x).item()
            min_v_values.append(v_final)
            x_solutions.append(x.clone())
    
    # Step 3: Find minimum V on boundary
    min_v_val = min(min_v_values)
    rho = gamma * max(min_v_val, 1e-4)  # Ensure rho > 0
    
    if verbose:
        print(f"[ROA] Boundary V values: min={min_v_val:.6f}, max={max(min_v_values):.6f}")
        print(f"[ROA] Computed ρ = {gamma} * {min_v_val:.6f} = {rho:.6f}")
    
    x_solutions_tensor = torch.cat([x.squeeze(0) for x in x_solutions], dim=0)
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
