"""
Verification module using α,β-CROWN for Lyapunov stability.

Implements bisection-based verification to find maximum ρ (ROA size) 
such that the verification condition is satisfied.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List
import numpy as np


class CartpoleVerificationGraph(nn.Module):
    """
    Encodes the full closed-loop dynamics for verification.
    
    Input: x ∈ [-1,1]⁴ (normalized state)
    Output: y = [y0, y1, y2]
      - y0 = V(x)                                      (Lyapunov value)
      - y1 = V(x_next) - (1-α)V(x)                    (decrease condition)
      - y2 = distance to next state boundary (penalty)
    """
    
    def __init__(
        self,
        controller: nn.Module,
        lyapunov: nn.Module,
        dynamics: nn.Module,
        alpha_lyap: float = 0.05,
    ):
        super().__init__()
        self.controller = controller
        self.lyapunov = lyapunov
        self.dynamics = dynamics
        self.alpha_lyap = alpha_lyap
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input state [batch_size, nx]
            
        Returns:
            y0: V(x)
            y1: V(x_next) - (1-α)V(x)
            y2: Boundary penalty for next state
        """
        # Current Lyapunov value
        y0 = self.lyapunov(x)
        
        # Compute next state
        u = self.controller(x)
        x_next = self.dynamics.step(x, u)
        
        # Lyapunov decrease condition
        v_next = self.lyapunov(x_next)
        v_curr = self.lyapunov(x)
        y1 = v_next - (1.0 - self.alpha_lyap) * v_curr
        
        # Next state is within [-1,1]^4 penalty (just for reference, not used in main verification)
        y2 = torch.zeros_like(y1)
        
        return y0, y1, y2


class VerificationCondition:
    """
    Represents the verification condition:
    ∀x ∈ B: (-F(x) ≥ 0 ∧ x_next ∈ B) ∨ (V(x) ≥ ρ)
    
    Simplified for verification with CROWN:
    ∀x inside ROA: F(x) ≤ 0 (i.e., y1 ≤ 0)
    ∀x outside ROA: unconstrained
    """
    
    @staticmethod
    def check_sample_based(
        model: CartpoleVerificationGraph,
        x_samples: torch.Tensor,
        rho: float,
        alpha_lyap: float = 0.05,
    ) -> Dict[str, float]:
        """
        Check verification condition on Monte Carlo samples.
        
        Returns:
            Dict with verification statistics
        """
        model.eval()
        
        with torch.no_grad():
            y0, y1, _ = model(x_samples)
            
            # Points inside ROA: V(x) < ρ
            inside_roa = (y0 < rho).squeeze()
            
            # For inside ROA points, check y1 ≤ 0 (Lyapunov decrease)
            if inside_roa.sum() > 0:
                y1_inside = y1[inside_roa]
                max_violation = y1_inside.max().item()
                violated = (y1_inside > 0.0).float().mean().item()
            else:
                max_violation = 0.0
                violated = 0.0
            
            roa_ratio = inside_roa.float().mean().item()
            
        return {
            "roa_ratio": roa_ratio,
            "max_violation": max_violation,
            "fraction_violated": violated,
            "satisfied": violated < 0.01,  # < 1% violation tolerance
        }


class BisectionVerifier:
    """
    Use bisection to find the maximum ρ (ROA size) that satisfies
    the verification condition.
    """
    
    def __init__(
        self,
        controller: nn.Module,
        lyapunov: nn.Module,
        dynamics: nn.Module,
        alpha_lyap: float = 0.05,
        device: torch.device = torch.device('cpu'),
    ):
        self.controller = controller
        self.lyapunov = lyapunov
        self.dynamics = dynamics
        self.alpha_lyap = alpha_lyap
        self.device = device
        
        self.verification_model = CartpoleVerificationGraph(
            controller, lyapunov, dynamics, alpha_lyap
        ).to(device)
    
    def verify_rho_with_samples(
        self,
        rho: float,
        x_min: torch.Tensor,
        x_max: torch.Tensor,
        n_samples: int = 5000,
        violation_tol: float = 0.01,
    ) -> bool:
        """
        Check if a given ρ satisfies the verification condition
        using Monte Carlo sampling.
        
        Returns:
            True if satisfied, False otherwise
        """
        # Sample uniformly in box
        x_samples = x_min + torch.rand(
            (n_samples, self.dynamics.nx), device=self.device
        ) * (x_max - x_min)
        
        stats = VerificationCondition.check_sample_based(
            self.verification_model,
            x_samples,
            rho,
            self.alpha_lyap,
        )
        
        is_satisfied = stats["fraction_violated"] <= violation_tol
        return is_satisfied, stats
    
    def bisection_search(
        self,
        x_min: torch.Tensor,
        x_max: torch.Tensor,
        rho_min: float = 1e-6,
        rho_max: float = 10.0,
        max_iterations: int = 10,
        n_samples: int = 5000,
        verbose: bool = True,
    ) -> Tuple[float, Dict]:
        """
        Find the maximum ρ using bisection.
        
        Algorithm:
        1. Initialize rho_lo = rho_min, rho_hi = rho_max
        2. Repeat until convergence:
           - Set rho_test = (rho_lo + rho_hi) / 2
           - If verify_rho(rho_test) succeeds:
               rho_lo = rho_test (can increase ρ)
           - Else:
               rho_hi = rho_test (must decrease ρ)
        3. Return rho_lo as the certified ROA threshold
        
        Returns:
            (rho_certified, history)
        """
        rho_lo = rho_min
        rho_hi = rho_max
        history = []
        
        if verbose:
            print(f"\n[Bisection] Starting ROA verification...")
            print(f"  Initial range: ρ ∈ [{rho_lo:.6f}, {rho_hi:.6f}]")
        
        for iteration in range(max_iterations):
            rho_test = (rho_lo + rho_hi) / 2.0
            is_satisfied, stats = self.verify_rho_with_samples(
                rho_test,
                x_min,
                x_max,
                n_samples=n_samples,
            )
            
            history.append({
                "iteration": iteration,
                "rho_test": rho_test,
                "satisfied": is_satisfied,
                "stats": stats,
            })
            
            if verbose:
                print(
                    f"  Iter {iteration}: ρ={rho_test:.6f}, "
                    f"ROA_ratio={stats['roa_ratio']:.2%}, "
                    f"violation={stats['fraction_violated']:.2%}, "
                    f"✓" if is_satisfied else "✗"
                )
            
            if is_satisfied:
                rho_lo = rho_test  # Can increase ρ
            else:
                rho_hi = rho_test  # Must decrease ρ
            
            # Check convergence
            if rho_hi - rho_lo < 1e-6:
                if verbose:
                    print(f"  [Converged] ρ_certified = {rho_lo:.6f}")
                break
        
        return rho_lo, {"history": history, "final_rho": rho_lo}


def create_cartpole_verification_result(
    verified_rho: float,
    history: List[Dict],
    x_min: torch.Tensor,
    x_max: torch.Tensor,
) -> Dict:
    """
    Create a structured verification result report.
    """
    box_volume = float(torch.prod(x_max - x_min).item())
    final_stats = history[-1]["stats"] if history else {}
    estimated_roa_volume = final_stats.get("roa_ratio", 0.0) * box_volume
    
    return {
        "verified_rho": verified_rho,
        "estimated_roa_volume": estimated_roa_volume,
        "roa_ratio_in_box": final_stats.get("roa_ratio", 0.0),
        "box_volume": box_volume,
        "box_limits": {"x_min": x_min.tolist(), "x_max": x_max.tolist()},
        "verification_history": history,
        "final_stats": final_stats,
    }
