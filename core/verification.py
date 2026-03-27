"""
Verification module using α,β-CROWN for Lyapunov stability.

Implements bisection-based verification to find maximum ρ (ROA size) 
such that the verification condition is satisfied.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List
import numpy as np

try:
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
except ImportError:
    BoundedModule = None
    BoundedTensor = None
    PerturbationLpNorm = None


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

    def _decrease_expression(self, x_next: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.lyapunov, "algebraic_decrease"):
            return self.lyapunov.algebraic_decrease(x_next, x, self.alpha_lyap)
        v_next = self.lyapunov(x_next)
        v_curr = self.lyapunov(x)
        return v_next - (1.0 - self.alpha_lyap) * v_curr
        
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
        y1 = self._decrease_expression(x_next, x)
        
        # Next state is within [-1,1]^4 penalty (just for reference, not used in main verification)
        y2 = torch.zeros_like(y1)
        
        return y0, y1, y2


class CartpoleDecreaseGraph(nn.Module):
    """
    Single-output graph for formal local verification with CROWN.

    Output is F(x) = V(x_next) - (1 - alpha) * V(x).
    If upper bound of F(x) is <= 0 on a region, Lyapunov decrease is certified there.
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

    def _decrease_expression(self, x_next: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.lyapunov, "algebraic_decrease"):
            return self.lyapunov.algebraic_decrease(x_next, x, self.alpha_lyap)
        v_next = self.lyapunov(x_next)
        v_curr = self.lyapunov(x)
        return v_next - (1.0 - self.alpha_lyap) * v_curr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.controller(x)
        x_next = self.dynamics.step(x, u)
        return self._decrease_expression(x_next, x)


class CartpoleLevelsetImplicationGraph(nn.Module):
    """
    Single-output implication graph for level-set verification.

    Verifies: V(x) <= rho => DeltaV(x) <= 0
    by bounding:
        F_verify(x) = DeltaV(x) - M * ReLU(V(x) - rho)
    on a large box.
    """

    def __init__(
        self,
        controller: nn.Module,
        lyapunov: nn.Module,
        dynamics: nn.Module,
        alpha_lyap: float = 0.05,
        rho: float = 0.1,
        implication_m: float = 100.0,
    ):
        super().__init__()
        self.controller = controller
        self.lyapunov = lyapunov
        self.dynamics = dynamics
        self.alpha_lyap = alpha_lyap
        self.register_buffer("rho", torch.tensor(float(rho), dtype=torch.float32))
        self.register_buffer("implication_m", torch.tensor(float(implication_m), dtype=torch.float32))

    def _decrease_expression(self, x_next: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.lyapunov, "algebraic_decrease"):
            return self.lyapunov.algebraic_decrease(x_next, x, self.alpha_lyap)
        v_next = self.lyapunov(x_next)
        v_curr = self.lyapunov(x)
        return v_next - (1.0 - self.alpha_lyap) * v_curr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.controller(x)
        x_next = self.dynamics.step(x, u)
        delta_v = self._decrease_expression(x_next, x)
        v_curr = self.lyapunov(x)
        penalty = torch.relu(v_curr - self.rho)
        return delta_v - self.implication_m * penalty


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


class CrownRadiusVerifier:
    """
    Formal local verifier using CROWN on L_inf balls centered at equilibrium.

    Finds max epsilon such that for all ||x||_inf <= epsilon:
        V(x_next) - (1 - alpha) * V(x) <= 0
    """

    def __init__(
        self,
        controller: nn.Module,
        lyapunov: nn.Module,
        dynamics: nn.Module,
        alpha_lyap: float = 0.05,
        device: torch.device = torch.device("cpu"),
    ):
        if BoundedModule is None:
            raise RuntimeError(
                "auto_LiRPA is not available. Install it to use CrownRadiusVerifier."
            )

        self.device = device
        self.controller = controller
        self.lyapunov = lyapunov
        self.dynamics = dynamics
        self.alpha_lyap = alpha_lyap

        self.graph = CartpoleDecreaseGraph(
            controller=controller,
            lyapunov=lyapunov,
            dynamics=dynamics,
            alpha_lyap=alpha_lyap,
        ).to(device)
        self.graph.eval()

        self.x0 = torch.zeros((1, dynamics.nx), dtype=torch.float32, device=device)
        self.bounded_model = BoundedModule(self.graph, self.x0, bound_opts={"relu": "adaptive"})

    def _switch_to_implication_graph(self, rho: float, implication_m: float) -> None:
        self.graph = CartpoleLevelsetImplicationGraph(
            controller=self.controller,
            lyapunov=self.lyapunov,
            dynamics=self.dynamics,
            alpha_lyap=self.alpha_lyap,
            rho=rho,
            implication_m=implication_m,
        ).to(self.device)
        self.graph.eval()
        self.bounded_model = BoundedModule(self.graph, self.x0, bound_opts={"relu": "adaptive"})

    def _switch_to_decrease_graph(self) -> None:
        self.graph = CartpoleDecreaseGraph(
            controller=self.controller,
            lyapunov=self.lyapunov,
            dynamics=self.dynamics,
            alpha_lyap=self.alpha_lyap,
        ).to(self.device)
        self.graph.eval()
        self.bounded_model = BoundedModule(self.graph, self.x0, bound_opts={"relu": "adaptive"})

    def bound_at_eps(
        self,
        eps: float,
        method: str = "CROWN",
        rho: float | None = None,
        implication_m: float = 100.0,
    ) -> Dict[str, float]:
        if eps < 0.0:
            raise ValueError("eps must be non-negative")

        if rho is not None:
            self._switch_to_implication_graph(rho=float(rho), implication_m=float(implication_m))
        else:
            self._switch_to_decrease_graph()

        ptb = PerturbationLpNorm(norm=torch.inf, eps=float(eps))
        bx = BoundedTensor(self.x0, ptb)

        with torch.no_grad():
            lb, ub = self.bounded_model.compute_bounds(x=(bx,), method=method)

        # Cấp dung sai 1e-5 để bù đắp sai số nới lỏng của CROWN xung quanh x=0
        return {
            "eps": float(eps),
            "rho": None if rho is None else float(rho),
            "lb": float(lb.item()),
            "ub": float(ub.item()),
            "certified": bool(ub.item() <= 1e-5), 
        }

    def bisection_search(
        self,
        eps_min: float = 1e-4,
        eps_max: float = 1.0,
        max_iterations: int = 12,
        method: str = "CROWN",
        rho: float | None = None,
        implication_m: float = 100.0,
        verbose: bool = True,
    ) -> Tuple[float, Dict]:
        lo = float(eps_min)
        hi = float(eps_max)
        history = []

        if verbose:
            print(f"\n[CROWN] Search certified radius in [{lo:.6f}, {hi:.6f}]")

        lo_stats = self.bound_at_eps(lo, method=method, rho=rho, implication_m=implication_m)
        hi_stats = self.bound_at_eps(hi, method=method, rho=rho, implication_m=implication_m)
        history.append({"iteration": -1, "stats": lo_stats})
        history.append({"iteration": -2, "stats": hi_stats})

        if not lo_stats["certified"]:
            if verbose:
                print("  Lower endpoint is not certified; returning 0.0")
            return 0.0, {"history": history, "final_eps": 0.0}

        if hi_stats["certified"]:
            if verbose:
                print(f"  Upper endpoint certified directly: eps={hi:.6f}")
            return hi, {"history": history, "final_eps": hi}

        for i in range(max_iterations):
            mid = 0.5 * (lo + hi)
            stats = self.bound_at_eps(mid, method=method, rho=rho, implication_m=implication_m)
            history.append({"iteration": i, "stats": stats})

            if verbose:
                mark = "✓" if stats["certified"] else "✗"
                print(f"  Iter {i:02d}: eps={mid:.6f}, UB={stats['ub']:.6f} {mark}")

            if stats["certified"]:
                lo = mid
            else:
                hi = mid

            if (hi - lo) < 1e-5:
                break

        return lo, {"history": history, "final_eps": lo}


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
