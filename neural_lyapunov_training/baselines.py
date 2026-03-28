import torch
import torch.nn as nn


class LQRController(nn.Module):
    """Linear LQR controller u(x) = -Kx with optional actuator saturation."""

    def __init__(self, K: torch.Tensor, u_bound: float = 6.0):
        super().__init__()
        if K.ndim != 2:
            raise ValueError("K must be a 2D tensor with shape [nu, nx].")
        self.register_buffer("K", K.detach().clone())
        self.u_bound = float(u_bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = -torch.matmul(x, self.K.T)
        return torch.clamp(u, min=-self.u_bound, max=self.u_bound)


class QuadraticLyapunov(nn.Module):
    """Quadratic Lyapunov candidate V(x) = x^T P x."""

    def __init__(self, P: torch.Tensor, eps_spd: float = 1e-6):
        super().__init__()
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError("P must be a square matrix tensor [nx, nx].")
        if eps_spd <= 0.0:
            raise ValueError("eps_spd must be > 0.")

        P_sym = 0.5 * (P + P.T)
        eye = torch.eye(P_sym.shape[0], dtype=P_sym.dtype, device=P_sym.device)
        P_spd = P_sym + eps_spd * eye
        self.register_buffer("P", P_spd.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Px = torch.matmul(x, self.P)
        return torch.sum(Px * x, dim=1, keepdim=True)