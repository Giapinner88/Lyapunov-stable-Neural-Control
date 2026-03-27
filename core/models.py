import torch
import torch.nn as nn

# ==========================================
# 1. NEURAL CONTROLLER
# ==========================================
class NeuralController(nn.Module):
    def __init__(
        self,
        nx: int,
        nu: int,
        hidden_sizes: list = [64, 64],
        u_bound: float = 6.0,
        state_limits: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    ):
        super().__init__()
        self.u_bound = u_bound
        self.nx = nx
        
        if state_limits is None:
            state_limits = torch.ones(nx)
        state_limits_tensor = torch.as_tensor(state_limits, dtype=torch.float32)
        if state_limits_tensor.numel() != nx:
            raise ValueError(f"state_limits must have {nx} elements, got {state_limits_tensor.numel()}")
        self.register_buffer("state_limits", state_limits_tensor.view(nx))
        self.register_buffer("origin", torch.zeros(nx), persistent=False)
        
        layers = []
        in_size = nx
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h, bias=False))
            layers.append(nn.Tanh())  
            in_size = h
            
        layers.append(nn.Linear(in_size, nu, bias=False))
        layers.append(nn.Tanh()) 
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x / self.state_limits

        zero_norm = self.origin.unsqueeze(0) / self.state_limits
        u_raw = self.net(x_norm)
        u_origin = self.net(zero_norm).squeeze(0)
        unclipped_output = (u_raw - u_origin) * self.u_bound

        # THAY THẾ CHO: torch.clamp(unclipped_output, -self.u_bound, self.u_bound)
        u_bound_tensor = torch.tensor(self.u_bound, device=x.device)
        f1 = torch.nn.functional.relu(unclipped_output - (-u_bound_tensor)) + (-u_bound_tensor)
        u_out = -(torch.nn.functional.relu(u_bound_tensor - f1) - u_bound_tensor)
        return u_out

    def load_state_dict(self, state_dict, strict: bool = True):
        # Backward compatibility: old checkpoints may contain bias tensors.
        filtered = {k: v for k, v in state_dict.items() if not k.endswith(".bias")}
        return super().load_state_dict(filtered, strict=strict)


# ==========================================
# 2. NEURAL LYAPUNOV FUNCTION
# ==========================================
class NeuralLyapunov(nn.Module):
    def __init__(
        self,
        nx: int,
        hidden_sizes: list = [64, 64],
        eps: float = 0.01,
        state_limits: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    ):
        super().__init__()
        if eps <= 0.0:
            raise ValueError("eps phải > 0 để đảm bảo V(x) dương xác định nghiêm ngặt")
        
        if state_limits is None:
            state_limits = torch.ones(nx)
        state_limits_tensor = torch.as_tensor(state_limits, dtype=torch.float32)
        if state_limits_tensor.numel() != nx:
            raise ValueError(f"state_limits must have {nx} elements, got {state_limits_tensor.numel()}")
        self.register_buffer("state_limits", state_limits_tensor.view(nx))
        
        layers = []
        in_size = nx
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h, bias=False))
            layers.append(nn.Tanh())
            in_size = h
        
        layers.append(nn.Linear(in_size, nx, bias=False)) 
        self.phi_V = nn.Sequential(*layers)
        
        # FIX: Khởi tạo R lớn hơn (từ 0.1 → 0.5) để P = R^T*R kích thước hợp lý
        self.R = nn.Parameter(torch.randn(nx, nx) * 0.1)
        self.eps = eps
        
        self.register_buffer("eye", torch.eye(nx))
        self.register_buffer("origin", torch.zeros(nx))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero_norm = self.origin.unsqueeze(0) / self.state_limits
        with torch.no_grad():
            phi_0 = self.phi_V(zero_norm).squeeze(0)

        x_norm = x / self.state_limits
        phi_x = self.phi_V(x_norm)
        
        # Use relu(x) + relu(-x) instead of torch.abs(x) since the verification code does relu splitting.
        diff = phi_x - phi_0
        term1 = torch.sum(torch.nn.functional.relu(diff) + torch.nn.functional.relu(-diff), dim=1, keepdim=True)
        
        P = self.eps * self.eye + torch.matmul(self.R.T, self.R)
        Px = torch.matmul(x, P)
        term2 = torch.sum(Px * x, dim=1, keepdim=True)
        
        V = term1 + term2
        return V

    def quadratic_matrix(self) -> torch.Tensor:
        return self.eps * self.eye + torch.matmul(self.R.T, self.R)

    def algebraic_decrease(self, x_next: torch.Tensor, x: torch.Tensor, alpha_lyap: float) -> torch.Tensor:
        # Cross-term-free form for tighter CROWN bounds:
        # x_next^T P x_next - (1-alpha) x^T P x
        scale = 1.0 - alpha_lyap
        P = self.quadratic_matrix()
        q_next = torch.sum(torch.matmul(x_next, P) * x_next, dim=1, keepdim=True)
        q_curr = torch.sum(torch.matmul(x, P) * x, dim=1, keepdim=True)
        return q_next - scale * q_curr

    def load_state_dict(self, state_dict, strict: bool = True):
        # Backward compatibility: old checkpoints stored origin as shape [1, nx].
        # Also drop bias tensors from legacy checkpoints.
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith(".bias")}
        origin_key = "origin"
        if origin_key in state_dict:
            origin_tensor = state_dict[origin_key]
            if origin_tensor.ndim == 2 and origin_tensor.shape[0] == 1:
                state_dict = state_dict.copy()
                state_dict[origin_key] = origin_tensor.squeeze(0)
        return super().load_state_dict(state_dict, strict=strict)