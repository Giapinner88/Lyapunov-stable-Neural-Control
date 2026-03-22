import torch
from core.dynamics import PendulumDynamics

# 1. Khởi tạo hệ thống con lắc
pendulum = PendulumDynamics()

# 2. Tạo một batch gồm 5 trạng thái ngẫu nhiên. BẮT BUỘC bật theo dõi đạo hàm!
# x có shape (5, 2) tương ứng với [theta, theta_dot]
x = torch.randn((5, 2), requires_grad=True)

# u có shape (5, 1) tương ứng với lực momen xoắn
u = torch.zeros((5, 1), requires_grad=True)

# 3. Tính trạng thái bước tiếp theo qua tích phân RK4
x_next = pendulum.step(x, u)

# 4. Giả lập một hàm mục tiêu: V_next = tổng bình phương các phần tử của x_next
V_next = torch.sum(x_next ** 2)

# 5. Truyền đạo hàm ngược (Backpropagation) từ V_next về x ban đầu
V_next.backward()

print("Trạng thái x ban đầu (requires_grad=True):\n", x.data)
print("\nTrạng thái x_next:\n", x_next.data)
print("\nĐạo hàm dV/dx (Gradient của x):\n", x.grad)