import sympy as sp

# 1. Định nghĩa biến ký hiệu
theta, dot_theta, u = sp.symbols('theta dot_theta u')
m, l, mu, g, dt = sp.symbols('m l mu g dt')

# 2. Định nghĩa hệ phương trình liên tục
ddot_theta = (g / l) * sp.sin(theta) - (mu / (m * l**2)) * dot_theta + u / (m * l**2)

f1 = dot_theta
f2 = ddot_theta

# 3. Tính ma trận Jacobian A và B
A_sym = sp.Matrix([f1, f2]).jacobian([theta, dot_theta])
B_sym = sp.Matrix([f1, f2]).jacobian([u])

# 4. Tuyến tính hóa tại điểm cân bằng (0, 0)
eq_dict = {theta: 0, dot_theta: 0, u: 0}
A_eq = A_sym.subs(eq_dict)
B_eq = B_sym.subs(eq_dict)

print("Ma trận A tại x=0:")
sp.pprint(A_eq)
print("\nMa trận B tại x=0:")
sp.pprint(B_eq)

# 5. Ma trận khả điều khiển C = [B, AB]
AB_eq = A_eq * B_eq
C_matrix = sp.Matrix.hstack(B_eq, AB_eq)
print("\nHạng (Rank) của ma trận Khả điều khiển:", C_matrix.rank())