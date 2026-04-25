import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
csv_path = PROJECT_ROOT / "output" / "pendulum_simulation.csv"

# Đọc dữ liệu
df = pd.read_csv(csv_path)

print("="*70)
print("PHÂN TÍCH ĐỘNG LỰC CON LẮC TỪ DỮ LIỆU MẠNG NƠRON")
print("="*70)

# --- Thống kê cơ bản ---
print(f"\n1. THỐNG KÊ CƠ BẢN:")
print(f"   - Thời gian mô phỏng: {df['time'].min():.2f}s → {df['time'].max():.2f}s")
print(f"   - Số điểm dữ liệu: {len(df)}")
print(f"   - Dt: {(df['time'].iloc[1] - df['time'].iloc[0]):.4f}s")

# --- Kiểm tra ROA ---
in_roa_count = df['in_roa'].sum()
print(f"\n2. TRẠNG THÁI ĐIỀU KHIỂN:")
print(f"   - Thời gian trong ROA (NN Stabilization): {in_roa_count} bước ({100*in_roa_count/len(df):.1f}%)")
print(f"   - Thời gian Swing-up: {len(df) - in_roa_count} bước ({100*(1-in_roa_count/len(df)):.1f}%)")

# --- Phân tích góc theta ---
print(f"\n3. PHÂN TÍCH GÓC THETA (θ):")
print(f"   - Min: {df['theta'].min():.4f} rad ({np.degrees(df['theta'].min()):.1f}°)")
print(f"   - Max: {df['theta'].max():.4f} rad ({np.degrees(df['theta'].max()):.1f}°)")
print(f"   - Mean: {df['theta'].mean():.4f} rad ({np.degrees(df['theta'].mean()):.1f}°)")
print(f"   - Std Dev: {df['theta'].std():.4f} rad")

# --- Kiểm tra vào ROA (|θ|<0.3, |ω|<0.7) ---
roa_threshold_theta = 0.30
roa_threshold_omega = 0.70
can_enter_roa = (np.abs(df['theta']) < roa_threshold_theta) & (np.abs(df['theta_dot']) < roa_threshold_omega)
print(f"   - Có thể vào ROA (|θ|<{roa_threshold_theta}, |ω|<{roa_threshold_omega}): {can_enter_roa.sum()} bước")
if can_enter_roa.sum() > 0:
    first_roa_time = df[can_enter_roa]['time'].iloc[0]
    print(f"     → Lần đầu vào ROA: t={first_roa_time:.2f}s")
else:
    print(f"     → ❌ KHÔNG BAO GIỜ THỎA ĐIỀU KIỆN VÀO ROA!")
    
# --- Phân tích vận tốc góc ---
print(f"\n4. PHÂN TÍCH VẬN TỐC GÓC (ω = dθ/dt):")
print(f"   - Min: {df['theta_dot'].min():.4f} rad/s")
print(f"   - Max: {df['theta_dot'].max():.4f} rad/s")
print(f"   - Mean: {df['theta_dot'].mean():.4f} rad/s")
print(f"   - Std Dev: {df['theta_dot'].std():.4f} rad/s")

# --- Phân tích năng lượng ---
print(f"\n5. PHÂN TÍCH NĂNG LƯỢNG (E):")
m, l, g = 0.15, 0.5, 9.81
E_upright = m * g * l  # Năng lượng ở θ=0 (cân bằng trên)
print(f"   - E_upright (tại θ=0): {E_upright:.6f} J")
print(f"   - E_current min: {df['E'].min():.6f} J")
print(f"   - E_current max: {df['E'].max():.6f} J")
print(f"   - E_current mean: {df['E'].mean():.6f} J")

energy_excess = df['E'] - E_upright
print(f"   - Dư năng lượng (ΔE = E - E_upright):")
print(f"     · Min: {energy_excess.min():.6f} J")
print(f"     · Max: {energy_excess.max():.6f} J")
print(f"     · Mean: {energy_excess.mean():.6f} J")

# --- Kiểm tra sự hội tụ cuối cùng ---
print(f"\n6. TRẠNG THÁI CUỐI CÙNG (30 bước cuối):")
final_window = df.tail(30)
print(f"   - θ cuối: {final_window['theta'].iloc[-1]:.6f} rad ({np.degrees(final_window['theta'].iloc[-1]):.2f}°)")
print(f"   - ω cuối: {final_window['theta_dot'].iloc[-1]:.6f} rad/s")
print(f"   - E cuối: {final_window['E'].iloc[-1]:.6f} J")
print(f"   - u cuối: {final_window['u'].iloc[-1]:.6f} N·m")
print(f"   - |θ| trung bình: {np.abs(final_window['theta']).mean():.6f} rad")
print(f"   - |ω| trung bình: {np.abs(final_window['theta_dot']).mean():.6f} rad/s")

# --- Kiểm tra ổn định ---
print(f"\n7. ĐÁNH GIÁ SỰ ỔN ĐỊNH:")
near_upright_zone = np.abs(df['theta']) < 0.30
print(f"   - Thời gian gần cân bằng (|θ|<0.30): {near_upright_zone.sum()} bước")

final_energy_excess = final_window['E'].mean() - E_upright
print(f"   - Dư năng lượng trung bình trong 30 bước cuối: {final_energy_excess:.6f} J")

# Kiểm tra xu hướng energy
recent_energy_trend = final_window['E'].iloc[-10:].mean() - final_window['E'].iloc[:10].mean()
print(f"   - Xu hướng năng lượng (bước 20-30 vs 0-10): {recent_energy_trend:.6f} J " + 
      ("(↑ tăng)" if recent_energy_trend > 0 else "(↓ giảm)" if recent_energy_trend < 0 else "(~ ổn định)"))

print(f"\n{'='*70}")
print("KẾT LUẬN:")
print(f"{'='*70}")

if can_enter_roa.sum() == 0:
    print("❌ VẤNĐỀ NGHIÊM TRỌNG:")
    print("   - Con lắc KHÔNG BAO GIỜ thỏa điều kiện vào ROA (|θ|<0.3, |ω|<0.7)")
    print("   - Điều khiển NN chưa bao giờ được kích hoạt!")
    print("   - Chỉ dùng swing-up control, nhưng không đủ để ổn định")
    print("\n💡 GIẢI PHÁP:")
    print("   1. ROA quá nhỏ? Cần tăng roa_enter_theta, roa_enter_omega")
    print("   2. Huấn luyện NN cho ROA lớn hơn")
    print("   3. Điều chỉnh tham số swing-up (swingup_gain, damping_gain, etc.)")
    print("   4. Tập trung vào bài toán 'balance only' thay vì swing-up+balance")
else:
    if in_roa_count / len(df) > 0.5:
        print("✓ NN Controller đã được sử dụng, đang kiểm tra ổn định...")
        if np.abs(final_window['theta']).mean() < 0.1:
            print("✓ Con lắc đang ổn định gần cân bằng")
        else:
            print("⚠ Con lắc vẫn còn dao động, không hoàn toàn ổn định")
    else:
        print("⚠ NN Controller hiếm khi được dùng, swing-up control là chủ yếu")
        print("  Cần điều chỉnh ROA hoặc tham số swing-up control")

print("\n" + "="*70)

# --- Vẽ biểu đồ ---
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Góc theta
axes[0, 0].plot(df['time'], df['theta'], 'b-', linewidth=1)
axes[0, 0].axhline(0.30, color='g', linestyle='--', alpha=0.5, label='ROA enter (0.30)')
axes[0, 0].axhline(-0.30, color='g', linestyle='--', alpha=0.5)
axes[0, 0].set_ylabel('θ (rad)')
axes[0, 0].set_title('Góc θ theo thời gian')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Vận tốc góc
axes[0, 1].plot(df['time'], df['theta_dot'], 'r-', linewidth=1)
axes[0, 1].axhline(0.70, color='g', linestyle='--', alpha=0.5, label='ROA enter (0.70)')
axes[0, 1].axhline(-0.70, color='g', linestyle='--', alpha=0.5)
axes[0, 1].set_ylabel('ω (rad/s)')
axes[0, 1].set_title('Vận tốc góc ω theo thời gian')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Năng lượng
axes[1, 0].plot(df['time'], df['E'], 'purple', linewidth=1, label='E (actual)')
axes[1, 0].axhline(E_upright, color='orange', linestyle='--', linewidth=2, label=f'E_upright = {E_upright:.4f}')
axes[1, 0].set_ylabel('E (J)')
axes[1, 0].set_title('Năng lượng E theo thời gian')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Điều khiển u
axes[1, 1].plot(df['time'], df['u'], 'brown', linewidth=1)
axes[1, 1].set_ylabel('u (N·m)')
axes[1, 1].set_title('Lực điều khiển u theo thời gian')
axes[1, 1].grid(True, alpha=0.3)

# Phase plane (ω vs θ)
axes[2, 0].plot(df['theta'], df['theta_dot'], 'k-', linewidth=0.5, alpha=0.7)
# Draw ROA boundary
theta_roa = np.linspace(-0.30, 0.30, 100)
omega_roa = 0.70
axes[2, 0].fill(theta_roa, omega_roa*np.ones_like(theta_roa), alpha=0.2, color='green', label='ROA')
axes[2, 0].fill(theta_roa, -omega_roa*np.ones_like(theta_roa), alpha=0.2, color='green')
axes[2, 0].set_xlabel('θ (rad)')
axes[2, 0].set_ylabel('ω (rad/s)')
axes[2, 0].set_title('Phase plane (ω vs θ)')
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].legend()

# Excess energy
axes[2, 1].plot(df['time'], energy_excess, 'orange', linewidth=1)
axes[2, 1].axhline(0, color='k', linestyle='-', alpha=0.3)
axes[2, 1].set_ylabel('ΔE = E - E_upright (J)')
axes[2, 1].set_title('Dư năng lượng theo thời gian')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "output" / "pendulum_analysis.png", dpi=150, bbox_inches='tight')
print("\n📊 Đã lưu biểu đồ: output/pendulum_analysis.png")

plt.show()
