import numpy as np

# Test wrapping formula
theta_upright_raw = np.pi

test_cases = [
    (np.pi + 0.01, "upright + 0.01"),
    (6.2, "6.2 rad (after 1+ rotation)"),
    (np.pi, "exactly π"),
    (2*np.pi + np.pi + 0.01, "2π + upright + 0.01"),
]

print("Testing wrapping formula:")
print("=" * 70)
print(f"theta_upright_raw = π = {np.pi:.4f}")
print()

for theta_raw, label in test_cases:
    # Current formula in code
    theta_norm_v1 = (theta_raw - theta_upright_raw + np.pi) % (2 * np.pi) - np.pi
    
    # Expected correct formula - center around 0 with range [-π, π]
    theta_diff = theta_raw - theta_upright_raw
    theta_norm_v2 = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
    
    print(f"θ_raw = {theta_raw:.4f} rad ({theta_raw*180/np.pi:.1f}°) [{label}]")
    print(f"  Formula v1: ({theta_raw:.4f} - {theta_upright_raw:.4f} + π) % (2π) - π = {theta_norm_v1:.4f}")
    print(f"  Formula v2 (arctan2): {theta_norm_v2:.4f}")
    print(f"  Diff: {theta_norm_v1 - theta_norm_v2:.6f}")
    print()

print()
print("Checking what happened at end of simulation:")
print("="*70)
theta_raw_final = 6.2016
theta_upright_raw = np.pi

theta_norm_final = (theta_raw_final - theta_upright_raw + np.pi) % (2 * np.pi) - np.pi
print(f"θ_raw = 6.2016 rad")
print(f"theta_norm = (6.2016 - π + π) % (2π) - π")
print(f"           = (6.2016) % (2π) - π")
print(f"           = {6.2016 % (2*np.pi):.4f} - {np.pi:.4f}")
print(f"           = {theta_norm_final:.4f}")
print()

# What should it be?
theta_normalized_correct = np.arctan2(np.sin(theta_raw_final - theta_upright_raw), 
                                       np.cos(theta_raw_final - theta_upright_raw))
print(f"Correct normalization (arctan2): {theta_normalized_correct:.4f}")
print()

# Actually, let me think about this differently
# If θ_raw = 6.2 rad ≈ 2π + 3.14/π ≈ 2π - 0.06
# Then relative to upright (π), it should be:
# 6.2 - π ≈ 3.06, but wrapped to [-π, π]:
# 3.06 > π, so we subtract 2π: 3.06 - 2π ≈ 3.06 - 6.28 ≈ -3.22, but that's wrong too

print("Let's think step by step:")
print(f"θ_raw_final = 6.2016")
print(f"θ_raw_final - π = 6.2016 - 3.1416 = {6.2016 - np.pi:.4f}")
print(f"This is > π, so we need to subtract 2π:")
print(f" {6.2016 - np.pi:.4f} - 2π = {6.2016 - np.pi - 2*np.pi:.4f}")
print()
print(f"Using modulo wrapping: (({6.2016 - np.pi:.4f} + π) % 2π) - π")
val = (6.2016 - np.pi + np.pi)
print(f"  = ({val:.4f} % 2π) - π")
print(f"  = {val % (2*np.pi):.4f} - π")
print(f"  = {(val % (2*np.pi)) - np.pi:.4f}")
