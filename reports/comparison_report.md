# Method Comparison: Quadratic Lyapunov vs alpha-CROWN

## Experiment Setup
- eval_box: [-0.1, 0.1]^2
- rho: 0.0
- pointwise_grid: 41x41
- trajectory_samples: 500, horizon: 120

## Point-wise and Trajectory Metrics
| Method | Max violation | Mean violation | Violation >= 0 | Converged | Diverged |
|---|---:|---:|---:|---:|---:|
| Quadratic (LQR + x^T P x) | -0.000000 | -0.001898 | 0.0% | 100.0% | 0.0% |
| Neural (NN + NN) | 0.000006 | -0.001220 | 0.5% | 100.0% | 0.0% |

## Formal Bounds (auto_LiRPA)
| Method | CROWN UB | alpha-CROWN UB | Certified radius CROWN | Certified radius alpha-CROWN |
|---|---:|---:|---:|---:|
| Quadratic | 0.164923 | 0.164923 | 0.000000 | 0.000000 |
| Neural | 0.025799 | 0.025799 | 0.000000 | 0.000000 |

## alpha-beta-CROWN Availability
- complete_verifier is NOT installed in this environment.
- This report uses alpha-CROWN (CROWN-Optimized) from auto_LiRPA as the tight bound baseline.
- To run full alpha-beta-CROWN, install complete_verifier and reuse the same ONNX export graph.
