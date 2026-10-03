#pragma once
/// @file acoustic_substep.hpp
/// @brief 声波子步循环（前向-后向 / HEVI）的实现细节。
///
/// 本节把"声波子步"从 RK3 中拆出，便于：
///   * 单独做线性稳定性分析；
///   * 在 GPU 上把整个子步循环融合成少量内核启动；
///   * 供半隐式积分器复用同一套线性算子系数。
///
/// 离散（地形追随坐标，[D1][D2]）
/// -------------------------------
/// 令 m = G^{1/2} = (H - z_s)/H，则质量守恒写成
///
///     d(rho' m)/dt = -m * div_h(rho0 u) - d/dzeta(rho0 w~) - rho0 m div(u')
///
/// 动量方程中的线性声波项为
///
///     du/dt ~ -cp * theta0 * dp'/dx
///     dw/dt ~ -cp * theta0 * dp'/dzeta / m
///
/// 前向-后向格式：
///     1. 用 u^tau 更新 rho^(tau+dtau)、pi'^(tau+dtau)
///     2. 用 pi'^(tau+dtau) 更新 u^(tau+dtau)
///
/// 该格式对声波等价于蛙跳格式，稳定性条件为
///
///     dtau <= 2 / ( c_s * sqrt( 1/dx^2 + 1/dy^2 + 1/dz^2 ) )
///
/// 垂直隐式（HEVI）时垂直部分改为 Crank-Nicolson，得到三对角方程
///
///     -a_k pi'_{k-1} + b_k pi'_k - c_k pi'_{k+1} = f_k
///
/// 系数只依赖参考态与 dtau（[T2][T6][T8]），可以预计算并缓存。
///
/// 文献：[D1][D6][T1][T2][T5][T6][T8][T15]。

#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/dyn/tendency.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::timeint {

/// 声波子步配置
struct AcousticOptions {
  bool vertical_implicit = false;   ///< HEVI
  bool horizontal_implicit = false; ///< 全隐式（需要 3D Helmholtz）
  Real divergence_damping = Real(0);
  int  divergence_damping_order = 2;
};

/// 声波子步状态（预计算的三对角系数等）
class AcousticSubstepper {
 public:
  AcousticSubstepper(const grid::Grid& g, const AcousticOptions& opt);

  /// 由参考态与子步长构造垂直三对角系数
  void update_coefficients(const dyn::ReferenceState& ref, Real dtau);

  /// 一个前向-后向子步
  void step(dyn::State& s, Real dtau, const dyn::ReferenceState& ref);

  /// 分裂：只更新质量与 pi'
  void update_mass_and_pressure(dyn::State& s, Real dtau, const dyn::ReferenceState& ref);

  /// 分裂：只更新动量
  void update_momentum(dyn::State& s, Real dtau, const dyn::ReferenceState& ref);

  /// 垂直隐式求解（三对角）
  void solve_vertical_implicit(dyn::State& s, Real dtau, const dyn::ReferenceState& ref);

  /// 声波 CFL 数
  Real acoustic_cfl(const dyn::State& s, Real dtau) const;

  /// 给定 dt 所需的最小声波子步数
  int required_substeps(const dyn::State& s, Real dt, Real safety = Real(0.8)) const;

 private:
  const grid::Grid* grid_;
  AcousticOptions opt_;
  std::vector<Real> tri_a_, tri_b_, tri_c_;    ///< 垂直三对角系数（逐列相同）
  std::vector<Real> tri_rhs_, tri_work_;
};

}  // namespace vibe::timeint
