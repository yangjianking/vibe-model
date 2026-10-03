#pragma once
/// @file helmholtz.hpp
/// @brief 三维 Helmholtz 求解器（半隐式时间推进的核心）。
///
/// 方程
/// ----
/// 半隐式（Crank-Nicolson / trapezoidal）时间离散后，Exner 扰动满足
///
///     div( a grad(pi') ) - b pi' = f ,    在域内
///     pi' = 0            ,                在上下边界
///
/// 其中
///     a = (dt/2)^2 * cp * theta0 * rho0 / m        （在面上）
///     b = 1 / (rho0 c_s^2)                         （在体心）
///
/// 系数 a、b 在参考态与 dt 确定后是空间的已知函数（通常只依赖 z 与地形），
/// 因此可以在多个时间步内复用。
///
/// 关于 "pi' 还是 pi"：推导中写 pi' 是因为参考态分离的表述更自然；但
///   * 参考态 pi0(z) 时间不变；
///   * Crank-Nicolson 只出现时间平均 (pi^{n+1} + pi^n)/2 与差分 (pi^{n+1} - pi^n)；
/// 因此对 pi 与对 pi' 建立的算子完全相同（只差一个不随时间变化的常数平移）。
/// 实现中求解器直接作用于 State::pi() 所在的**全量** Exner 场，与上式一致。
///
/// 求解器族
/// --------
///   * `VerticalTridiagonal`：只隐式处理垂直方向（HEVI），逐列 Thomas 算法，
///     复杂度 O(nx ny nz)，无迭代、无残差，是最快也最常用的选择
///     （[T2] Tapp & White 1976；[T6] Cullen 1990）。
///   * `KrylovJacobi`：完整的 3D 求解，用 Jacobi/Chebyshev 预条件的
///     GMRES（[T11]）或 BiCGSTAB（[T12]）迭代。
///   * `KrylovMultigrid`：以几何多重网格（[T13]）为预条件子的 Krylov 方法，
///     网格间转移算子复用嵌套的 restrict/prolong，收敛率与网格无关。
///
/// 收敛判据与精度
/// --------------
///     相对残差 ||r||_2 / ||f||_2 < tol
/// 混合精度下采用**迭代精化**（[G5]）：以单精度求解、双精度计算残差并修正，
/// 直到双精度残差满足 tol。这在大规模 GPU 计算中可显著加速。
///
/// 文献：[T2][T6][T7][T8][T9][T10][T11][T12][T13][G5]。

#include <memory>
#include <string>
#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/gpu/precision.hpp"
#include "vibe/grid/field.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::config { struct ModelConfig; }

namespace vibe::timeint {

/// 求解器种类
enum class HelmholtzKind { VerticalTridiagonal, KrylovMultigrid, KrylovJacobi };

HelmholtzKind helmholtz_kind_from_string(const std::string& s);
const char* to_string(HelmholtzKind k) noexcept;

/// 求解器接口
class HelmholtzSolver {
 public:
  virtual ~HelmholtzSolver() = default;

  /// 设置算子系数：a 在面（可用体心场近似），b 在体心
  virtual void set_coefficients(const grid::Field<Real>& a,
                                const grid::Field<Real>& b) = 0;

  /// 求解 div(a grad x) - b x = f
  virtual void solve(const grid::Field<Real>& a,
                     const grid::Field<Real>& b,
                     const grid::Field<Real>& f,
                     grid::Field<Real>& x) = 0;

  /// 迭代次数（直接法返回 1）
  virtual int iterations() const noexcept = 0;
  /// 最终相对残差
  virtual Real residual() const noexcept = 0;
  /// 名称
  virtual const char* name() const noexcept = 0;

  /// 是否支持迭代精化（混合精度，[G5]）
  virtual void set_iterative_refinement(bool /*on*/, vibe::gpu::Precision /*reduce*/) {}
};

/// 工厂
std::unique_ptr<HelmholtzSolver> make_helmholtz_solver(HelmholtzKind kind,
                                                       const grid::Grid& g,
                                                       const config::ModelConfig& cfg);

/// 逐列 Thomas 算法（供直接法与预条件使用）；输入长度 nz
/// a: 次对角 (nz-1)，b: 主对角 (nz)，c: 上对角 (nz-1)，d: 右端 (nz)，x: 解
void thomas_solve(const std::vector<Real>& a, const std::vector<Real>& b,
                  const std::vector<Real>& c, const std::vector<Real>& d,
                  std::vector<Real>& x);

/// 三对角矩阵的 LU 分解（系数不变时可复用）
struct TridiagonalFactorization {
  std::vector<Real> sub;       ///< 次对角 (nz-1)，回代需要
  std::vector<Real> c_prime;   ///< 修正的上对角 (nz-1)
  std::vector<Real> denom;     ///< 主元 (nz)
  bool factorized = false;
};

void thomas_factorize(const std::vector<Real>& a, const std::vector<Real>& b,
                      const std::vector<Real>& c, TridiagonalFactorization& f);

void thomas_solve_factored(const TridiagonalFactorization& f,
                           const std::vector<Real>& d, std::vector<Real>& x);

/// 几何多重网格的转移算子（与嵌套模块共用实现，[N4][T13]）
void multigrid_restrict(const grid::Field<Real>& fine, grid::Field<Real>& coarse);
void multigrid_prolong(const grid::Field<Real>& coarse, grid::Field<Real>& fine);

/// 残差计算 ||div(a grad x) - b x - f||_2
Real helmholtz_residual_l2(const grid::Grid& g, const grid::Field<Real>& a,
                           const grid::Field<Real>& b, const grid::Field<Real>& f,
                           const grid::Field<Real>& x);

}  // namespace vibe::timeint
