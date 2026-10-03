#pragma once
/// @file minimizer.hpp
/// @brief 代价函数极小化算法。
///
/// 提供的算法
/// ----------
///   * **L-BFGS**（[V13] Liu & Nocedal 1989）：有限内存拟牛顿，
///     对 4D-Var 这类大规模光滑问题收敛快，是业务系统的主流选择；
///   * **非线性共轭梯度（CG）**：内存开销最小，配合 B 预条件；
///   * **Lanczos**：用于估计 B 的特征谱与构造低秩近似（诊断）。
///
/// 线搜索
/// ------
///   * 强 Wolfe 条件的 More-Thuente 算法；
///   * 对 4D-Var 也可用固定步长 + 下降性检查（省去额外代价函数评估）。
///
/// 收敛判据
/// --------
///     ||grad J|| / max(1, |J|) < gtol   或   |J_k - J_{k-1}| < ftol
/// 同时限制迭代上限（内层循环）。
///
/// 文献：[B8] Nocedal & Wright (2006) 第 3、6、7 章；[V13] Liu & Nocedal (1989)。

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/da/da_types.hpp"

namespace vibe::da {

/// 通用目标函数接口
class Objective {
 public:
  virtual ~Objective() = default;
  virtual Real value(const Vector& x) const = 0;
  virtual void gradient(const Vector& x, Vector& g) const = 0;
  virtual Real value_and_gradient(const Vector& x, Vector& g) const = 0;
  virtual Size size() const noexcept = 0;
};

/// 极小化选项
struct MinimizerOptions {
  std::string method = "lbfgs";
  int  max_iterations = 50;
  Real gradient_tolerance = Real(1e-6);
  Real cost_tolerance = Real(1e-8);
  Real step_tolerance = Real(1e-10);
  int  memory = 10;
  Real initial_step = Real(0.1);
  bool use_strong_wolfe = true;
  bool verbose = true;
  int  print_interval = 1;
  bool use_preconditioner = true;
};

/// 极小化器接口
class Minimizer {
 public:
  virtual ~Minimizer() = default;
  /// 求解；返回最终 J
  virtual Real minimize(Objective& J, Vector& x) = 0;
  virtual int  iterations() const noexcept = 0;
  virtual int  function_evaluations() const noexcept = 0;
  virtual int  gradient_evaluations() const noexcept = 0;
  virtual Real final_gradient_norm() const noexcept = 0;
  virtual const std::vector<IterationRecord>& history() const noexcept = 0;
  virtual const char* name() const noexcept = 0;
};

std::unique_ptr<Minimizer> make_minimizer(const MinimizerOptions& opt,
                                          const Vector* preconditioner = nullptr);

/// 线搜索：满足强 Wolfe 条件的回溯法
struct LineSearchResult {
  Real step = Real(0);
  Real value = Real(0);
  Real directional_derivative = Real(0);
  int  evaluations = 0;
  bool converged = false;
};
LineSearchResult strong_wolfe_line_search(const Objective& J, const Vector& x,
                                          const Vector& direction, Real f0,
                                          Real g0, Real c1 = Real(1e-4),
                                          Real c2 = Real(0.9),
                                          Real step_max = Real(1e3));

/// 向量工具
namespace vec {
Real dot(const Vector& a, const Vector& b);
Real norm2(const Vector& a);
void axpy(Real a, const Vector& x, Vector& y);
void scale(Real a, Vector& x);
void copy(const Vector& x, Vector& y);
Real max_abs(const Vector& x);
}  // namespace vec

}  // namespace vibe::da
