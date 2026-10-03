#pragma once
/// @file obs_operator.hpp
/// @brief 观测算子接口（非线性 / 切线性 / 伴随三件套）。
///
/// 统一定义（[V14] Ide et al. 1997）
/// --------------------------------
///   非线性    : y = H(x)
///   切线性    : dy = H'(x) dx        （dx 为状态扰动）
///   伴随      : dx* = H'^T(x) dy     （需要 x 作为线性化点）
///
/// 三条必须满足的性质（[A3][A4]）
/// -----------------------------
///   1. **一致性**：H'(x) dx 与 (H(x + eps dx) - H(x)) / eps 在 eps -> 0 时一致；
///   2. **伴随性（点积检验）**：对所有 dx, dy，
///          < H'(x) dx , dy >_R = < dx , H'^T(x) dy >_B
///      相对误差 < 1e-10（[A3] Sirkes & Tziperman 1997）；
///   3. **稀疏性**：每个观测只依赖少数格点，实现中必须显式使用该稀疏结构。
///
/// 实现规范
/// --------
///   * `apply` 必须先把状态插值到观测点（水平双线性/三线性 + 垂直单调插值）；
///   * `applyTL` 中，**所有系数（插值权重、物理量导数）必须冻结在基础态 x 上**；
///   * `applyAD` 必须严格按 `applyTL` 的逆序累加，且插值权重要转置使用；
///   * 对包含不连续过程（如饱和调整、对流触发）的部分，采用
///     "冻结开关" 策略：用基础态的开关状态，不写不连续项的导数（[A6][A8]）。
///
/// 文献：[O1][O3][O4][O6][O8][V14][A3][A4][A6][A8]。

#include <memory>
#include <string>
#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/grid/geometry.hpp"
#include "vibe/obs/observations.hpp"

namespace vibe::config { struct DaConfig; }

namespace vibe::obs {

/// 供观测算子读取的模式状态视图（避免算子直接依赖完整 State）
struct ModelStateView {
  const dyn::State* state = nullptr;
  const dyn::ReferenceState* ref = nullptr;
  const grid::Grid* grid = nullptr;
  Real time = Real(0);
  bool valid() const noexcept { return state != nullptr && grid != nullptr; }
};

/// 观测算子基类
class ObservationOperator {
 public:
  virtual ~ObservationOperator() = default;

  virtual const char* name() const noexcept = 0;
  /// 该算子处理的观测类型
  virtual ObsType type() const noexcept = 0;

  /// y = H(x)
  virtual void apply(const ModelStateView& x, const ObsSpace& obs,
                     std::vector<Real>& y) const = 0;

  /// dy = H'(x) dx
  virtual void applyTL(const ModelStateView& x, const dyn::State& dx,
                       const ObsSpace& obs, std::vector<Real>& dy) const = 0;

  /// dx = H'^T(x) dy
  virtual void applyAD(const ModelStateView& x, const std::vector<Real>& dy,
                       const ObsSpace& obs, dyn::State& dx) const = 0;

  /// 伴随点积检验；返回最大相对误差
  Real check_adjoint(const ModelStateView& x, const ObsSpace& obs,
                     unsigned seed = 1) const;

  /// 切线性一致性检验（有限差分）；返回最大相对误差
  Real check_tangent(const ModelStateView& x, const ObsSpace& obs,
                     Real eps = Real(1e-6), unsigned seed = 1) const;

  /// 观测误差（O - H(x)）诊断
  void innovations(const ModelStateView& x, const ObsSpace& obs,
                   std::vector<Real>& d) const;

 protected:
  /// 辅助：把状态场插值到观测点（水平双线性 + 垂直线性）
  static Real sample_scalar(const grid::Field<Real>& f, const grid::Grid& g,
                            Real x, Real y, Real z);
  static Real sample_staggered(const grid::Field<Real>& f, const grid::Grid& g,
                               grid::Stagger s, Real x, Real y, Real z);
  /// 辅助：伴随插值（把观测点值按权重散射回格点）
  static void scatter_adjoint(grid::Field<Real>& f, const grid::Grid& g,
                              grid::Stagger s, Real x, Real y, Real z, Real value);
};

/// 工厂：按观测类型创建算子
std::unique_ptr<ObservationOperator> make_operator(ObsType t,
                                                   const grid::Grid& g,
                                                   const config::DaConfig& cfg);

/// 复合算子：把多个算子串成一个 H（4D-Var 中一次性处理所有观测类型）
class CompositeOperator final : public ObservationOperator {
 public:
  void add(std::unique_ptr<ObservationOperator> op);
  const char* name() const noexcept override { return "composite"; }
  ObsType type() const noexcept override { return ObsType::Count; }
  void apply(const ModelStateView& x, const ObsSpace& obs,
             std::vector<Real>& y) const override;
  void applyTL(const ModelStateView& x, const dyn::State& dx,
               const ObsSpace& obs, std::vector<Real>& dy) const override;
  void applyAD(const ModelStateView& x, const std::vector<Real>& dy,
               const ObsSpace& obs, dyn::State& dx) const override;
  Size size() const noexcept { return ops_.size(); }

 private:
  std::unique_ptr<ObservationOperator> find(ObsType t) const;
  std::vector<std::unique_ptr<ObservationOperator>> ops_;
};

}  // namespace vibe::obs
