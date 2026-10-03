#pragma once
/// @file control_vector.hpp
/// @brief 控制变量与其到模式状态的变换（B = U U^T 中的 U）。
///
/// 控制变量分解（[V7][V9]）
/// -----------------------
/// 采用顺序变换
///
///     dx = U v = K_v V E K_h v
///
///   * `K_h`：水平相关，用**扩散算子**实现（隐式扩散多次迭代近似高斯相关）
///        (I - lambda^2 Del^2)^p 的平方根
///     等价于相关函数 `rho(r) = ...`，避免显式的 TOEPLITZ 矩阵（[V6][V7]）；
///   * `E`   ：垂直模（EOF / 垂直扩散）展开，用 N 个模态近似垂直相关；
///   * `V`   ：方差与**多变量平衡**（线性平衡、omega 方程），把小尺度的
///     质量场与风场耦合起来（[V8][V9]）；
///   * `K_v` ：控制变量到模式变量的最终组装。
///
/// 控制变量的选择直接决定 4D-Var 的收敛速度与物理合理性，因此本类
/// 把所有开关暴露给配置。
///
/// 文献：[V5][V6][V7][V8][V9][V12]。

#include <memory>
#include <string>
#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/config/config.hpp"
#include "vibe/da/da_types.hpp"
#include "vibe/dyn/reference_state.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::da {

/// 平衡关系形式
enum class BalanceForm { None, LinearBalance, OmegaEquation };

const char* to_string(BalanceForm b) noexcept;
BalanceForm balance_from_string(const std::string& s);

/// 控制变量变换接口
class ControlVariableTransform {
 public:
  virtual ~ControlVariableTransform() = default;

  /// v -> dx
  virtual void to_state(const Vector& v, dyn::State& dx) const = 0;
  /// dx -> v
  virtual void from_state(const dyn::State& dx, Vector& v) const = 0;
  /// B v（通过 U U^T）
  virtual void applyB(const Vector& v, Vector& Bv) const = 0;
  /// B^{-1} v
  virtual void applyBinv(const Vector& v, Vector& Binvv) const = 0;
  /// B^{1/2} v（= U v）
  virtual void applySqrtB(const Vector& v, Vector& out) const = 0;
  /// B^{-1/2} v（= U^{-1} v）
  virtual void applyInvSqrtB(const Vector& v, Vector& out) const = 0;

  /// 预条件的代价函数 J(v) 中用到的 B 对角（方差）
  virtual const Vector& variance() const = 0;
  /// 控制向量维度
  virtual Size size() const noexcept = 0;
  virtual const char* name() const noexcept = 0;

  /// 诊断：打印方差范围、长度尺度、EOF 能量占比
  virtual std::string describe() const = 0;
};

/// 通用实现
struct ControlVariableConfig {
  BalanceForm balance = BalanceForm::LinearBalance;
  Real horizontal_length_scale = Real(200000);
  Real vertical_length_scale = Real(3000);
  int  diffusion_iterations = 4;      ///< 扩散算子的迭代次数 p
  Real diffusion_order = Real(2);     ///< 阶数（2 = Del^2, 4 = Del^4）
  int  n_vertical_modes = 10;
  bool use_ensemble_component = false;
  Real hybrid_weight = Real(0.5);
  std::vector<Real> variance_unbalance;   ///< 不平衡部分的方差占比
  Real level_scale_unbalance = Real(1);
};

/// 扩散型相关算子（[V6][V7]）
class DiffusionCorrelation : public LinearOperator {
 public:
  DiffusionCorrelation(const grid::Grid& g, Real length_scale, int iterations,
                       Real order);
  void apply(const Vector& x, Vector& y) const override;
  Size size() const noexcept override { return n_; }
  const char* name() const noexcept override { return "diffusion_correlation"; }

  Real length_scale() const noexcept { return length_scale_; }
  int  iterations() const noexcept { return iterations_; }

 private:
  /// 一次隐式扩散步（推进方向为"平滑"）：(I - c Del^2) y = x
  void smooth_once(const Vector& in, Vector& out) const;
  const grid::Grid* grid_;
  Real length_scale_;
  int iterations_;
  Real order_;
  Size n_ = 0;
};

/// 垂直 EOF 展开（[V9]）
class VerticalEofTransform {
 public:
  VerticalEofTransform() = default;
  VerticalEofTransform(std::vector<std::vector<Real>> eofs,
                       std::vector<Real> variances);

  /// 由 NMC 样本估计 EOF（[V5][V12]）
  static VerticalEofTransform estimate_from_samples(
      const std::vector<std::vector<Real>>& samples, int n_modes);

  void expand(const std::vector<Real>& modes, std::vector<Real>& profile) const;
  void project(const std::vector<Real>& profile, std::vector<Real>& modes) const;

  int n_modes() const noexcept { return static_cast<int>(eofs_.size()); }
  int n_levels() const noexcept { return n_levels_; }
  const std::vector<Real>& variance() const noexcept { return variance_; }
  /// 前 k 个模态累计解释方差
  Real explained_variance(int k) const;

 private:
  std::vector<std::vector<Real>> eofs_;   ///< [mode][level]
  std::vector<Real> variance_;            ///< [mode]
  int n_levels_ = 0;
};

/// 多变量平衡（[V8][V9]）
class BalanceOperator {
 public:
  BalanceOperator(const grid::Grid& g, const dyn::ReferenceState& ref,
                  BalanceForm form);

  /// 由质量场（pi' 或 theta'）诊断风场的不平衡分量
  void derive_wind(const grid::Field<Real>& mass, grid::Field<Real>& u,
                   grid::Field<Real>& v) const;

  /// 伴随：由风场诊断质量场贡献
  void derive_mass_adjoint(const grid::Field<Real>& u, const grid::Field<Real>& v,
                           grid::Field<Real>& mass) const;

  /// 线性平衡关系中的系数（依赖参考态）
  void update_coefficients();

  BalanceForm form() const noexcept { return form_; }

 private:
  const grid::Grid* grid_;
  const dyn::ReferenceState* ref_;
  BalanceForm form_;
  Real f0_ = Real(1e-4);
};

/// 工厂
std::unique_ptr<ControlVariableTransform> make_control_variable_transform(
    const grid::Grid& g, const dyn::ReferenceState& ref,
    const config::DaConfig& da_cfg, const ControlVariableConfig& cv_cfg);

}  // namespace vibe::da
