#pragma once
/// @file state.hpp
/// @brief 预报状态向量与趋势向量的内存组织。
///
/// 预报变量（dry, compressible, nonhydrostatic，[D1][D5][D13]）
/// ----------------------------------------------------------
///     u, v, w     : 速度分量 (FaceX, FaceY, FaceZ)
///     rho         : 干空气**全量**密度  rho0(z) + rho'   (Cell)
///     theta       : 位温**全量**        theta0(z) + theta' (Cell)
///     pi          : Exner **全量**       pi0(z) + pi'    (Cell)
///     qv          : 水汽混合比
///     qc, qr, qi, qs, qg : 云水、雨水、云冰、雪、霰
///
/// 参考态分离（[D1] Klemp & Wilhelmson 1978）——**数学形式 vs 存储形式**
/// -------------------------------------------------------------------
/// 控制方程按分离形式推导（这决定了线性声波-重力波项的系数只依赖 z，
/// 因此半隐式的 Helmholtz 算子系数可以离线构造）：
///
///     rho = rho0(z) + rho',  theta = theta0(z) + theta',  pi = pi0(z) + pi'
///
/// 但 **State 中存储的是三个全量场**，不是扰动场。参考态由
/// ReferenceState 单独持有（State::reference()），扰动在需要时现场相减，例如
///
///     const Real theta_p = s.theta()(i,j,k) - ref.theta0()(i,j,k);   // equations.cpp
///
/// 这样做的理由：
///   1. 守恒律（质量、能量、位涡）的检验直接对全量场做体积分，不需要还原；
///   2. 观测算子与物理参数化（饱和调整、辐射、微物理）天然使用全量热力学量，
///      避免每处都要"加回参考态"的样板块；
///   3. 状态方程反解 pi（Equations::diagnose）与海绵层松弛到参考态
///      （Damping::apply_sponge）在全量表述下都是一行。
///
/// **改动本层时必须注意**：任何新增的观测算子、TL/AD 内核与诊断量都按
/// "全量场"读写；若误按扰动场处理，会在基础态上引入常数偏置，
/// 而伴随点积检验**不会**发现这类错误（它是线性的）。
///
/// 打包顺序（供 4D-Var 控制向量使用）固定为
///     [u, v, w, rho, theta, pi, qv, qc, qr, qi, qs, qg]
/// 且只打包**内部点**，halo 由交换重建。

#include <string>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/common/types.hpp"
#include "vibe/dyn/reference_state.hpp"
#include "vibe/grid/field.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::dyn {

/// 预报变量枚举；顺序即为打包顺序
enum class Species : int {
  U = 0, V, W, Rho, Theta, Pi, Qv, Qc, Qr, Qi, Qs, Qg, Count
};

inline constexpr int kNumSpecies = static_cast<int>(Species::Count);

inline const char* species_name(Species s) noexcept {
  switch (s) {
    case Species::U: return "u";
    case Species::V: return "v";
    case Species::W: return "w";
    case Species::Rho: return "rho";
    case Species::Theta: return "theta";
    case Species::Pi: return "pi";
    case Species::Qv: return "qv";
    case Species::Qc: return "qc";
    case Species::Qr: return "qr";
    case Species::Qi: return "qi";
    case Species::Qs: return "qs";
    case Species::Qg: return "qg";
    default: return "?";
  }
}

/// 是否为水物质（需要非负约束）
inline bool is_moisture(Species s) noexcept {
  return s >= Species::Qv;
}

/// 是否为动量分量（位于面）
inline bool is_momentum(Species s) noexcept {
  return s == Species::U || s == Species::V || s == Species::W;
}

/// 各变量所属错位
inline grid::Stagger stagger_of(Species s) noexcept {
  switch (s) {
    case Species::U: return grid::Stagger::FaceX;
    case Species::V: return grid::Stagger::FaceY;
    case Species::W: return grid::Stagger::FaceZ;
    default: return grid::Stagger::Cell;
  }
}

/// 预报状态
class State {
 public:
  State() = default;
  explicit State(const grid::Grid& g) { allocate(g); }

  void allocate(const grid::Grid& g);

  grid::Field<Real>& field(Species s) { return fields_[static_cast<Size>(s)]; }
  const grid::Field<Real>& field(Species s) const { return fields_[static_cast<Size>(s)]; }

  grid::Field<Real>& u() { return field(Species::U); }
  grid::Field<Real>& v() { return field(Species::V); }
  grid::Field<Real>& w() { return field(Species::W); }
  grid::Field<Real>& rho() { return field(Species::Rho); }
  grid::Field<Real>& theta() { return field(Species::Theta); }
  grid::Field<Real>& pi() { return field(Species::Pi); }
  grid::Field<Real>& qv() { return field(Species::Qv); }

  const grid::Field<Real>& u() const { return field(Species::U); }
  const grid::Field<Real>& v() const { return field(Species::V); }
  const grid::Field<Real>& w() const { return field(Species::W); }
  const grid::Field<Real>& rho() const { return field(Species::Rho); }
  const grid::Field<Real>& theta() const { return field(Species::Theta); }
  const grid::Field<Real>& pi() const { return field(Species::Pi); }
  const grid::Field<Real>& qv() const { return field(Species::Qv); }

  /// 水凝物总量 qc+qr+qi+qs+qg
  Real total_hydrometeor(Int i, Int j, Int k) const;

  /// 参考态（构建后必须设置；用于诊断 pi0、rho0）
  void set_reference(const ReferenceState* r) noexcept { ref_ = r; }
  const ReferenceState* reference() const noexcept { return ref_; }

  Real time = Real(0);                    ///< 相对模式起报时刻（秒）
  int step = 0;                           ///< 时间步计数

  // ---- 向量运算 -----------------------------------------------------------
  /// *this = a * x + b * (*this)
  void axpy(Real a, const State& x, Real b);
  /// *this = a * x + (*this)
  void add_scaled(Real a, const State& x);
  /// *this *= a
  void scale(Real a);
  /// 欧氏范数（含全部内部点，按格点体积加权可选）
  Real norm2(Real volume_weight = Real(0)) const;
  /// 点积 <*this, x>
  Real dot(const State& x, Real volume_weight = Real(0)) const;
  /// 是否出现非有限值
  bool has_nonfinite() const;

  /// 打包控制向量（只含内部点）
  void pack(std::vector<Real>& out) const;
  /// 解包（halo 不在内，需后续交换）
  void unpack(const std::vector<Real>& in);
  /// 控制向量长度
  Size packed_size() const;

  State clone() const { return *this; }

  /// 输出摘要
  std::string describe() const;

  const grid::Grid& grid() const { VIBE_CHECK(grid_ != nullptr); return *grid_; }

 private:
  const grid::Grid* grid_ = nullptr;
  const ReferenceState* ref_ = nullptr;
  std::vector<grid::Field<Real>> fields_;   ///< size = kNumSpecies
};

}  // namespace vibe::dyn
