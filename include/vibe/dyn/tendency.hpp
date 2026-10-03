#pragma once
/// @file tendency.hpp
/// @brief 时间趋势（d/dt）容器，与 State 同布局。
///
/// 分离"大时间步趋势"与"声波趋势"是时间分裂格式的核心
/// （[D5] Skamarock & Klemp 2008；[D6] Klemp et al. 2008）：
///
///     dphi/dt = F_slow(phi) + F_acoustic(phi)
///
/// 其中 F_slow 由平流、科氏力、物理过程与（可选的）大时间步梯度力组成，
/// F_acoustic 只含线性化的声波-重力波项，在半隐式格式中被隐式处理。
///
/// 文献：[D5][D6][T5] Wicker & Skamarock (2002)。

#include <string>
#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/dyn/reference_state.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/grid/field.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::dyn {

class Tendency {
 public:
  Tendency() = default;
  explicit Tendency(const grid::Grid& g) { allocate(g); }

  /// 依据网格分配全部物种的倾向场。
  /// 布局与 dyn::State **严格一致**（u: FaceX, v: FaceY, w: FaceZ, 其余 Cell），
  /// 因此 State::add_scaled / axpy 可以直接做整体更新。
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

  void zero();
  void axpy(Real a, const Tendency& x, Real b);
  Real norm2() const;
  bool has_nonfinite() const;

 private:
  std::vector<grid::Field<Real>> fields_;
};

/// 物理参数化返回的倾向（局地变化率，已转为温度/水物质/动量形式）
struct PhysicsTendency {
  grid::Field<Real> theta;   ///< K/s
  grid::Field<Real> qv, qc, qr, qi, qs, qg;   ///< kg/kg/s
  grid::Field<Real> u, v;    ///< m/s^2（次网格动量通量辐合）
  grid::Field<Real> tke;     ///< m^2/s^3（PBL 诊断）
  Real surface_flux_heat = Real(0);   ///< W/m^2
  Real surface_flux_moist = Real(0);  ///< W/m^2
  Real surface_flux_momentum = Real(0);
  Real precipitation_rate = Real(0);  ///< mm/s（域平均）
  Real top_of_atmosphere_flux = Real(0);

  PhysicsTendency() = default;
  explicit PhysicsTendency(const grid::Grid& g) { allocate(g); }
  void allocate(const grid::Grid& g);
  void zero();
};

}  // namespace vibe::dyn
