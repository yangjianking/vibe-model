#pragma once
/// @file diagnostics.hpp
/// @brief 由预报状态导出的诊断量。
///
/// 提供：相对涡度、散度、垂直速度、位涡、位温、相当位温、露点、
/// 相对湿度、CAPE/CIN、对流有效位能、反射率、海平面气压、
/// 动能/内能/总能量（用于守恒检验）。
///
/// 位涡（Ertel PV，[B4] Holton & Hakim 第 4 章）
/// ---------------------------------------------
///     PV = (1/rho) * (omega_a . grad(theta)),
///     omega_a = ( -dv/dz, du/dz, f + dv/dx - du/dy )
///
/// CAPE（[B4] 第 3 章；[P11] Kain-Fritsch 气块法）
/// ----------------------------------------------
///     CAPE = g * integral_{z_LFC}^{z_EL} (theta_v,parcel - theta_v,env)/theta_v,env dz
///
/// 检验与后处理均使用这些诊断量，因此实现集中在 dyn 层，Python 侧只做包装。

#include <string>
#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/dyn/reference_state.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/grid/field.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::dyn {

/// 诊断量集合
struct Diagnostics {
  grid::Field<Real> vorticity;    ///< 相对涡度 zeta (1/s)
  grid::Field<Real> divergence;   ///< 水平散度 (1/s)
  grid::Field<Real> pv;           ///< Ertel 位涡 (K m^2 kg^-1 s^-1)
  grid::Field<Real> pressure;     ///< 气压 (Pa)
  grid::Field<Real> temperature;  ///< 温度 (K)
  grid::Field<Real> theta_e;      ///< 相当位温 (K)
  grid::Field<Real> rh;           ///< 相对湿度 (0-1)
  grid::Field<Real> dewpoint;     ///< 露点 (K)
  grid::Field<Real> height;       ///< 位势高度 (m)
  grid::Field<Real> reflectivity; ///< 雷达反射率 (dBZ)

  Diagnostics() = default;
  explicit Diagnostics(const grid::Grid& g) { allocate(g); }
  void allocate(const grid::Grid& g);
};

/// 全量诊断（除 CAPE，后者按列计算）
void diagnose_all(const State& s, const ReferenceState& ref,
                  Diagnostics& out);

/// 单点诊断
Real relative_vorticity(const State& s, const grid::Grid& g, Int i, Int j, Int k);
Real divergence(const State& s, const grid::Grid& g, Int i, Int j, Int k);
Real ertel_pv(const State& s, const ReferenceState& ref, Int i, Int j, Int k);
Real exner_to_pressure(Real pi);
Real exner_to_temperature(Real pi, Real theta);
Real theta_e_from_rho(Real theta, Real qv, Real pi);
Real relative_humidity(Real pi, Real theta, Real qv, Real qc = Real(0));
Real dewpoint_from_qv(Real pressure, Real qv);

/// 单列 CAPE / CIN（气块法，[P11]）
struct CapeResult { Real cape = Real(0), cin = Real(0), lfc = Real(0), el = Real(0); };
CapeResult cape_cin_column(const std::vector<Real>& z,
                           const std::vector<Real>& p,
                           const std::vector<Real>& t,
                           const std::vector<Real>& qv,
                           Real z_start);
CapeResult cape_cin(const State& s, const ReferenceState& ref, Int i, Int j);

/// 雷达反射率（[P3] Thompson et al. 2008 式 (A1) 的简化形式）
Real reflectivity_dbz(Real qr, Real qs, Real qg, Real rho, Real t);

/// 海平面气压（静力外推，[D16] WRF 第 3 章）
Real sea_level_pressure(const State& s, const ReferenceState& ref, Int i, Int j);

/// 全区域能量收支（守恒检验用）
struct EnergyBudget {
  Real kinetic = Real(0), internal = Real(0), potential = Real(0),
       latent = Real(0), total = Real(0), mass = Real(0);
  std::string to_string() const;
};
EnergyBudget energy_budget(const State& s, const ReferenceState& ref);

}  // namespace vibe::dyn
