/// @file tendency.cpp
/// @brief Trend 容器的分配与向量运算实现。
///
/// 之所以把 Tendency 与 PhysicsTendency 放在同一个翻译单元：
/// 两者都是"与 State 同布局的倾向容器"，共享同一套分配与清零逻辑，
/// 且都会被 TL/AD 以完全相同的方式使用（TL 只需要 axpy/zero）。
///
/// 布局约定（与 dyn::State 严格一致，见 include/vibe/dyn/tendency.hpp）：
///   * u  -> FaceX（x 方向比体心多一列，i = 0..nx）
///   * v  -> FaceY
///   * w  -> FaceZ
///   * 其余 -> Cell
/// 该一致性是 `State::add_scaled` / `Field::add_scaled` 能够直接做整体
/// 更新的前提（它们在内部断言两侧存储大小相同）。

#include "vibe/dyn/tendency.hpp"

#include <algorithm>
#include <cmath>

#include "vibe/common/error.hpp"

namespace vibe::dyn {

// ===========================================================================
// Tendency
// ===========================================================================

void Tendency::allocate(const grid::Grid& g) {
  fields_.clear();
  fields_.reserve(static_cast<Size>(kNumSpecies));
  for (int s = 0; s < kNumSpecies; ++s) {
    const Species sp = static_cast<Species>(s);
    // 名字加后缀，便于 NetCDF 输出与调试时区分"状态量"与"趋势量"
    fields_.emplace_back(g, stagger_of(sp), std::string(species_name(sp)) + "_dt");
  }
}

void Tendency::zero() {
  for (auto& f : fields_) f.fill(Real(0));
}

void Tendency::axpy(Real a, const Tendency& x, Real b) {
  VIBE_CHECK(x.fields_.size() == fields_.size());
  for (Size s = 0; s < fields_.size(); ++s) {
    fields_[s].axpy(a, x.fields_[s], b);
  }
}

Real Tendency::norm2() const {
  Real acc = Real(0);
  for (int s = 0; s < kNumSpecies; ++s) {
    const Species sp = static_cast<Species>(s);
    const grid::Field<Real>& f = fields_[static_cast<Size>(s)];
    const Int ni = f.nx() + ((sp == Species::U) ? 1 : 0);
    const Int nj = f.ny() + ((sp == Species::V) ? 1 : 0);
    const Int nk = f.nz() + ((sp == Species::W) ? 1 : 0);
    for (Int k = 0; k < nk; ++k)
      for (Int j = 0; j < nj; ++j)
        for (Int i = 0; i < ni; ++i) acc += sqr(f(i, j, k));
  }
  return std::sqrt(acc);
}

bool Tendency::has_nonfinite() const {
  for (const auto& f : fields_) {
    if (f.has_nonfinite()) return true;
  }
  return false;
}

// ===========================================================================
// PhysicsTendency
// ===========================================================================

void PhysicsTendency::allocate(const grid::Grid& g) {
  auto cell = [&](const char* n) { return grid::Field<Real>(g, grid::Stagger::Cell, n); };
  theta = cell("phys_dtheta");
  qv = cell("phys_dqv");
  qc = cell("phys_dqc");
  qr = cell("phys_dqr");
  qi = cell("phys_dqi");
  qs = cell("phys_dqs");
  qg = cell("phys_dqg");
  tke = cell("phys_tke");
  // 动量倾向写在面点上，才能被 Tendency::u()/v() 直接整体累加
  u = grid::Field<Real>(g, grid::Stagger::FaceX, "phys_du");
  v = grid::Field<Real>(g, grid::Stagger::FaceY, "phys_dv");
  surface_flux_heat = Real(0);
  surface_flux_moist = Real(0);
  surface_flux_momentum = Real(0);
  precipitation_rate = Real(0);
  top_of_atmosphere_flux = Real(0);
}

void PhysicsTendency::zero() {
  theta.fill(Real(0));
  qv.fill(Real(0));
  qc.fill(Real(0));
  qr.fill(Real(0));
  qi.fill(Real(0));
  qs.fill(Real(0));
  qg.fill(Real(0));
  u.fill(Real(0));
  v.fill(Real(0));
  tke.fill(Real(0));
  surface_flux_heat = Real(0);
  surface_flux_moist = Real(0);
  surface_flux_momentum = Real(0);
  precipitation_rate = Real(0);
  top_of_atmosphere_flux = Real(0);
}

}  // namespace vibe::dyn
