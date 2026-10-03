/// @file state.cpp
/// @brief 预报状态向量的分配与向量运算。

#include "vibe/dyn/state.hpp"

#include <cmath>
#include <sstream>

namespace vibe::dyn {

void State::allocate(const grid::Grid& g) {
  grid_ = &g;
  fields_.clear();
  fields_.reserve(static_cast<Size>(kNumSpecies));
  for (int s = 0; s < kNumSpecies; ++s) {
    const Species sp = static_cast<Species>(s);
    fields_.emplace_back(g, stagger_of(sp), species_name(sp));
  }
}

Real State::total_hydrometeor(Int i, Int j, Int k) const {
  return field(Species::Qc)(i, j, k) + field(Species::Qr)(i, j, k) +
         field(Species::Qi)(i, j, k) + field(Species::Qs)(i, j, k) +
         field(Species::Qg)(i, j, k);
}

void State::axpy(Real a, const State& x, Real b) {
  for (int s = 0; s < kNumSpecies; ++s) {
    fields_[static_cast<Size>(s)].axpy(a, x.fields_[static_cast<Size>(s)], b);
  }
}

void State::add_scaled(Real a, const State& x) {
  for (int s = 0; s < kNumSpecies; ++s) {
    fields_[static_cast<Size>(s)].add_scaled(a, x.fields_[static_cast<Size>(s)]);
  }
}

void State::scale(Real a) {
  for (auto& f : fields_) f.scale(a);
}

Real State::norm2(Real volume_weight) const {
  Real s = Real(0);
  for (int sp = 0; sp < kNumSpecies; ++sp) {
    const grid::Field<Real>& f = fields_[static_cast<Size>(sp)];
    const Species species = static_cast<Species>(sp);
    const bool face = is_momentum(species);
    const Int ni = f.nx() + ((species == Species::U) ? 1 : 0);
    const Int nj = f.ny() + ((species == Species::V) ? 1 : 0);
    const Int nk = f.nz() + ((species == Species::W) ? 1 : 0);
    VIBE_UNUSED(face);
    for (Int k = 0; k < nk; ++k)
      for (Int j = 0; j < nj; ++j)
        for (Int i = 0; i < ni; ++i) {
          const Real v = f(i, j, k);
          Real w = Real(1);
          if (volume_weight > Real(0)) {
            w = volume_weight * grid_->dx_at(std::min(i, grid_->nx() - 1)) *
                grid_->dy_at(std::min(j, grid_->ny() - 1)) *
                grid_->jacobian(std::min(i, grid_->nx() - 1), std::min(j, grid_->ny() - 1)) *
                grid_->dzeta(std::min(k, grid_->nz() - 1)) * grid_->geom().z_top;
          }
          s += w * v * v;
        }
  }
  return std::sqrt(s);
}

Real State::dot(const State& x, Real volume_weight) const {
  Real s = Real(0);
  for (int sp = 0; sp < kNumSpecies; ++sp) {
    const grid::Field<Real>& a = fields_[static_cast<Size>(sp)];
    const grid::Field<Real>& b = x.fields_[static_cast<Size>(sp)];
    const Species species = static_cast<Species>(sp);
    const Int ni = a.nx() + ((species == Species::U) ? 1 : 0);
    const Int nj = a.ny() + ((species == Species::V) ? 1 : 0);
    const Int nk = a.nz() + ((species == Species::W) ? 1 : 0);
    for (Int k = 0; k < nk; ++k)
      for (Int j = 0; j < nj; ++j)
        for (Int i = 0; i < ni; ++i) {
          Real w = Real(1);
          if (volume_weight > Real(0)) {
            const Int ii = std::min(i, grid_->nx() - 1);
            const Int jj = std::min(j, grid_->ny() - 1);
            const Int kk = std::min(k, grid_->nz() - 1);
            w = volume_weight * grid_->dx_at(ii) * grid_->dy_at(jj) *
                grid_->jacobian(ii, jj) * grid_->dzeta(kk) * grid_->geom().z_top;
          }
          s += w * a(i, j, k) * b(i, j, k);
        }
  }
  return s;
}

bool State::has_nonfinite() const {
  for (const auto& f : fields_) {
    if (f.has_nonfinite()) return true;
  }
  return false;
}

Size State::packed_size() const {
  Size n = 0;
  for (int sp = 0; sp < kNumSpecies; ++sp) {
    const Species species = static_cast<Species>(sp);
    const grid::Field<Real>& f = fields_[static_cast<Size>(sp)];
    const Int ni = f.nx() + ((species == Species::U) ? 1 : 0);
    const Int nj = f.ny() + ((species == Species::V) ? 1 : 0);
    const Int nk = f.nz() + ((species == Species::W) ? 1 : 0);
    n += static_cast<Size>(ni) * static_cast<Size>(nj) * static_cast<Size>(nk);
  }
  return n;
}

void State::pack(std::vector<Real>& out) const {
  out.clear();
  out.reserve(packed_size());
  for (int sp = 0; sp < kNumSpecies; ++sp) {
    const Species species = static_cast<Species>(sp);
    const grid::Field<Real>& f = fields_[static_cast<Size>(sp)];
    const Int ni = f.nx() + ((species == Species::U) ? 1 : 0);
    const Int nj = f.ny() + ((species == Species::V) ? 1 : 0);
    const Int nk = f.nz() + ((species == Species::W) ? 1 : 0);
    for (Int k = 0; k < nk; ++k)
      for (Int j = 0; j < nj; ++j)
        for (Int i = 0; i < ni; ++i) out.push_back(f(i, j, k));
  }
}

void State::unpack(const std::vector<Real>& in) {
  VIBE_CHECK(in.size() == packed_size());
  Size n = 0;
  for (int sp = 0; sp < kNumSpecies; ++sp) {
    const Species species = static_cast<Species>(sp);
    grid::Field<Real>& f = fields_[static_cast<Size>(sp)];
    const Int ni = f.nx() + ((species == Species::U) ? 1 : 0);
    const Int nj = f.ny() + ((species == Species::V) ? 1 : 0);
    const Int nk = f.nz() + ((species == Species::W) ? 1 : 0);
    for (Int k = 0; k < nk; ++k)
      for (Int j = 0; j < nj; ++j)
        for (Int i = 0; i < ni; ++i) f(i, j, k) = in[n++];
  }
}

std::string State::describe() const {
  std::ostringstream os;
  os << "状态 (t=" << time << " s, step=" << step << ")\n";
  for (int sp = 0; sp < kNumSpecies; ++sp) {
    const auto& f = fields_[static_cast<Size>(sp)];
    const auto st = f.stats();
    os << "  " << species_name(static_cast<Species>(sp)) << ": min=" << st.min
       << " max=" << st.max << " mean=" << st.mean << " rms=" << st.rms << "\n";
  }
  if (has_nonfinite()) os << "  ** 警告：存在非有限值 **\n";
  return os.str();
}

}  // namespace vibe::dyn
