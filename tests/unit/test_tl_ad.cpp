/// @file test_tl_ad.cpp
/// @brief 切线性/伴随模式、轨迹存储与点积/切线检验的单元测试。
///
/// 覆盖：点积检验（< 1e-8）、TL 线性性、AD 线性性、单步伴随性、
/// TL 与有限差分一致的 U 形误差曲线、check_tangent 返回最优 eps、
/// Trajectory 的内存统计与状态存取、线性增长率与描述字符串。
///
/// 测试里实现的非线性模式与 src/da/tangent_linear.cpp 的 TL 核使用**同一套**
/// 离散算子（中心差分、内部点、冻结全量基础态系数），因此其 Fréchet 导数
/// 严格等于该 TL 核 —— 这正是 U 形误差曲线能够成立的前提。
///
/// 入口由 tests/unit/test_main.cpp（目标 vibe_test_main）统一提供。

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "vibe/common/constants.hpp"
#include "vibe/common/test.hpp"
#include "vibe/config/config.hpp"
#include "vibe/da/tangent_linear.hpp"
#include "vibe/dyn/reference_state.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/dyn/tendency.hpp"
#include "vibe/grid/decomposition.hpp"
#include "vibe/grid/field.hpp"
#include "vibe/grid/geometry.hpp"

using vibe::Int;
using vibe::Real;
using vibe::Size;
using vibe::da::AdjointCheckResult;
using vibe::da::AdjointModel;
using vibe::da::TangentCheckResult;
using vibe::da::TangentLinearModel;

namespace {

using vibe::grid::Field;
using vibe::grid::Grid;

const Real kCoriolisTest = Real(2) * vibe::kOmega * std::sin(vibe::kReferenceLat * vibe::kDegToRad);

grid::Grid make_grid(Int nx = 8, Int ny = 6, Int nz = 5) {
  grid::Geometry geom;
  geom.nx = nx;
  geom.ny = ny;
  geom.nz = nz;
  geom.dx = Real(1000);
  geom.dy = Real(1000);
  geom.z_top = Real(10000);
  geom.zeta = grid::Geometry::default_zeta_levels(nz, Real(1.0));
  const grid::Decomposition dec = grid::Decomposition::make(nx, ny, nz, 1, 1, 4);
  return grid::Grid(geom, dec);
}

// ---- 与 TL 核一致的离散算子（供测试用非线性模式使用）---------------------

void dop_x(const Field<Real>& f, const Field<Real>& c, const Grid& g, Field<Real>& out,
           Real scale) {
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  for (Int k = 0; k < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 1; i + 1 < nx; ++i) {
        const Real inv = Real(0.5) / std::max(g.dx_at(i), Real(1e-6));
        out.at(i, j, k) +=
            scale * c.at(i, j, k) * (f.at(i + 1, j, k) - f.at(i - 1, j, k)) * inv;
      }
}

void dop_y(const Field<Real>& f, const Field<Real>& c, const Grid& g, Field<Real>& out,
           Real scale) {
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  for (Int k = 0; k < nz; ++k)
    for (Int j = 1; j + 1 < ny; ++j) {
      const Real inv = Real(0.5) / std::max(g.dy_at(j), Real(1e-6));
      for (Int i = 0; i < nx; ++i) {
        out.at(i, j, k) +=
            scale * c.at(i, j, k) * (f.at(i, j + 1, k) - f.at(i, j - 1, k)) * inv;
      }
    }
}

void dop_z(const Field<Real>& f, const Field<Real>& c, const Grid& g, Field<Real>& out,
           Real scale) {
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  for (Int k = 1; k + 1 < nz; ++k)
    for (Int j = 0; j < ny; ++j)
      for (Int i = 0; i < nx; ++i) {
        const Real inv = Real(0.5) * g.inv_dz(i, j, k);
        out.at(i, j, k) +=
            scale * c.at(i, j, k) * (f.at(i, j, k + 1) - f.at(i, j, k - 1)) * inv;
      }
}

/// 非线性趋势（其 Fréchet 导数恰为 src/da/tangent_linear.cpp 的 tendency_tl）
void nl_tendency(const dyn::State& x, const dyn::State& xb, const dyn::ReferenceState& ref,
                 const Grid& g, dyn::Tendency& d) {
  d.zero();
  const Field<Real>& u = x.u();
  const Field<Real>& v = x.v();
  const Field<Real>& w = x.w();
  VIBE_UNUSED(ref);
  // State 存全量场（见 dyn::Equations::diagnose / initial_conditions）
  const Field<Real>& rho = x.rho();
  const Field<Real>& th = x.theta();
  const Field<Real>& pi = x.pi();
  const Field<Real>& th_b = xb.theta();
  const Real rdcv = vibe::kRd / vibe::kCv;
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();

  const dyn::Species scalars[] = {dyn::Species::Theta, dyn::Species::Qv, dyn::Species::Qc,
                                  dyn::Species::Qr,    dyn::Species::Qi, dyn::Species::Qs,
                                  dyn::Species::Qg};
  for (auto sp : scalars) {
    const Field<Real>& q = (sp == dyn::Species::Theta) ? th : x.field(sp);
    Field<Real>& out = d.field(sp);
    dop_x(q, u, g, out, Real(-1));
    dop_y(q, v, g, out, Real(-1));
    dop_z(q, w, g, out, Real(-1));
  }
  {
    Field<Real>& out = d.rho();
    dop_x(u, rho, g, out, Real(-1));
    dop_y(v, rho, g, out, Real(-1));
    dop_z(w, rho, g, out, Real(-1));
    dop_x(rho, u, g, out, Real(-1));
    dop_y(rho, v, g, out, Real(-1));
    dop_z(rho, w, g, out, Real(-1));
  }
  {
    Field<Real>& out = d.pi();
    dop_x(pi, u, g, out, Real(-1));
    dop_y(pi, v, g, out, Real(-1));
    dop_z(pi, w, g, out, Real(-1));
    dop_x(u, pi, g, out, -rdcv);
    dop_y(v, pi, g, out, -rdcv);
    dop_z(w, pi, g, out, -rdcv);
  }
  {
    Field<Real>& out = d.u();
    dop_x(u, u, g, out, Real(-1));
    dop_y(u, v, g, out, Real(-1));
    dop_z(u, w, g, out, Real(-1));
    dop_x(pi, th, g, out, -vibe::kCp);
    for (Int k = 0; k < nz; ++k)
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i) out.at(i, j, k) += kCoriolisTest * v.at(i, j, k);
  }
  {
    Field<Real>& out = d.v();
    dop_x(v, u, g, out, Real(-1));
    dop_y(v, v, g, out, Real(-1));
    dop_z(v, w, g, out, Real(-1));
    dop_y(pi, th, g, out, -vibe::kCp);
    for (Int k = 0; k < nz; ++k)
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i) out.at(i, j, k) -= kCoriolisTest * u.at(i, j, k);
  }
  {
    Field<Real>& out = d.w();
    dop_x(w, u, g, out, Real(-1));
    dop_y(w, v, g, out, Real(-1));
    dop_z(w, w, g, out, Real(-1));
    dop_z(pi, th, g, out, -vibe::kCp);
    for (Int k = 0; k < nz; ++k)
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i) {
          const Real tb = std::max(th_b.at(i, j, k), Real(1));
          out.at(i, j, k) += vibe::kGravity * (th.at(i, j, k) / tb - Real(1));
        }
  }
}

void state_add_tendency(dyn::State& s, Real a, const dyn::Tendency& t) {
  for (int sp = 0; sp < dyn::kNumSpecies; ++sp) {
    s.field(static_cast<dyn::Species>(sp))
        .add_scaled(a, t.field(static_cast<dyn::Species>(sp)));
  }
}

/// 非线性 RK3（与 TL 核同一离散格式）
void nl_rk3(const dyn::State& x0, dyn::State& xf, Real dt, int n_steps,
            const dyn::State& xb, const dyn::ReferenceState& ref, const Grid& g) {
  dyn::State x = x0;
  for (int n = 0; n < n_steps; ++n) {
    dyn::Tendency d1(g), d2(g), d3(g);
    nl_tendency(x, xb, ref, g, d1);
    dyn::State x1 = x;
    state_add_tendency(x1, dt / Real(3), d1);
    nl_tendency(x1, xb, ref, g, d2);
    dyn::State x2 = x;
    state_add_tendency(x2, dt / Real(2), d2);
    nl_tendency(x2, xb, ref, g, d3);
    state_add_tendency(x, dt, d3);
  }
  xf = x;
}

void fill_base(dyn::State& s, const dyn::ReferenceState& ref) {
  const Grid& g = s.grid();
  s.set_reference(&ref);
  for (Int k = 0; k < g.nz(); ++k) {
    for (Int j = 0; j < g.ny(); ++j) {
      for (Int i = 0; i < g.nx(); ++i) {
        const Real x = Real(i), y = Real(j);
        s.u().at(i, j, k) = Real(5) + Real(0.5) * std::sin(Real(0.4) * x + Real(0.2) * y);
        s.v().at(i, j, k) = Real(2) + Real(0.3) * std::cos(Real(0.3) * x - Real(0.5) * y);
        s.w().at(i, j, k) = Real(0.1) * std::sin(Real(0.2) * x);
        // State 存全量场：参考态 + 小扰动
        s.theta().at(i, j, k) = ref.theta0().at(i, j, k) +
                                Real(1.5) * std::sin(Real(0.35) * x + Real(0.15) * y);
        s.pi().at(i, j, k) = ref.pi0().at(i, j, k) + Real(1e-4) * std::cos(Real(0.25) * y);
        s.qv().at(i, j, k) = Real(0.005) + Real(1e-4) * std::sin(Real(0.5) * x);
        s.rho().at(i, j, k) = ref.rho0().at(i, j, k) +
                              Real(1e-3) * std::cos(Real(0.2) * x + Real(0.1) * y);
        s.field(dyn::Species::Qc).at(i, j, k) = Real(1e-5);
        s.field(dyn::Species::Qr).at(i, j, k) = Real(1e-5) * std::sin(Real(0.3) * x);
        s.field(dyn::Species::Qi).at(i, j, k) = Real(0);
        s.field(dyn::Species::Qs).at(i, j, k) = Real(0);
        s.field(dyn::Species::Qg).at(i, j, k) = Real(0);
      }
    }
  }
}

void fill_random(dyn::State& s, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<Real> nd(Real(0), Real(1));
  for (int sp = 0; sp < dyn::kNumSpecies; ++sp) {
    auto& f = s.field(static_cast<dyn::Species>(sp));
    for (Int k = 0; k < f.nz(); ++k)
      for (Int j = 0; j < f.ny(); ++j)
        for (Int i = 0; i < f.nx(); ++i) f.at(i, j, k) = nd(rng);
  }
}

}  // namespace

// ===========================================================================
// 1. 点积检验通过（相对误差 < 1e-8）
// ===========================================================================
VIBE_TEST(tlad_adjoint_dot_product) {
  const grid::Grid g = make_grid();
  const dyn::ReferenceState ref = dyn::ReferenceState::isothermal(g, Real(300), Real(100000));
  dyn::State xb(g);
  fill_base(xb, ref);
  config::ModelConfig mc;
  config::DaConfig dc;
  TangentLinearModel tl(g, ref, mc, dc);
  AdjointModel ad(g, ref, mc, dc);
  const AdjointCheckResult r = vibe::da::check_adjoint(tl, ad, xb, Real(30), 3, 1u, Real(1e-8));
  VIBE_CHECK(r.passed);
  VIBE_CHECK(r.relative_error < Real(1e-8));
  VIBE_CHECK(!r.describe().empty());
}

// ===========================================================================
// 2. 多个随机种子下的点积检验
// ===========================================================================
VIBE_TEST(tlad_adjoint_multiple_seeds) {
  const grid::Grid g = make_grid();
  const dyn::ReferenceState ref = dyn::ReferenceState::isothermal(g, Real(290), Real(98000));
  dyn::State xb(g);
  fill_base(xb, ref);
  config::ModelConfig mc;
  config::DaConfig dc;
  TangentLinearModel tl(g, ref, mc, dc);
  AdjointModel ad(g, ref, mc, dc);
  for (unsigned seed : {1u, 2u, 17u, 99u}) {
    const auto r = vibe::da::check_adjoint(tl, ad, xb, Real(20), 2, seed, Real(1e-8));
    VIBE_CHECK(r.passed);
  }
}

// ===========================================================================
// 3. TL 的线性性
// ===========================================================================
VIBE_TEST(tlad_tangent_linearity) {
  const grid::Grid g = make_grid();
  const dyn::ReferenceState ref = dyn::ReferenceState::isothermal(g, Real(300), Real(100000));
  dyn::State xb(g);
  fill_base(xb, ref);
  config::ModelConfig mc;
  config::DaConfig dc;
  TangentLinearModel tl(g, ref, mc, dc);

  dyn::State d1(g), d2(g), d3(g);
  fill_random(d1, 5u);
  fill_random(d2, 6u);
  const Real a = Real(1.3), b = Real(-0.7);
  d3.axpy(a, d1, Real(0));
  for (int sp = 0; sp < dyn::kNumSpecies; ++sp) {
    d3.field(static_cast<dyn::Species>(sp))
        .add_scaled(b, d2.field(static_cast<dyn::Species>(sp)));
  }
  dyn::State o1(g), o2(g), o3(g);
  tl.propagate(xb, d1, Real(20), 2, o1);
  tl.propagate(xb, d2, Real(20), 2, o2);
  tl.propagate(xb, d3, Real(20), 2, o3);
  Real maxdiff = Real(0);
  for (int sp = 0; sp < dyn::kNumSpecies; ++sp) {
    const auto s = static_cast<dyn::Species>(sp);
    const auto& f3 = o3.field(s);
    const auto& f1 = o1.field(s);
    const auto& f2 = o2.field(s);
    for (Int k = 0; k < f3.nz(); ++k)
      for (Int j = 0; j < f3.ny(); ++j)
        for (Int i = 0; i < f3.nx(); ++i) {
          maxdiff = std::max(maxdiff,
                             std::abs(f3.at(i, j, k) - a * f1.at(i, j, k) -
                                      b * f2.at(i, j, k)));
        }
  }
  VIBE_CHECK(maxdiff < Real(1e-11));
}

// ===========================================================================
// 4. AD 的线性性
// ===========================================================================
VIBE_TEST(tlad_adjoint_linearity) {
  const grid::Grid g = make_grid();
  const dyn::ReferenceState ref = dyn::ReferenceState::isothermal(g, Real(300), Real(100000));
  dyn::State xb(g);
  fill_base(xb, ref);
  config::ModelConfig mc;
  config::DaConfig dc;
  TangentLinearModel tl(g, ref, mc, dc);
  AdjointModel ad(g, ref, mc, dc);

  dyn::State dummy(g);
  fill_random(dummy, 3u);
  dyn::State sink(g);
  tl.propagate(xb, dummy, Real(20), 2, sink);  // 记录轨迹

  dyn::State y1(g), y2(g), y3(g);
  fill_random(y1, 7u);
  fill_random(y2, 8u);
  const Real a = Real(-0.4), b = Real(2.1);
  y3.axpy(a, y1, Real(0));
  for (int sp = 0; sp < dyn::kNumSpecies; ++sp) {
    y3.field(static_cast<dyn::Species>(sp))
        .add_scaled(b, y2.field(static_cast<dyn::Species>(sp)));
  }
  dyn::State a1(g), a2(g), a3(g);
  ad.propagate_with_trajectory(tl.trajectory(), y1, a1);
  ad.propagate_with_trajectory(tl.trajectory(), y2, a2);
  ad.propagate_with_trajectory(tl.trajectory(), y3, a3);
  Real maxdiff = Real(0);
  for (int sp = 0; sp < dyn::kNumSpecies; ++sp) {
    const auto s = static_cast<dyn::Species>(sp);
    const auto& f3 = a3.field(s);
    const auto& f1 = a1.field(s);
    const auto& f2 = a2.field(s);
    for (Int k = 0; k < f3.nz(); ++k)
      for (Int j = 0; j < f3.ny(); ++j)
        for (Int i = 0; i < f3.nx(); ++i) {
          maxdiff = std::max(maxdiff,
                             std::abs(f3.at(i, j, k) - a * f1.at(i, j, k) -
                                      b * f2.at(i, j, k)));
        }
  }
  VIBE_CHECK(maxdiff < Real(1e-11));
}

// ===========================================================================
// 5. 单步 tl.step 与 ad.step_adjoint 互补
// ===========================================================================
VIBE_TEST(tlad_single_step_adjoint) {
  const grid::Grid g = make_grid();
  const dyn::ReferenceState ref = dyn::ReferenceState::isothermal(g, Real(300), Real(100000));
  dyn::State xb(g);
  fill_base(xb, ref);
  config::ModelConfig mc;
  config::DaConfig dc;
  TangentLinearModel tl(g, ref, mc, dc);
  AdjointModel ad(g, ref, mc, dc);

  dyn::State dx(g), dy(g);
  fill_random(dx, 21u);
  fill_random(dy, 22u);
  dyn::State mdx(g), mtdy(g);
  tl.step(xb, dx, Real(25), mdx);
  ad.step_adjoint(xb, dy, Real(25), mtdy);
  const Real lhs = mdx.dot(dy, Real(0));
  const Real rhs = dx.dot(mtdy, Real(0));
  VIBE_CHECK(vibe::rel_error(lhs, rhs) < Real(1e-8));
}

// ===========================================================================
// 6. propagate 与 step 一致
// ===========================================================================
VIBE_TEST(tlad_step_matches_propagate) {
  const grid::Grid g = make_grid();
  const dyn::ReferenceState ref = dyn::ReferenceState::isothermal(g, Real(300), Real(100000));
  dyn::State xb(g);
  fill_base(xb, ref);
  config::ModelConfig mc;
  config::DaConfig dc;
  TangentLinearModel tl(g, ref, mc, dc);
  dyn::State dx(g);
  fill_random(dx, 31u);
  dyn::State a(g), b(g);
  tl.step(xb, dx, Real(18), a);
  tl.propagate(xb, dx, Real(18), 1, b);
  Real maxdiff = Real(0);
  for (int sp = 0; sp < dyn::kNumSpecies; ++sp) {
    const auto s = static_cast<dyn::Species>(sp);
    const auto& fa = a.field(s);
    const auto& fb = b.field(s);
    for (Int k = 0; k < fa.nz(); ++k)
      for (Int j = 0; j < fa.ny(); ++j)
        for (Int i = 0; i < fa.nx(); ++i) {
          maxdiff = std::max(maxdiff, std::abs(fa.at(i, j, k) - fb.at(i, j, k)));
        }
  }
  VIBE_CHECK(maxdiff < Real(1e-14));
}

// ===========================================================================
// 7. TL 与有限差分一致性：U 形误差曲线
// ===========================================================================
VIBE_TEST(tlad_tangent_finite_difference_u_shape) {
  const grid::Grid g = make_grid();
  const dyn::ReferenceState ref = dyn::ReferenceState::isothermal(g, Real(300), Real(100000));
  dyn::State xb(g);
  fill_base(xb, ref);
  config::ModelConfig mc;
  config::DaConfig dc;
  TangentLinearModel tl(g, ref, mc, dc);
  dyn::State dx(g);
  fill_random(dx, 41u);

  const auto model = [&](const dyn::State& x0, dyn::State& xf, Real dt, int n) {
    nl_rk3(x0, xf, dt, n, xb, ref, g);
  };
  const std::vector<Real> eps = {Real(1e-1), Real(1e-2), Real(1e-3), Real(1e-4),
                                 Real(1e-5), Real(1e-6), Real(1e-7), Real(1e-8),
                                 Real(1e-9), Real(1e-10)};
  std::vector<Real> errs;
  for (Real e : eps) {
    const auto r = vibe::da::check_tangent(model, tl, xb, dx, Real(30), 2, {e});
    errs.push_back(r.relative_error);
  }
  // 找内部最小值：截断误差 O(eps) 主导大 eps，舍入 O(1/eps) 主导小 eps
  Size argmin = 0;
  for (Size i = 1; i < errs.size(); ++i) {
    if (errs[i] < errs[argmin]) argmin = i;
  }
  VIBE_CHECK(argmin > 0);
  VIBE_CHECK(argmin + 1 < errs.size());
  VIBE_CHECK(errs.front() > errs[argmin]);
  VIBE_CHECK(errs.back() > errs[argmin]);
  // 最优 eps 附近的相对误差应很小
  VIBE_CHECK(errs[argmin] < Real(1e-5));
}

// ===========================================================================
// 8. check_tangent 返回最优 eps 与描述
// ===========================================================================
VIBE_TEST(tlad_check_tangent_optimal_eps) {
  const grid::Grid g = make_grid();
  const dyn::ReferenceState ref = dyn::ReferenceState::isothermal(g, Real(300), Real(100000));
  dyn::State xb(g);
  fill_base(xb, ref);
  config::ModelConfig mc;
  config::DaConfig dc;
  TangentLinearModel tl(g, ref, mc, dc);
  dyn::State dx(g);
  fill_random(dx, 51u);
  const auto model = [&](const dyn::State& x0, dyn::State& xf, Real dt, int n) {
    nl_rk3(x0, xf, dt, n, xb, ref, g);
  };
  const std::vector<Real> eps = {Real(1e-1), Real(1e-3), Real(1e-5), Real(1e-7), Real(1e-9)};
  const TangentCheckResult r =
      vibe::da::check_tangent(model, tl, xb, dx, Real(30), 2, eps);
  VIBE_CHECK(r.optimal_eps > Real(0));
  VIBE_CHECK(r.tl_norm > Real(0));
  VIBE_CHECK(r.fd_norm > Real(0));
  VIBE_CHECK(!r.describe().empty());
  // 最优 eps 必须是列表中的某一个
  bool found = false;
  for (Real e : eps) {
    if (std::abs(e - r.optimal_eps) <= Real(1e-16) * e) found = true;
  }
  VIBE_CHECK(found);
}

// ===========================================================================
// 9. Trajectory 的存取与内存统计
// ===========================================================================
VIBE_TEST(tlad_trajectory_storage_and_memory) {
  const grid::Grid g = make_grid();
  const dyn::ReferenceState ref = dyn::ReferenceState::isothermal(g, Real(300), Real(100000));
  dyn::State xb(g);
  fill_base(xb, ref);
  config::ModelConfig mc;
  config::DaConfig dc;
  TangentLinearModel tl(g, ref, mc, dc);
  dyn::State dx(g);
  fill_random(dx, 61u);
  dyn::State out(g);
  tl.propagate(xb, dx, Real(15), 4, out);
  const auto& traj = tl.trajectory();
  VIBE_CHECK(traj.size() == 4);
  VIBE_CHECK(traj.memory_bytes() > 0);
  // 存储的基础态时间按 dt 递增
  VIBE_CHECK_NEAR(traj.state(1).time - traj.state(0).time, Real(15), Real(1e-12));
  VIBE_CHECK_NEAR(traj.state(3).time - traj.state(0).time, Real(45), Real(1e-12));
  // 状态与趋势都可访问且布局正确
  VIBE_CHECK(traj.state(0).u().nx() == g.nx());
  VIBE_CHECK(traj.slow(0).u().ny() == g.ny());
  VIBE_CHECK(traj.acoustic(0).u().nz() == g.nz());
}

// ===========================================================================
// 10. record_trajectory 的记录长度与清空
// ===========================================================================
VIBE_TEST(tlad_record_trajectory) {
  const grid::Grid g = make_grid();
  const dyn::ReferenceState ref = dyn::ReferenceState::isothermal(g, Real(300), Real(100000));
  dyn::State xb(g);
  fill_base(xb, ref);
  config::ModelConfig mc;
  config::DaConfig dc;
  TangentLinearModel tl(g, ref, mc, dc);
  tl.record_trajectory(xb, Real(10), 6);
  VIBE_CHECK(tl.trajectory().size() == 6);
  VIBE_CHECK(tl.trajectory().memory_bytes() > 0);
  tl.trajectory().clear();
  VIBE_CHECK(tl.trajectory().size() == 0);
  VIBE_CHECK(tl.trajectory().memory_bytes() == 0);
}

// ===========================================================================
// 11. 线性增长率：零扰动为零，非零扰动有限
// ===========================================================================
VIBE_TEST(tlad_linear_growth_rate) {
  const grid::Grid g = make_grid();
  const dyn::ReferenceState ref = dyn::ReferenceState::isothermal(g, Real(300), Real(100000));
  dyn::State xb(g);
  fill_base(xb, ref);
  config::ModelConfig mc;
  config::DaConfig dc;
  TangentLinearModel tl(g, ref, mc, dc);
  dyn::State zero(g);
  VIBE_CHECK_NEAR(tl.linear_growth_rate(xb, zero, Real(10), 3), Real(0), Real(1e-15));
  dyn::State dx(g);
  fill_random(dx, 71u);
  const Real rate = tl.linear_growth_rate(xb, dx, Real(10), 3);
  VIBE_CHECK(std::isfinite(rate));
}

// ===========================================================================
// 12. 物理线性化开关与描述字符串
// ===========================================================================
VIBE_TEST(tlad_physics_switch_and_strings) {
  VIBE_CHECK(std::string(vibe::da::to_string(vibe::da::PhysicsLinearization::Adiabatic)) ==
             "adiabatic");
  VIBE_CHECK(std::string(vibe::da::to_string(vibe::da::PhysicsLinearization::FrozenSwitch)) ==
             "frozen_switch");
  const grid::Grid g = make_grid();
  const dyn::ReferenceState ref = dyn::ReferenceState::isothermal(g, Real(300), Real(100000));
  config::ModelConfig mc;
  config::DaConfig dc;
  TangentLinearModel tl(g, ref, mc, dc);
  tl.set_physics_linearization(vibe::da::PhysicsLinearization::Simplified);
  VIBE_CHECK(tl.physics_linearization() == vibe::da::PhysicsLinearization::Simplified);
  tl.set_checkpointing(vibe::da::CheckpointStrategy::Revolve);
  AdjointCheckResult r;
  r.lhs = Real(1);
  r.rhs = Real(1) + Real(1e-12);
  r.relative_error = Real(5e-13);
  r.passed = true;
  VIBE_CHECK(r.describe().find("AdjointCheck") != std::string::npos);
  TangentCheckResult t;
  t.relative_error = Real(1e-6);
  t.optimal_eps = Real(1e-5);
  t.passed = true;
  VIBE_CHECK(t.describe().find("TangentCheck") != std::string::npos);
}
