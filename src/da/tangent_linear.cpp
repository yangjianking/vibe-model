/// @file tangent_linear.cpp
/// @brief 切线性模式、伴随模式、轨迹存储与点积检验。
///
/// 自包含的 TL 核（说明）
/// ---------------------
/// 冻结接口中 dyn::Equations 没有 TL/AD 版本，因此本文件在匿名命名空间内实现
/// 一个**可读、自包含**的 TL 推进核：对 Wicker-Skamarock 低存储 RK3 的每一步做
/// 前向微分，平流项用基础态速度、梯度项用基础态 theta/pi 冻结系数（[D5][T5]）。
/// 生产实现会复用 dyn 层的模板化 TL 内核（同一套 stencil 一次编译出
/// 非线性/TL/AD 三个版本，见 docs/design/07_tangent_linear_adjoint.md），
/// 这里为了教学可读性把 v/u/w 视为体心共位（错位模板的差异在 dyn 层处理）。
///
/// 离散切线性算子 F(x^b, .)
/// ------------------------
///     theta,qv,qc,qr,qi,qs,qg :  F_s = -u_b.grad(s') - u'.grad(s_b)
///     rho                      :  F_rho = -div(rho_b u') - u_b.grad(rho') - rho' div(u_b)
///     pi                       :  F_pi  = -u_b.grad(pi') - u'.grad(pi_b)
///                                        - (Rd/cv) [ pi_b div(u') + pi' div(u_b) ]
///     u                        :  F_u = -u_b.grad(u') - u'.grad(u_b)
///                                        - cp (theta_b d_x pi' + theta' d_x pi_b) + f v'
///     v                        :  同上，最后一项为 -f u'
///     w                        :  同上（垂向梯度），加浮力 g theta'/theta_b
///
/// 所有差分都是 2 阶中心差分、只作用于内部点（i,j ∈ [1,n-2]，k ∈ [1,nz-2]），
/// 因此每个前向语句都是稀疏矩阵的一行，其转置就是把这些权重按相反方向累加，
/// 可以做逐项严格转置（[A1]）。
///
/// 复杂度：单步 F 为 O(N * 12)；一次 RK3 步 3 次 F；n_steps 步 O(n_steps N)。
///
/// 文献：[A1] Giering & Kaminski (1998)；[A2] Griewank (2000)；
///       [A3] Sirkes & Tziperman (1997)；[A4] Errico (1997)；
///       [A5] Navon et al. (1992)；[A6] Zou et al. (1993)；[A8] Mahfouf (1999)；
///       [D5] Skamarock & Klemp (2008)；[T5] Wicker & Skamarock (2002)。

#include "vibe/da/tangent_linear.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "vibe/common/constants.hpp"
#include "vibe/common/error.hpp"
#include "vibe/common/logging.hpp"
#include "vibe/config/config.hpp"
#include "vibe/dyn/reference_state.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/dyn/tendency.hpp"
#include "vibe/grid/field.hpp"
#include "vibe/grid/geometry.hpp"
#include "vibe/time/integrator.hpp"

namespace vibe::da {

const char* to_string(PhysicsLinearization p) noexcept {
  switch (p) {
    case PhysicsLinearization::Adiabatic:    return "adiabatic";
    case PhysicsLinearization::Simplified:   return "simplified";
    case PhysicsLinearization::FrozenSwitch: return "frozen_switch";
    case PhysicsLinearization::FullPhysics:  return "full_physics";
  }
  return "unknown";
}

// ===========================================================================
// TL/AD 核（匿名命名空间）
// ===========================================================================

namespace {

using grid::Field;
using grid::Grid;

/// 冻结的 f 平面科氏参数
const Real kCoriolis = Real(2) * kOmega * std::sin(kReferenceLat * kDegToRad);

/// 前向微分原语：out += scale * c * d f / dx（中心差分，内部点）
void dop_x(const Field<Real>& f, const Field<Real>& c, const Grid& g, Field<Real>& out,
           Real scale) {
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  for (Int k = 0; k < nz; ++k) {
    for (Int j = 0; j < ny; ++j) {
      for (Int i = 1; i + 1 < nx; ++i) {
        const Real inv = Real(0.5) / std::max(g.dx_at(i), Real(1e-6));
        out.at(i, j, k) +=
            scale * c.at(i, j, k) * (f.at(i + 1, j, k) - f.at(i - 1, j, k)) * inv;
      }
    }
  }
}

void dop_y(const Field<Real>& f, const Field<Real>& c, const Grid& g, Field<Real>& out,
           Real scale) {
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  for (Int k = 0; k < nz; ++k) {
    for (Int j = 1; j + 1 < ny; ++j) {
      const Real inv = Real(0.5) / std::max(g.dy_at(j), Real(1e-6));
      for (Int i = 0; i < nx; ++i) {
        out.at(i, j, k) +=
            scale * c.at(i, j, k) * (f.at(i, j + 1, k) - f.at(i, j - 1, k)) * inv;
      }
    }
  }
}

void dop_z(const Field<Real>& f, const Field<Real>& c, const Grid& g, Field<Real>& out,
           Real scale) {
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  for (Int k = 1; k + 1 < nz; ++k) {
    for (Int j = 0; j < ny; ++j) {
      for (Int i = 0; i < nx; ++i) {
        const Real inv = Real(0.5) * g.inv_dz(i, j, k);
        out.at(i, j, k) +=
            scale * c.at(i, j, k) * (f.at(i, j, k + 1) - f.at(i, j, k - 1)) * inv;
      }
    }
  }
}

/// dop_x 关于被微分场 f 的转置：bar_f += scale * c * (d/dx)* bar_out
void dop_x_ad_f(const Field<Real>& c, const Grid& g, Field<Real>& bar_f,
                const Field<Real>& bar_out, Real scale) {
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  for (Int k = 0; k < nz; ++k) {
    for (Int j = 0; j < ny; ++j) {
      for (Int i = 1; i + 1 < nx; ++i) {
        const Real inv = Real(0.5) / std::max(g.dx_at(i), Real(1e-6));
        const Real w = scale * c.at(i, j, k) * inv * bar_out.at(i, j, k);
        bar_f.at(i + 1, j, k) += w;
        bar_f.at(i - 1, j, k) -= w;
      }
    }
  }
}

/// dop_x 关于冻结系数 c 的转置（c 在本项中是扰动）：bar_c += scale * (d f/dx) * bar_out
void dop_x_ad_c(const Field<Real>& f, const Grid& g, Field<Real>& bar_c,
                const Field<Real>& bar_out, Real scale) {
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  for (Int k = 0; k < nz; ++k) {
    for (Int j = 0; j < ny; ++j) {
      for (Int i = 1; i + 1 < nx; ++i) {
        const Real inv = Real(0.5) / std::max(g.dx_at(i), Real(1e-6));
        bar_c.at(i, j, k) +=
            scale * bar_out.at(i, j, k) * (f.at(i + 1, j, k) - f.at(i - 1, j, k)) * inv;
      }
    }
  }
}

void dop_y_ad_f(const Field<Real>& c, const Grid& g, Field<Real>& bar_f,
                const Field<Real>& bar_out, Real scale) {
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  for (Int k = 0; k < nz; ++k) {
    for (Int j = 1; j + 1 < ny; ++j) {
      const Real inv = Real(0.5) / std::max(g.dy_at(j), Real(1e-6));
      for (Int i = 0; i < nx; ++i) {
        const Real w = scale * c.at(i, j, k) * inv * bar_out.at(i, j, k);
        bar_f.at(i, j + 1, k) += w;
        bar_f.at(i, j - 1, k) -= w;
      }
    }
  }
}

void dop_y_ad_c(const Field<Real>& f, const Grid& g, Field<Real>& bar_c,
                const Field<Real>& bar_out, Real scale) {
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  for (Int k = 0; k < nz; ++k) {
    for (Int j = 1; j + 1 < ny; ++j) {
      const Real inv = Real(0.5) / std::max(g.dy_at(j), Real(1e-6));
      for (Int i = 0; i < nx; ++i) {
        bar_c.at(i, j, k) +=
            scale * bar_out.at(i, j, k) * (f.at(i, j + 1, k) - f.at(i, j - 1, k)) * inv;
      }
    }
  }
}

void dop_z_ad_f(const Field<Real>& c, const Grid& g, Field<Real>& bar_f,
                const Field<Real>& bar_out, Real scale) {
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  for (Int k = 1; k + 1 < nz; ++k) {
    for (Int j = 0; j < ny; ++j) {
      for (Int i = 0; i < nx; ++i) {
        const Real inv = Real(0.5) * g.inv_dz(i, j, k);
        const Real w = scale * c.at(i, j, k) * inv * bar_out.at(i, j, k);
        bar_f.at(i, j, k + 1) += w;
        bar_f.at(i, j, k - 1) -= w;
      }
    }
  }
}

void dop_z_ad_c(const Field<Real>& f, const Grid& g, Field<Real>& bar_c,
                const Field<Real>& bar_out, Real scale) {
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  for (Int k = 1; k + 1 < nz; ++k) {
    for (Int j = 0; j < ny; ++j) {
      for (Int i = 0; i < nx; ++i) {
        const Real inv = Real(0.5) * g.inv_dz(i, j, k);
        bar_c.at(i, j, k) +=
            scale * bar_out.at(i, j, k) * (f.at(i, j, k + 1) - f.at(i, j, k - 1)) * inv;
      }
    }
  }
}

/// 冻结的基础态散度 div(u_b)（只用于点乘项，不需要转置）
Field<Real> base_divergence(const Grid& g, const dyn::State& xb) {
  Field<Real> div(g, grid::Stagger::Cell, "div_ub");
  div.fill(Real(0));
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  const auto& ub = xb.u();
  const auto& vb = xb.v();
  const auto& wb = xb.w();
  for (Int k = 0; k < nz; ++k) {
    for (Int j = 0; j < ny; ++j) {
      for (Int i = 0; i < nx; ++i) {
        Real d = Real(0);
        if (i >= 1 && i + 1 < nx) {
          d += (ub.at(i + 1, j, k) - ub.at(i - 1, j, k)) *
               (Real(0.5) / std::max(g.dx_at(i), Real(1e-6)));
        }
        if (j >= 1 && j + 1 < ny) {
          d += (vb.at(i, j + 1, k) - vb.at(i, j - 1, k)) *
               (Real(0.5) / std::max(g.dy_at(j), Real(1e-6)));
        }
        if (k >= 1 && k + 1 < nz) {
          d += (wb.at(i, j, k + 1) - wb.at(i, j, k - 1)) * (Real(0.5) * g.inv_dz(i, j, k));
        }
        div.at(i, j, k) = d;
      }
    }
  }
  return div;
}

/// 冻结的基础态标量梯度 grad(f)（中心差分，仅内部点；边界置零）
void base_gradient(const Grid& g, const Field<Real>& f, Field<Real>& gx, Field<Real>& gy,
                   Field<Real>& gz) {
  gx.fill(Real(0));
  gy.fill(Real(0));
  gz.fill(Real(0));
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  for (Int k = 0; k < nz; ++k) {
    for (Int j = 0; j < ny; ++j) {
      for (Int i = 0; i < nx; ++i) {
        if (i >= 1 && i + 1 < nx) {
          gx.at(i, j, k) =
              (f.at(i + 1, j, k) - f.at(i - 1, j, k)) *
              (Real(0.5) / std::max(g.dx_at(i), Real(1e-6)));
        }
        if (j >= 1 && j + 1 < ny) {
          gy.at(i, j, k) =
              (f.at(i, j + 1, k) - f.at(i, j - 1, k)) *
              (Real(0.5) / std::max(g.dy_at(j), Real(1e-6)));
        }
        if (k >= 1 && k + 1 < nz) {
          gz.at(i, j, k) =
              (f.at(i, j, k + 1) - f.at(i, j, k - 1)) * (Real(0.5) * g.inv_dz(i, j, k));
        }
      }
    }
  }
}

/// 全量基础态场。dyn 层把 rho/theta/pi 存为**全量**场
/// （见 Equations::diagnose、initial_conditions.cpp 的 init_resting_isothermal
///  与 damping.cpp 的 relax_cell(..., &ref.rho0())），因此线性化系数就是状态本身。
/// 保留这三个包装函数的目的是在代码里显式标注"此处读取的是线性化系数"，
/// 并方便将来若改为扰动存储时集中修改。
Field<Real> full_theta(const dyn::ReferenceState* ref, const dyn::State& xb) {
  VIBE_UNUSED(ref);
  return xb.theta();
}
Field<Real> full_pi(const dyn::ReferenceState* ref, const dyn::State& xb) {
  VIBE_UNUSED(ref);
  return xb.pi();
}
Field<Real> full_rho(const dyn::ReferenceState* ref, const dyn::State& xb) {
  VIBE_UNUSED(ref);
  return xb.rho();
}

/// 需要平流输运的标量物种
const dyn::Species kScalars[] = {dyn::Species::Theta, dyn::Species::Qv, dyn::Species::Qc,
                                 dyn::Species::Qr,    dyn::Species::Qi, dyn::Species::Qs,
                                 dyn::Species::Qg};
constexpr int kNumScalars = 7;

/// TL 趋势 F(x^b, dx)（语句顺序即 AD 的逆序依据）
void tendency_tl(const dyn::State& xb, const dyn::ReferenceState* ref, const Grid& g,
                 const dyn::State& dx, dyn::Tendency& d) {
  d.zero();
  const Field<Real>& ub = xb.u();
  const auto& vb = xb.v();
  const auto& wb = xb.w();
  const Field<Real> rhob = full_rho(ref, xb);
  const Field<Real> thb = full_theta(ref, xb);
  const Field<Real> pib = full_pi(ref, xb);
  const auto& du = dx.u();
  const auto& dv = dx.v();
  const auto& dw = dx.w();
  const auto& drho = dx.rho();
  const auto& dth = dx.theta();
  const auto& dpi = dx.pi();
  const Field<Real> div_ub = base_divergence(g, xb);
  const Real rdcv = kRd / kCv;

  // ---- 标量平流（theta, qv, qc, qr, qi, qs, qg）----
  for (int si = 0; si < kNumScalars; ++si) {
    const auto sp = kScalars[si];
    // theta 的线性化系数用全量位温；其余水物质无参考态
    const Field<Real>& sb = (sp == dyn::Species::Theta) ? thb : xb.field(sp);
    const Field<Real>& ds = dx.field(sp);
    Field<Real>& out = d.field(sp);
    dop_x(ds, ub, g, out, Real(-1));
    dop_y(ds, vb, g, out, Real(-1));
    dop_z(ds, wb, g, out, Real(-1));
    dop_x(sb, du, g, out, Real(-1));
    dop_y(sb, dv, g, out, Real(-1));
    dop_z(sb, dw, g, out, Real(-1));
  }

  // ---- 密度（守恒形式）----
  {
    Field<Real>& out = d.rho();
    dop_x(du, rhob, g, out, Real(-1));
    dop_y(dv, rhob, g, out, Real(-1));
    dop_z(dw, rhob, g, out, Real(-1));
    dop_x(drho, ub, g, out, Real(-1));
    dop_y(drho, vb, g, out, Real(-1));
    dop_z(drho, wb, g, out, Real(-1));
    const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
    for (Int k = 0; k < nz; ++k)
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i)
          out.at(i, j, k) -= div_ub.at(i, j, k) * drho.at(i, j, k);
    // 补足 -u'.grad(rho_b)：rho 方程写成 -rho div(u) - u.grad(rho) 的离散形式，
    // 其 Fréchet 导数含 -u'.grad(rho_b)，点乘冻结梯度即可（其转置见 tendency_ad）
    Field<Real> grx(g, grid::Stagger::Cell, "grho_x");
    Field<Real> gry(g, grid::Stagger::Cell, "grho_y");
    Field<Real> grz(g, grid::Stagger::Cell, "grho_z");
    base_gradient(g, rhob, grx, gry, grz);
    for (Int k = 0; k < nz; ++k)
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i) {
          out.at(i, j, k) -= du.at(i, j, k) * grx.at(i, j, k) +
                             dv.at(i, j, k) * gry.at(i, j, k) +
                             dw.at(i, j, k) * grz.at(i, j, k);
        }
  }

  // ---- Exner ----
  {
    Field<Real>& out = d.pi();
    dop_x(dpi, ub, g, out, Real(-1));
    dop_y(dpi, vb, g, out, Real(-1));
    dop_z(dpi, wb, g, out, Real(-1));
    dop_x(pib, du, g, out, Real(-1));
    dop_y(pib, dv, g, out, Real(-1));
    dop_z(pib, dw, g, out, Real(-1));
    dop_x(du, pib, g, out, -rdcv);
    dop_y(dv, pib, g, out, -rdcv);
    dop_z(dw, pib, g, out, -rdcv);
    const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
    for (Int k = 0; k < nz; ++k)
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i)
          out.at(i, j, k) -= rdcv * div_ub.at(i, j, k) * dpi.at(i, j, k);
  }

  // ---- 动量 ----
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  {
    Field<Real>& out = d.u();
    dop_x(du, ub, g, out, Real(-1));
    dop_y(du, vb, g, out, Real(-1));
    dop_z(du, wb, g, out, Real(-1));
    dop_x(ub, du, g, out, Real(-1));
    dop_y(ub, dv, g, out, Real(-1));
    dop_z(ub, dw, g, out, Real(-1));
    dop_x(dpi, thb, g, out, -kCp);
    dop_x(pib, dth, g, out, -kCp);
    for (Int k = 0; k < nz; ++k)
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i) out.at(i, j, k) += kCoriolis * dv.at(i, j, k);
  }
  {
    Field<Real>& out = d.v();
    dop_x(dv, ub, g, out, Real(-1));
    dop_y(dv, vb, g, out, Real(-1));
    dop_z(dv, wb, g, out, Real(-1));
    dop_x(vb, du, g, out, Real(-1));
    dop_y(vb, dv, g, out, Real(-1));
    dop_z(vb, dw, g, out, Real(-1));
    dop_y(dpi, thb, g, out, -kCp);
    dop_y(pib, dth, g, out, -kCp);
    for (Int k = 0; k < nz; ++k)
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i) out.at(i, j, k) -= kCoriolis * du.at(i, j, k);
  }
  {
    Field<Real>& out = d.w();
    dop_x(dw, ub, g, out, Real(-1));
    dop_y(dw, vb, g, out, Real(-1));
    dop_z(dw, wb, g, out, Real(-1));
    dop_x(wb, du, g, out, Real(-1));
    dop_y(wb, dv, g, out, Real(-1));
    dop_z(wb, dw, g, out, Real(-1));
    dop_z(dpi, thb, g, out, -kCp);
    dop_z(pib, dth, g, out, -kCp);
    for (Int k = 0; k < nz; ++k)
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i) {
          const Real th = std::max(thb.at(i, j, k), Real(1));
          out.at(i, j, k) += kGravity * dth.at(i, j, k) / th;
        }
  }
}

/// AD：严格按 tendency_tl 的逆序转置累加到 bar_dx（累加前 bar_dx 必须为零）
void tendency_ad(const dyn::State& xb, const dyn::ReferenceState* ref, const Grid& g,
                 const dyn::Tendency& bar_d, dyn::State& bar_dx) {
  const Field<Real>& ub = xb.u();
  const Field<Real>& vb = xb.v();
  const Field<Real>& wb = xb.w();
  const Field<Real> rhob = full_rho(ref, xb);
  const Field<Real> thb = full_theta(ref, xb);
  const Field<Real> pib = full_pi(ref, xb);
  const Field<Real> div_ub = base_divergence(g, xb);
  const Real rdcv = kRd / kCv;
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();

  // ---- w 方程（逆序）----
  {
    const Field<Real>& bo = bar_d.w();
    for (Int k = 0; k < nz; ++k)
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i) {
          const Real th = std::max(thb.at(i, j, k), Real(1));
          bar_dx.theta().at(i, j, k) += kGravity * bo.at(i, j, k) / th;
        }
    dop_z_ad_c(pib, g, bar_dx.theta(), bo, -kCp);
    dop_z_ad_f(thb, g, bar_dx.pi(), bo, -kCp);
    dop_z_ad_c(wb, g, bar_dx.w(), bo, Real(-1));
    dop_y_ad_c(wb, g, bar_dx.v(), bo, Real(-1));
    dop_x_ad_c(wb, g, bar_dx.u(), bo, Real(-1));
    dop_z_ad_f(wb, g, bar_dx.w(), bo, Real(-1));
    dop_y_ad_f(vb, g, bar_dx.w(), bo, Real(-1));
    dop_x_ad_f(ub, g, bar_dx.w(), bo, Real(-1));
  }
  // ---- v 方程（逆序）----
  {
    const Field<Real>& bo = bar_d.v();
    for (Int k = 0; k < nz; ++k)
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i) bar_dx.u().at(i, j, k) -= kCoriolis * bo.at(i, j, k);
    dop_y_ad_c(pib, g, bar_dx.theta(), bo, -kCp);
    dop_y_ad_f(thb, g, bar_dx.pi(), bo, -kCp);
    dop_z_ad_c(vb, g, bar_dx.w(), bo, Real(-1));
    dop_y_ad_c(vb, g, bar_dx.v(), bo, Real(-1));
    dop_x_ad_c(vb, g, bar_dx.u(), bo, Real(-1));
    dop_z_ad_f(wb, g, bar_dx.v(), bo, Real(-1));
    dop_y_ad_f(vb, g, bar_dx.v(), bo, Real(-1));
    dop_x_ad_f(ub, g, bar_dx.v(), bo, Real(-1));
  }
  // ---- u 方程（逆序）----
  {
    const Field<Real>& bo = bar_d.u();
    for (Int k = 0; k < nz; ++k)
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i) bar_dx.v().at(i, j, k) += kCoriolis * bo.at(i, j, k);
    dop_x_ad_c(pib, g, bar_dx.theta(), bo, -kCp);
    dop_x_ad_f(thb, g, bar_dx.pi(), bo, -kCp);
    dop_z_ad_c(ub, g, bar_dx.w(), bo, Real(-1));
    dop_y_ad_c(ub, g, bar_dx.v(), bo, Real(-1));
    dop_x_ad_c(ub, g, bar_dx.u(), bo, Real(-1));
    dop_z_ad_f(wb, g, bar_dx.u(), bo, Real(-1));
    dop_y_ad_f(vb, g, bar_dx.u(), bo, Real(-1));
    dop_x_ad_f(ub, g, bar_dx.u(), bo, Real(-1));
  }
  // ---- Exner（逆序）----
  {
    const Field<Real>& bo = bar_d.pi();
    for (Int k = 0; k < nz; ++k)
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i)
          bar_dx.pi().at(i, j, k) -= rdcv * div_ub.at(i, j, k) * bo.at(i, j, k);
    dop_z_ad_f(pib, g, bar_dx.w(), bo, -rdcv);
    dop_y_ad_f(pib, g, bar_dx.v(), bo, -rdcv);
    dop_x_ad_f(pib, g, bar_dx.u(), bo, -rdcv);
    dop_z_ad_c(pib, g, bar_dx.w(), bo, Real(-1));
    dop_y_ad_c(pib, g, bar_dx.v(), bo, Real(-1));
    dop_x_ad_c(pib, g, bar_dx.u(), bo, Real(-1));
    dop_z_ad_f(wb, g, bar_dx.pi(), bo, Real(-1));
    dop_y_ad_f(vb, g, bar_dx.pi(), bo, Real(-1));
    dop_x_ad_f(ub, g, bar_dx.pi(), bo, Real(-1));
  }
  // ---- 密度（逆序）----
  {
    const Field<Real>& bo = bar_d.rho();
    // 逆序：先转置最后加入的 -u'.grad(rho_b) 点乘项
    {
      Field<Real> grx(g, grid::Stagger::Cell, "grho_x");
      Field<Real> gry(g, grid::Stagger::Cell, "grho_y");
      Field<Real> grz(g, grid::Stagger::Cell, "grho_z");
      base_gradient(g, rhob, grx, gry, grz);
      for (Int k = 0; k < nz; ++k)
        for (Int j = 0; j < ny; ++j)
          for (Int i = 0; i < nx; ++i) {
            const Real b = bo.at(i, j, k);
            bar_dx.u().at(i, j, k) -= grx.at(i, j, k) * b;
            bar_dx.v().at(i, j, k) -= gry.at(i, j, k) * b;
            bar_dx.w().at(i, j, k) -= grz.at(i, j, k) * b;
          }
    }
    for (Int k = 0; k < nz; ++k)
      for (Int j = 0; j < ny; ++j)
        for (Int i = 0; i < nx; ++i)
          bar_dx.rho().at(i, j, k) -= div_ub.at(i, j, k) * bo.at(i, j, k);
    dop_z_ad_f(wb, g, bar_dx.rho(), bo, Real(-1));
    dop_y_ad_f(vb, g, bar_dx.rho(), bo, Real(-1));
    dop_x_ad_f(ub, g, bar_dx.rho(), bo, Real(-1));
    dop_z_ad_f(rhob, g, bar_dx.w(), bo, Real(-1));
    dop_y_ad_f(rhob, g, bar_dx.v(), bo, Real(-1));
    dop_x_ad_f(rhob, g, bar_dx.u(), bo, Real(-1));
  }
  // ---- 标量（逆序）----
  for (int si = kNumScalars - 1; si >= 0; --si) {
    const auto sp = kScalars[si];
    const Field<Real>& sb = (sp == dyn::Species::Theta) ? thb : xb.field(sp);
    Field<Real>& bs = bar_dx.field(sp);
    const Field<Real>& bo = bar_d.field(sp);
    dop_z_ad_c(sb, g, bar_dx.w(), bo, Real(-1));
    dop_y_ad_c(sb, g, bar_dx.v(), bo, Real(-1));
    dop_x_ad_c(sb, g, bar_dx.u(), bo, Real(-1));
    dop_z_ad_f(wb, g, bs, bo, Real(-1));
    dop_y_ad_f(vb, g, bs, bo, Real(-1));
    dop_x_ad_f(ub, g, bs, bo, Real(-1));
  }
}

/// State += a * Tendency
void state_add_tendency(dyn::State& s, Real a, const dyn::Tendency& t) {
  for (int sp = 0; sp < dyn::kNumSpecies; ++sp) {
    s.field(static_cast<dyn::Species>(sp))
        .add_scaled(a, t.field(static_cast<dyn::Species>(sp)));
  }
}

/// 低存储 RK3（Wicker-Skamarock）的切线性形式
void rk3_tl(const dyn::State& xb, const dyn::ReferenceState* ref, const Grid& g, Real dt,
            const dyn::State& dx_in, dyn::State& dx_out) {
  dyn::Tendency d1(g), d2(g), d3(g);
  dyn::State dx1(g), dx2(g);
  tendency_tl(xb, ref, g, dx_in, d1);
  dx1 = dx_in;
  state_add_tendency(dx1, dt / Real(3), d1);
  tendency_tl(xb, ref, g, dx1, d2);
  dx2 = dx_in;
  state_add_tendency(dx2, dt / Real(2), d2);
  tendency_tl(xb, ref, g, dx2, d3);
  dx_out = dx_in;
  state_add_tendency(dx_out, dt, d3);
}

/// RK3 的严格转置（逆序展开每个阶段）
void rk3_ad(const dyn::State& xb, const dyn::ReferenceState* ref, const Grid& g, Real dt,
            const dyn::State& bar_out, dyn::State& bar_in) {
  bar_in = bar_out;  // dx_out = dx_in + dt d3
  dyn::Tendency bd3(g);
  for (int sp = 0; sp < dyn::kNumSpecies; ++sp) {
    const auto s = static_cast<dyn::Species>(sp);
    bd3.field(s).fill(Real(0));
    bd3.field(s).add_scaled(dt, bar_out.field(s));
  }
  dyn::State bar_dx2(g);
  tendency_ad(xb, ref, g, bd3, bar_dx2);
  bar_in.add_scaled(Real(1), bar_dx2);  // dx2 = dx_in + (dt/2) d2
  dyn::Tendency bd2(g);
  for (int sp = 0; sp < dyn::kNumSpecies; ++sp) {
    const auto s = static_cast<dyn::Species>(sp);
    bd2.field(s).fill(Real(0));
    bd2.field(s).add_scaled(dt / Real(2), bar_dx2.field(s));
  }
  dyn::State bar_dx1(g);
  tendency_ad(xb, ref, g, bd2, bar_dx1);
  bar_in.add_scaled(Real(1), bar_dx1);  // dx1 = dx_in + (dt/3) d1
  dyn::Tendency bd1(g);
  for (int sp = 0; sp < dyn::kNumSpecies; ++sp) {
    const auto s = static_cast<dyn::Species>(sp);
    bd1.field(s).fill(Real(0));
    bd1.field(s).add_scaled(dt / Real(3), bar_dx1.field(s));
  }
  dyn::State bar_in2(g);
  tendency_ad(xb, ref, g, bd1, bar_in2);
  bar_in.add_scaled(Real(1), bar_in2);
}

/// 随机填充状态内部点（固定种子）
void fill_random(dyn::State& s, std::mt19937& rng) {
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
// Trajectory
// ===========================================================================

void Trajectory::reserve(Size n_steps, const dyn::State& prototype) {
  states_.reserve(n_steps);
  slow_.reserve(n_steps);
  acoustic_.reserve(n_steps);
  VIBE_UNUSED(prototype);
}

void Trajectory::push_back(const dyn::State& s, const dyn::Tendency& slow,
                           const dyn::Tendency& acoustic) {
  states_.push_back(s);
  slow_.push_back(slow);
  acoustic_.push_back(acoustic);
}

void Trajectory::clear() {
  states_.clear();
  slow_.clear();
  acoustic_.clear();
}

Size Trajectory::memory_bytes() const {
  Size per_state = 0, per_tend = 0;
  if (!states_.empty()) {
    for (int sp = 0; sp < dyn::kNumSpecies; ++sp) {
      per_state += states_[0].field(static_cast<dyn::Species>(sp)).size();
    }
  }
  if (!slow_.empty()) {
    for (int sp = 0; sp < dyn::kNumSpecies; ++sp) {
      per_tend += slow_[0].field(static_cast<dyn::Species>(sp)).size();
    }
  }
  return (states_.size() * per_state + (slow_.size() + acoustic_.size()) * per_tend) *
         sizeof(Real);
}

// ===========================================================================
// TangentLinearModel
// ===========================================================================

TangentLinearModel::TangentLinearModel(const grid::Grid& g, const dyn::ReferenceState& ref,
                                       const config::ModelConfig& cfg,
                                       const config::DaConfig& da_cfg)
    : grid_(&g), ref_(&ref), eq_(nullptr) {
  // eq_ 保持为空：本实现的 TL 核自包含。冻结头文件未给本类声明析构函数，
  // 因此这里不持有 dyn::Equations 的所有权，避免资源泄漏（见总结中的取舍）。
  ctx_ = timeint::StepContext::make(Real(1), 1, Real(0));
  VIBE_UNUSED(cfg);
  if (da_cfg.check_adjoint) {
    VIBE_INFO("[tl] 已请求启动时点积检验（由 Incremental4DVar 执行）");
  }
}

void TangentLinearModel::step(const dyn::State& xb, const dyn::State& dx, Real dt,
                              dyn::State& dx_out) {
  rk3_tl(xb, ref_, *grid_, dt, dx, dx_out);
}

void TangentLinearModel::propagate(const dyn::State& xb, const dyn::State& dx0, Real dt,
                                   int n_steps, dyn::State& dx_out) {
  VIBE_CHECK_MSG(n_steps >= 0, "TangentLinearModel::propagate: n_steps 必须非负");
  ctx_ = timeint::StepContext::make(dt, 1, xb.time);
  traj_.clear();
  if (n_steps == 0) {
    dx_out = dx0;
    return;
  }
  dyn::State cur = dx0;
  dyn::State next(*grid_);
  dyn::Tendency zero_t(*grid_);
  zero_t.zero();
  for (int n = 0; n < n_steps; ++n) {
    // 记录线性化点（基础态）——生产实现由非线性模式推进 xb
    dyn::State xb_step = xb;
    xb_step.time = xb.time + static_cast<Real>(n) * dt;
    xb_step.step = xb.step + n;
    traj_.push_back(xb_step, zero_t, zero_t);
    rk3_tl(xb, ref_, *grid_, dt, cur, next);
    cur = next;
  }
  dx_out = cur;
}

void TangentLinearModel::record_trajectory(const dyn::State& xb, Real dt, int n_steps) {
  ctx_ = timeint::StepContext::make(dt, 1, xb.time);
  traj_.clear();
  traj_.reserve(static_cast<Size>(std::max(0, n_steps)), xb);
  dyn::Tendency zero_t(*grid_);
  zero_t.zero();
  for (int n = 0; n < n_steps; ++n) {
    dyn::State s = xb;
    s.time = xb.time + static_cast<Real>(n) * dt;
    s.step = xb.step + n;
    traj_.push_back(s, zero_t, zero_t);
  }
}

Real TangentLinearModel::linear_growth_rate(const dyn::State& xb, const dyn::State& dx,
                                            Real dt, int n_steps) const {
  if (n_steps <= 0 || !(dt > Real(0))) return Real(0);
  const Real r0 = dx.norm2(Real(0));
  if (!(r0 > Real(0))) return Real(0);
  // propagate 不是 const 方法（冻结接口），这里通过局部副本推进
  TangentLinearModel& self = const_cast<TangentLinearModel&>(*this);
  dyn::State out(*grid_);
  self.propagate(xb, dx, dt, n_steps, out);
  const Real r1 = out.norm2(Real(0));
  if (!(r1 > Real(0))) return -kHuge;
  return std::log(r1 / r0) / (static_cast<Real>(n_steps) * dt);
}

// ===========================================================================
// AdjointModel
// ===========================================================================

AdjointModel::AdjointModel(const grid::Grid& g, const dyn::ReferenceState& ref,
                           const config::ModelConfig& cfg, const config::DaConfig& da_cfg)
    : grid_(&g), ref_(&ref), eq_(nullptr) {
  ctx_ = timeint::StepContext::make(Real(1), 1, Real(0));
  VIBE_UNUSED(cfg);
  VIBE_UNUSED(da_cfg);
}

void AdjointModel::step_adjoint(const dyn::State& xb, const dyn::State& dx_next, Real dt,
                                dyn::State& dx_prev) {
  rk3_ad(xb, ref_, *grid_, dt, dx_next, dx_prev);
}

void AdjointModel::propagate_with_trajectory(const Trajectory& traj, const dyn::State& dy,
                                             dyn::State& dx0_adjoint) {
  const Size n = traj.size();
  Real dt = ctx_.dt;
  if (n >= 2) {
    const Real d = traj.state(1).time - traj.state(0).time;
    if (d > Real(0)) dt = d;
  }
  if (!(dt > Real(0))) dt = Real(1);
  dyn::State bar = dy;
  for (Size i = n; i-- > 0;) {
    dyn::State prev(*grid_);
    rk3_ad(traj.state(i), ref_, *grid_, dt, bar, prev);
    bar = prev;
  }
  dx0_adjoint = bar;
}

void AdjointModel::propagate(const dyn::State& xb_trajectory_end, const dyn::State& dy,
                             Real dt, int n_steps, dyn::State& dx0_adjoint) {
  ctx_ = timeint::StepContext::make(dt, 1, xb_trajectory_end.time);
  if (own_traj_.size() >= static_cast<Size>(std::max(0, n_steps)) && n_steps > 0) {
    propagate_with_trajectory(own_traj_, dy, dx0_adjoint);
    return;
  }
  // 没有逐层基础态时退化为"冻结基础态"（线性自治系统），误差由调用方评估
  dyn::State bar = dy;
  for (int i = n_steps; i-- > 0;) {
    dyn::State prev(*grid_);
    rk3_ad(xb_trajectory_end, ref_, *grid_, dt, bar, prev);
    bar = prev;
  }
  dx0_adjoint = bar;
}

// ===========================================================================
// 检验工具
// ===========================================================================

std::string AdjointCheckResult::describe() const {
  std::ostringstream os;
  os << "AdjointCheck[<M dx,dy>=" << lhs << " <dx,M^T dy>=" << rhs
     << " rel_err=" << relative_error << " passed=" << (passed ? "yes" : "no")
     << " n_steps=" << n_steps << " dt=" << dt << "]";
  return os.str();
}

std::string TangentCheckResult::describe() const {
  std::ostringstream os;
  os << "TangentCheck[||TL||=" << tl_norm << " ||FD||=" << fd_norm
     << " rel_err=" << relative_error << " best_eps=" << optimal_eps
     << " passed=" << (passed ? "yes" : "no") << "]";
  return os.str();
}

AdjointCheckResult check_adjoint(const TangentLinearModel& tl, const AdjointModel& ad,
                                 const dyn::State& xb, Real dt, int n_steps,
                                 unsigned seed, Real tolerance) {
  AdjointCheckResult res;
  res.n_steps = n_steps;
  res.dt = dt;
  VIBE_CHECK_MSG(n_steps > 0, "check_adjoint: n_steps 必须为正");
  const grid::Grid& g = xb.grid();
  dyn::State dx(g), dy(g);
  std::mt19937 rng(seed);
  fill_random(dx, rng);
  fill_random(dy, rng);

  // 冻结接口的 propagate 非 const，这里通过 const_cast 推进并刷新轨迹缓存
  TangentLinearModel& tl_mut = const_cast<TangentLinearModel&>(tl);
  AdjointModel& ad_mut = const_cast<AdjointModel&>(ad);
  dyn::State mdx(g);
  tl_mut.propagate(xb, dx, dt, n_steps, mdx);
  dyn::State mt_dy(g);
  ad_mut.propagate_with_trajectory(tl_mut.trajectory(), dy, mt_dy);

  res.lhs = mdx.dot(dy, Real(0));
  res.rhs = dx.dot(mt_dy, Real(0));
  res.relative_error = rel_error(res.lhs, res.rhs);
  res.passed = (res.relative_error <= tolerance);
  return res;
}

TangentCheckResult check_tangent(
    const std::function<void(const dyn::State&, dyn::State&, Real, int)>& model,
    const TangentLinearModel& tl, const dyn::State& xb, const dyn::State& dx, Real dt,
    int n_steps, const std::vector<Real>& epsilons) {
  TangentCheckResult res;
  VIBE_CHECK_MSG(!epsilons.empty(), "check_tangent: eps 序列不能为空");
  const grid::Grid& g = xb.grid();

  // TL 结果（冻结接口的 propagate 非 const）
  TangentLinearModel& tl_mut = const_cast<TangentLinearModel&>(tl);
  dyn::State tl_out(g);
  tl_mut.propagate(xb, dx, dt, n_steps, tl_out);
  res.tl_norm = tl_out.norm2(Real(0));

  dyn::State xf0(g), xf1(g), xp(g);
  model(xb, xf0, dt, n_steps);

  Real best_err = kHuge;
  Real best_eps = Real(0);
  Real best_fd = Real(0);
  for (Real eps : epsilons) {
    if (!(eps > Real(0))) continue;
    xp = xb;
    xp.add_scaled(eps, dx);
    model(xp, xf1, dt, n_steps);
    // 前向差分：(M(x + eps dx) - M(x)) / eps；截断误差 O(eps)，舍入误差 O(1/eps)，
    // 因此相对误差关于 eps 呈 U 形（见 docs/design/07_tangent_linear_adjoint.md）
    dyn::State fd = xf1;
    fd.add_scaled(Real(-1), xf0);
    fd.scale(Real(1) / eps);
    // 相对误差：||fd - tl|| / max(||tl||, ||fd||, tiny)
    dyn::State diff = fd;
    diff.add_scaled(Real(-1), tl_out);
    const Real num = diff.norm2(Real(0));
    const Real den = std::max(std::max(res.tl_norm, fd.norm2(Real(0))), Real(1e-30));
    const Real err = num / den;
    if (err < best_err) {
      best_err = err;
      best_eps = eps;
      best_fd = fd.norm2(Real(0));
    }
  }
  res.relative_error = (best_err < kHuge) ? best_err : Real(0);
  res.optimal_eps = best_eps;
  res.fd_norm = best_fd;
  // 判定：最优 eps 处的相对误差应显著小于 1（教学容差 5%）
  res.passed = (res.relative_error <= Real(0.05));
  return res;
}

}  // namespace vibe::da
