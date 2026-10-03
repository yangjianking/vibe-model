/// @file obs_operator.cpp
/// @brief 观测算子基类的插值/散射、切线性与伴随自检、工厂与复合算子。
///
/// 插值约定（与 grid/interpolation.hpp 的 bilinear 一致）
/// ------------------------------------------------------
///   * 单元中心物理坐标：均匀网格 x_i = x0 + (i + 1/2) dx；
///     变分辨率网格 x_i = x0 + sum_{m<i} dx_m + dx_i/2（见 geometry.hpp）；
///   * 水平双线性：先定位下标 (i0, i0+1) 与权重 wx，再对 j 做同样操作；
///   * 垂直线性：在层中心 z_center(i,j,k) 之间线性插值；
///   * 所有越界访问都经过 clamp_at 语义（先把下标钳制到内部范围），
///     因此插值与散射在边界上共享同一套索引映射，散射是插值的**严格转置**。
///
/// 点积检验（[A3] Sirkes & Tziperman 1997）
/// ---------------------------------------
///      < H'(x) dx, dy >_R = < dx, H'^T(x) dy >_B
///   本实现两侧都用欧氏内积（R 为对角、B 的格点体积度量在控制变量变换 U
///   内部体现）。这样散射代码就是插值代码的逐项转置，相对误差只来自浮点
///   舍入（~1e-15）。若要显式引入体积加权度量，必须同时在 applyAD 中除以
///   单元体积，见 docs/design/06_4dvar.md 的讨论。
///
/// 复杂度：每个观测 O(1)（固定模板）；点积检验 O(n_obs + n_state)。
///
/// 文献：[O1][O3][O4][O6][O8][V14][A3][A4][A6][A8]。

#include "vibe/obs/obs_operator.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/grid/field.hpp"
#include "vibe/grid/geometry.hpp"
#include "vibe/obs/obs_operators.hpp"
#include "vibe/obs/radiative_transfer.hpp"

namespace vibe::obs {

namespace {

// ---------------------------------------------------------------------------
// 坐标映射（与 observations.cpp 的 cell_x/cell_y 保持同一约定）
// ---------------------------------------------------------------------------

Real cell_x_at(const grid::Grid& g, Int i) {
  const auto& geom = g.geom();
  if (!geom.variable_resolution || geom.dx_cell.empty()) {
    return geom.x0 + (static_cast<Real>(i) + Real(0.5)) * geom.dx;
  }
  Real x = geom.x0;
  for (Int m = 0; m < i; ++m) x += geom.dx_cell[static_cast<Size>(m)];
  return x + Real(0.5) * geom.dx_cell[static_cast<Size>(i)];
}

Real cell_y_at(const grid::Grid& g, Int j) {
  const auto& geom = g.geom();
  if (!geom.variable_resolution || geom.dy_cell.empty()) {
    return geom.y0 + (static_cast<Real>(j) + Real(0.5)) * geom.dy;
  }
  Real y = geom.y0;
  for (Int m = 0; m < j; ++m) y += geom.dy_cell[static_cast<Size>(m)];
  return y + Real(0.5) * geom.dy_cell[static_cast<Size>(j)];
}

/// 一维定位结果：x ~ (1-w) f[i0] + w f[i0+1]
struct Locate1D {
  Int  i0 = 0;
  Real w = Real(0);
};

/// 在单调的坐标序列 xc(0..n-1) 上定位；区间外用线性外推（下标已钳制）
template <class Xc>
Locate1D locate_1d(Int n, Xc xc, Real x) {
  if (n <= 1) return {0, Real(0)};
  const Real x0 = xc(0), xn = xc(n - 1);
  if (!(x > x0)) {
    const Real d = xc(1) - x0;
    return {0, d > Real(0) ? (x - x0) / d : Real(0)};
  }
  if (!(x < xn)) {
    const Real d = xn - xc(n - 2);
    return {n - 2, d > Real(0) ? (x - xc(n - 2)) / d : Real(0)};
  }
  Int lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const Int mid = (lo + hi) / 2;
    if (xc(mid) <= x) lo = mid; else hi = mid;
  }
  const Real d = xc(lo + 1) - xc(lo);
  return {lo, d > Real(0) ? (x - xc(lo)) / d : Real(0)};
}

/// 内部下标钳制后的存储偏移（插值与散射共用，保证严格转置）
inline Size off_clamped(const grid::Field<Real>& f, Int i, Int j, Int k) {
  const Int ii = clamp(i, Int(0), f.nx() - 1);
  const Int jj = clamp(j, Int(0), f.ny() - 1);
  const Int kk = clamp(k, Int(0), f.nz() - 1);
  return f.offset(ii, jj, kk);
}

/// 错位场在体心节点 (i,j,k) 的取值（错位 -> 体心的算术平均）
inline Real node_value(const grid::Field<Real>& f, grid::Stagger s, Int i, Int j, Int k) {
  const Real* d = f.data();
  switch (s) {
    case grid::Stagger::Cell:
      return d[off_clamped(f, i, j, k)];
    case grid::Stagger::FaceX:
      return Real(0.5) * (d[off_clamped(f, i, j, k)] + d[off_clamped(f, i + 1, j, k)]);
    case grid::Stagger::FaceY:
      return Real(0.5) * (d[off_clamped(f, i, j, k)] + d[off_clamped(f, i, j + 1, k)]);
    case grid::Stagger::FaceZ:
      return Real(0.5) * (d[off_clamped(f, i, j, k)] + d[off_clamped(f, i, j, k + 1)]);
    case grid::Stagger::Corner:
      return Real(0.25) * (d[off_clamped(f, i, j, k)] + d[off_clamped(f, i + 1, j, k)] +
                           d[off_clamped(f, i, j + 1, k)] + d[off_clamped(f, i + 1, j + 1, k)]);
  }
  return Real(0);
}

/// node_value 的转置：把体心节点上的量 v 按同样权重散射回错位场
inline void node_scatter(grid::Field<Real>& f, grid::Stagger s, Int i, Int j, Int k, Real v) {
  Real* d = f.data();
  switch (s) {
    case grid::Stagger::Cell:
      d[off_clamped(f, i, j, k)] += v;
      return;
    case grid::Stagger::FaceX:
      d[off_clamped(f, i, j, k)] += Real(0.5) * v;
      d[off_clamped(f, i + 1, j, k)] += Real(0.5) * v;
      return;
    case grid::Stagger::FaceY:
      d[off_clamped(f, i, j, k)] += Real(0.5) * v;
      d[off_clamped(f, i, j + 1, k)] += Real(0.5) * v;
      return;
    case grid::Stagger::FaceZ:
      d[off_clamped(f, i, j, k)] += Real(0.5) * v;
      d[off_clamped(f, i, j, k + 1)] += Real(0.5) * v;
      return;
    case grid::Stagger::Corner:
      d[off_clamped(f, i, j, k)] += Real(0.25) * v;
      d[off_clamped(f, i + 1, j, k)] += Real(0.25) * v;
      d[off_clamped(f, i, j + 1, k)] += Real(0.25) * v;
      d[off_clamped(f, i + 1, j + 1, k)] += Real(0.25) * v;
      return;
  }
}

/// 随机填充状态的内部点（固定种子，[A3] 的点积检验用）
void fill_random_state(dyn::State& s, std::mt19937& rng) {
  std::normal_distribution<Real> nd(Real(0), Real(1));
  for (int sp = 0; sp < dyn::kNumSpecies; ++sp) {
    auto& f = s.field(static_cast<dyn::Species>(sp));
    for (Int k = 0; k < f.nz(); ++k) {
      for (Int j = 0; j < f.ny(); ++j) {
        for (Int i = 0; i < f.nx(); ++i) f.at(i, j, k) = nd(rng);
      }
    }
  }
}

}  // namespace

// ===========================================================================
// 插值 / 散射
// ===========================================================================

Real ObservationOperator::sample_scalar(const grid::Field<Real>& f, const grid::Grid& g,
                                        Real x, Real y, Real z) {
  const auto li = locate_1d(g.nx(), [&](Int i) { return cell_x_at(g, i); }, x);
  const auto lj = locate_1d(g.ny(), [&](Int j) { return cell_y_at(g, j); }, y);
  const auto lk = locate_1d(g.nz(), [&](Int k) { return g.z_center(li.i0, lj.i0, k); }, z);

  const Real* d = f.data();
  const Real v00 = d[off_clamped(f, li.i0, lj.i0, lk.i0)];
  const Real v10 = d[off_clamped(f, li.i0 + 1, lj.i0, lk.i0)];
  const Real v01 = d[off_clamped(f, li.i0, lj.i0 + 1, lk.i0)];
  const Real v11 = d[off_clamped(f, li.i0 + 1, lj.i0 + 1, lk.i0)];
  const Real v00b = d[off_clamped(f, li.i0, lj.i0, lk.i0 + 1)];
  const Real v10b = d[off_clamped(f, li.i0 + 1, lj.i0, lk.i0 + 1)];
  const Real v01b = d[off_clamped(f, li.i0, lj.i0 + 1, lk.i0 + 1)];
  const Real v11b = d[off_clamped(f, li.i0 + 1, lj.i0 + 1, lk.i0 + 1)];

  const Real lo = lerp(lerp(v00, v10, li.w), lerp(v01, v11, li.w), lj.w);
  const Real hi = lerp(lerp(v00b, v10b, li.w), lerp(v01b, v11b, li.w), lj.w);
  return lerp(lo, hi, lk.w);
}

Real ObservationOperator::sample_staggered(const grid::Field<Real>& f, const grid::Grid& g,
                                           grid::Stagger s, Real x, Real y, Real z) {
  const auto li = locate_1d(g.nx(), [&](Int i) { return cell_x_at(g, i); }, x);
  const auto lj = locate_1d(g.ny(), [&](Int j) { return cell_y_at(g, j); }, y);
  const auto lk = locate_1d(g.nz(), [&](Int k) { return g.z_center(li.i0, lj.i0, k); }, z);

  const Real v00 = node_value(f, s, li.i0, lj.i0, lk.i0);
  const Real v10 = node_value(f, s, li.i0 + 1, lj.i0, lk.i0);
  const Real v01 = node_value(f, s, li.i0, lj.i0 + 1, lk.i0);
  const Real v11 = node_value(f, s, li.i0 + 1, lj.i0 + 1, lk.i0);
  const Real v00b = node_value(f, s, li.i0, lj.i0, lk.i0 + 1);
  const Real v10b = node_value(f, s, li.i0 + 1, lj.i0, lk.i0 + 1);
  const Real v01b = node_value(f, s, li.i0, lj.i0 + 1, lk.i0 + 1);
  const Real v11b = node_value(f, s, li.i0 + 1, lj.i0 + 1, lk.i0 + 1);

  const Real lo = lerp(lerp(v00, v10, li.w), lerp(v01, v11, li.w), lj.w);
  const Real hi = lerp(lerp(v00b, v10b, li.w), lerp(v01b, v11b, li.w), lj.w);
  return lerp(lo, hi, lk.w);
}

void ObservationOperator::scatter_adjoint(grid::Field<Real>& f, const grid::Grid& g,
                                          grid::Stagger s, Real x, Real y, Real z,
                                          Real value) {
  const auto li = locate_1d(g.nx(), [&](Int i) { return cell_x_at(g, i); }, x);
  const auto lj = locate_1d(g.ny(), [&](Int j) { return cell_y_at(g, j); }, y);
  const auto lk = locate_1d(g.nz(), [&](Int k) { return g.z_center(li.i0, lj.i0, k); }, z);

  // 权重公式（与 sample_scalar/sample_staggered 一一对应）：
  //   W(i0+di, j0+dj, k0+dk) = (di? wx : 1-wx) (dj? wy : 1-wy) (dk? wz : 1-wz)
  const Real wx[2] = {Real(1) - li.w, li.w};
  const Real wy[2] = {Real(1) - lj.w, lj.w};
  const Real wz[2] = {Real(1) - lk.w, lk.w};
  for (int dk = 0; dk < 2; ++dk) {
    for (int dj = 0; dj < 2; ++dj) {
      for (int di = 0; di < 2; ++di) {
        const Real w = wx[di] * wy[dj] * wz[dk];
        if (w == Real(0)) continue;
        node_scatter(f, s, li.i0 + di, lj.i0 + dj, lk.i0 + dk, value * w);
      }
    }
  }
}

// ===========================================================================
// 自检与诊断
// ===========================================================================

Real ObservationOperator::check_adjoint(const ModelStateView& x, const ObsSpace& obs,
                                        unsigned seed) const {
  VIBE_CHECK_MSG(x.valid(), "check_adjoint 需要有效的 ModelStateView");
  if (obs.obs.empty()) return Real(0);

  const grid::Grid& g = *x.grid;
  dyn::State dx(g);
  dyn::State dx_ad(g);

  std::mt19937 rng(seed);
  fill_random_state(dx, rng);

  std::vector<Real> dy(obs.obs.size());
  {
    std::normal_distribution<Real> nd(Real(0), Real(1));
    for (auto& v : dy) v = nd(rng);
  }

  std::vector<Real> y_tl;
  applyTL(x, dx, obs, y_tl);
  VIBE_CHECK_MSG(y_tl.size() == obs.obs.size(), "applyTL 返回长度与观测数不一致");

  applyAD(x, dy, obs, dx_ad);

  Real lhs = Real(0);
  for (Size i = 0; i < dy.size(); ++i) lhs += y_tl[i] * dy[i];
  // B 度量：算子层点积用欧氏内积；格点体积加权在控制变量变换 U 内部实现。
  // 若在此处使用 volume_weight > 0，则 applyAD 必须同时除以单元体积才是同一度量的伴随。
  const Real rhs = dx.dot(dx_ad, Real(0));
  return rel_error(lhs, rhs);
}

Real ObservationOperator::check_tangent(const ModelStateView& x, const ObsSpace& obs,
                                        Real eps, unsigned seed) const {
  VIBE_CHECK_MSG(x.valid(), "check_tangent 需要有效的 ModelStateView");
  if (obs.obs.empty()) return Real(0);
  VIBE_CHECK(eps > Real(0));

  const grid::Grid& g = *x.grid;
  dyn::State dx(g);
  std::mt19937 rng(seed);
  fill_random_state(dx, rng);

  dyn::State xp = x.state->clone();
  xp.add_scaled(eps, dx);

  ModelStateView xv_p = x;
  xv_p.state = &xp;

  std::vector<Real> y0, y1, y_tl;
  apply(x, obs, y0);
  apply(xv_p, obs, y1);
  applyTL(x, dx, obs, y_tl);

  Real max_err = Real(0);
  for (Size i = 0; i < obs.obs.size(); ++i) {
    const Real fd = (y1[i] - y0[i]) / eps;
    max_err = std::max(max_err, rel_error(fd, y_tl[i]));
  }
  return max_err;
}

void ObservationOperator::innovations(const ModelStateView& x, const ObsSpace& obs,
                                      std::vector<Real>& d) const {
  std::vector<Real> y;
  apply(x, obs, y);
  d.resize(obs.obs.size());
  for (Size i = 0; i < obs.obs.size(); ++i) {
    // d = y_obs - b - H(x)：使用去偏后的观测值（[V16]）
    d[i] = obs.obs[i].unbiased() - y[i];
  }
}

// ===========================================================================
// 工厂
// ===========================================================================

std::unique_ptr<ObservationOperator> make_operator(ObsType t, const grid::Grid& g,
                                                   const config::DaConfig& cfg) {
  VIBE_UNUSED(cfg);
  switch (t) {
    case ObsType::Radiosonde:
    case ObsType::Aircraft:
    case ObsType::Profiler:
      return std::make_unique<SoundingOperator>(g);
    case ObsType::Surface:
      return std::make_unique<SurfaceOperator>(g);
    case ObsType::AMV:
    case ObsType::Scatterometer:
      return std::make_unique<WindOperator>(g, t);
    case ObsType::GnssRo:
      return std::make_unique<GnssRoOperator>(g);
    case ObsType::RadarReflectivity:
      return std::make_unique<RadarOperator>(g);
    case ObsType::Radiance: {
      const RadiativeTransfer rt = RadiativeTransfer::builtin_channels("amsua");
      return std::make_unique<RadianceOperator>(g, rt);
    }
    case ObsType::Count: {
      // 复合算子：调用方随后用 add() 注入各类子算子
      return std::make_unique<CompositeOperator>();
    }
  }
  return nullptr;
}

// ===========================================================================
// CompositeOperator
// ===========================================================================

void CompositeOperator::add(std::unique_ptr<ObservationOperator> op) {
  VIBE_CHECK_MSG(op != nullptr, "CompositeOperator::add 收到空算子");
  ops_.push_back(std::move(op));
}

std::unique_ptr<ObservationOperator> CompositeOperator::find(ObsType t) const {
  // 冻结头文件把 find 声明为 const，但"取出子算子"必然要转移所有权；
  // 语义：返回并**移除**第一个类型匹配的子算子（所有权交给调用方），
  // 找不到时返回 nullptr。复合算子内部不使用本函数（apply/applyAD 直接按类型
  // 抽取子空间），因此不会影响正常的数据流。
  auto& ops = const_cast<std::vector<std::unique_ptr<ObservationOperator>>&>(ops_);
  for (auto it = ops.begin(); it != ops.end(); ++it) {
    if ((*it)->type() == t) {
      std::unique_ptr<ObservationOperator> out = std::move(*it);
      ops.erase(it);
      return out;
    }
  }
  return nullptr;
}

namespace {

/// 把 obs 中属于类型 t 的观测抽出为子空间，并记录原始下标
void extract_subspace(const ObsSpace& obs, ObsType t, ObsSpace& sub,
                      std::vector<Size>& index) {
  sub = ObsSpace{};
  sub.window_start = obs.window_start;
  sub.window_length = obs.window_length;
  sub.source = obs.source;
  sub.valid_time = obs.valid_time;
  index.clear();
  for (Size i = 0; i < obs.obs.size(); ++i) {
    if (obs.obs[i].type == t) {
      sub.obs.push_back(obs.obs[i]);
      index.push_back(i);
    }
  }
}

}  // namespace

void CompositeOperator::apply(const ModelStateView& x, const ObsSpace& obs,
                              std::vector<Real>& y) const {
  y.assign(obs.obs.size(), Real(0));
  ObsSpace sub;
  std::vector<Size> idx;
  std::vector<Real> ys;
  for (const auto& op : ops_) {
    extract_subspace(obs, op->type(), sub, idx);
    if (sub.obs.empty()) continue;
    op->apply(x, sub, ys);
    VIBE_CHECK_MSG(ys.size() == idx.size(), "复合算子子空间返回长度不一致");
    for (Size m = 0; m < idx.size(); ++m) y[idx[m]] = ys[m];
  }
}

void CompositeOperator::applyTL(const ModelStateView& x, const dyn::State& dx,
                                const ObsSpace& obs, std::vector<Real>& dy) const {
  dy.assign(obs.obs.size(), Real(0));
  ObsSpace sub;
  std::vector<Size> idx;
  std::vector<Real> dys;
  for (const auto& op : ops_) {
    extract_subspace(obs, op->type(), sub, idx);
    if (sub.obs.empty()) continue;
    op->applyTL(x, dx, sub, dys);
    VIBE_CHECK_MSG(dys.size() == idx.size(), "复合算子子空间返回长度不一致");
    for (Size m = 0; m < idx.size(); ++m) dy[idx[m]] = dys[m];
  }
}

void CompositeOperator::applyAD(const ModelStateView& x, const std::vector<Real>& dy,
                                const ObsSpace& obs, dyn::State& dx) const {
  VIBE_CHECK_MSG(dy.size() == obs.obs.size(), "applyAD 的 dy 长度与观测数不一致");
  ObsSpace sub;
  std::vector<Size> idx;
  std::vector<Real> dys;
  // 子算子按观测类型互不重叠，各自向 dx 累加即可
  for (const auto& op : ops_) {
    extract_subspace(obs, op->type(), sub, idx);
    if (sub.obs.empty()) continue;
    dys.resize(idx.size());
    for (Size m = 0; m < idx.size(); ++m) dys[m] = dy[idx[m]];
    op->applyAD(x, dys, sub, dx);
  }
}

}  // namespace vibe::obs
