/// @file background_error.cpp
/// @brief 背景误差统计：NMC / 集合估计、局地化与 Desroziers 诊断。
///
/// 方法（[V5][V8][V9][V18][V19][V20]）
/// ----------------------------------
///   * NMC（[V5]）：样本 d = x^{T+24} - x^{T+12} 是两次预报误差之差，
///         B ~ 1/2 E[d d^T]
///     本实现据此给出逐变量方差与水平/垂直相关函数，并拟合长度尺度。
///   * 集合（[V19][V20]）：以集合偏差 x'_i = x_i - x_bar 估计
///         B ~ 1/(N-1) sum_i x'_i x'_i^T
///   * 局地化：Schur 乘积 C -> C o rho，rho 取 Gaspari-Cohn（紧支撑 C^2 核）
///         rho(z) = ... (|z| <= 2)，z 为归一化距离；复杂度 O(n^2)。
///   * Desroziers 诊断（[V18]）：
///         chi^2 = mean( (O-B)^2 / sigma^2 )      期望 ~ 1
///         Gleit 比 = rms(O-A) / rms(O-B)          期望 ~ sqrt(0.5)
///     若 chi^2 明显偏离 1，说明 R 的量级不正确；Gleit 比偏离 sqrt(0.5)
///     说明 B/R 的相对权重需要调整。
///
/// 长度尺度拟合：对相关系数取对数做加权线性回归，L = -1/slope；
/// 若回归失败则回退到 e-folding 距离（corr 首次降到 e^{-1} 的距离）。
///
/// 复杂度：estimate O(n_samples * N)；localize O(n^2)；diagnose O(n)。
///
/// 文献：[V5] Parrish & Derber (1992)；[V8] Bannister (2008) I；
///       [V9] Bannister (2008) II；[V18] Desroziers et al. (2005)；
///       [V19] Evensen (1994)；[V20] Hamill & Snyder (2000)。

#include "vibe/da/background_error.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "vibe/common/constants.hpp"
#include "vibe/common/error.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/grid/field.hpp"
#include "vibe/grid/geometry.hpp"
#include "vibe/io/field_io.hpp"

namespace vibe::da {

// ===========================================================================
// 统计量描述
// ===========================================================================

std::string ErrorStatistics::describe() const {
  std::ostringstream os;
  os << "ErrorStatistics[samples=" << samples << " L_h=" << horizontal_length_scale
     << " m L_v=" << vertical_length_scale << " m vars=" << variance.size() << "]";
  if (!variance.empty()) {
    Real lo = variance[0], hi = variance[0], sum = Real(0);
    for (Real v : variance) {
      lo = std::min(lo, v);
      hi = std::max(hi, v);
      sum += v;
    }
    os << " var=[" << lo << ", " << hi << "] mean=" << sum / static_cast<Real>(variance.size());
  }
  if (!vertical_correlation.empty()) {
    os << " 垂直相关(lag=1.." << vertical_correlation.size() << ")=";
    const Size m = std::min<Size>(vertical_correlation.size(), 5);
    for (Size i = 0; i < m; ++i) {
      os << vertical_correlation[i];
      if (i + 1 < m) os << ',';
    }
  }
  return os.str();
}

std::string DiagnosticRatios::describe() const {
  std::ostringstream os;
  os << "Diagnostics[O-B rms=" << ob_minus_b_rms << " O-A rms=" << ob_minus_a_rms
     << " B-A rms=" << b_minus_a_rms << " chi2/obs=" << chi_square_per_obs
     << " Gleit=" << gleit_ratio << " (期望 sqrt(0.5)=" << std::sqrt(Real(0.5)) << ")]";
  return os.str();
}

// ===========================================================================
// 内部工具
// ===========================================================================

namespace {

/// 物种数
inline constexpr int kSpecies = dyn::kNumSpecies;

/// 对相关系数廓线拟合长度尺度（spacing 为相邻 lag 的物理距离）
Real fit_length_scale(const std::vector<Real>& corr, Real spacing) {
  if (corr.empty() || !(spacing > Real(0))) return Real(0);
  // 线性回归 ln(rho) = a - d/L，d = (m+1)*spacing
  Real sx = Real(0), sy = Real(0), sxx = Real(0), sxy = Real(0);
  int n = 0;
  for (Size m = 0; m < corr.size(); ++m) {
    const Real rho = corr[m];
    if (!(rho > Real(1e-6))) continue;  // 只使用正值点
    const Real d = static_cast<Real>(m + 1) * spacing;
    const Real y = std::log(rho);
    sx += d; sy += y; sxx += d * d; sxy += d * y;
    ++n;
  }
  if (n >= 2) {
    const Real denom = static_cast<Real>(n) * sxx - sx * sx;
    if (std::abs(denom) > Real(1e-30)) {
      const Real slope = (static_cast<Real>(n) * sxy - sx * sy) / denom;
      if (slope < Real(-1e-30)) return -Real(1) / slope;
    }
  }
  // 回退：e-folding 距离（线性插值到 rho = e^{-1}）
  const Real target = std::exp(Real(-1));
  for (Size m = 0; m < corr.size(); ++m) {
    if (corr[m] <= target) {
      const Real r0 = (m == 0) ? Real(1) : corr[m - 1];
      const Real d0 = (m == 0) ? Real(0) : static_cast<Real>(m) * spacing;
      const Real d1 = static_cast<Real>(m + 1) * spacing;
      const Real r1 = corr[m];
      if (std::abs(r0 - r1) < Real(1e-30)) return d1;
      const Real w = (r0 - target) / (r0 - r1);
      return d0 + w * (d1 - d0);
    }
  }
  return static_cast<Real>(corr.size()) * spacing;
}

/// 由"偏差样本"（长度 n_samples，每个 State）计算逐物种方差与相关廓线
struct SampleStats {
  std::vector<Real> variance;              ///< [species]
  std::vector<Real> horizontal_corr;       ///< lag = 1..mlag
  std::vector<Real> vertical_corr;         ///< lag = 1..nz-1
  Real horizontal_length_scale = Real(0);
  Real vertical_length_scale = Real(0);
  int samples = 0;
};

SampleStats analyze_samples(const std::vector<dyn::State>& samples, const grid::Grid& g,
                            bool half) {
  SampleStats st;
  st.variance.assign(static_cast<Size>(kSpecies), Real(0));
  st.samples = static_cast<int>(samples.size());
  if (samples.empty()) return st;
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();
  const Size N = static_cast<Size>(nx) * static_cast<Size>(ny) * static_cast<Size>(nz);
  const Real nsamp = static_cast<Real>(samples.size());

  for (int sp = 0; sp < kSpecies; ++sp) {
    Real sum2 = Real(0);
    for (const auto& s : samples) {
      const auto& f = s.field(static_cast<dyn::Species>(sp));
      for (Int k = 0; k < nz; ++k) {
        for (Int j = 0; j < ny; ++j) {
          for (Int i = 0; i < nx; ++i) {
            const Real v = f.at(i, j, k);
            sum2 += v * v;
          }
        }
      }
    }
    Real var = sum2 / (nsamp * static_cast<Real>(N));
    if (half) var *= Real(0.5);  // NMC：d 是两个误差之差
    st.variance[static_cast<Size>(sp)] = var;
  }

  // 选择参考物种（方差最大的动量/位温类变量，优先 U, V, Theta）
  int ref = 0;
  for (int sp : {0, 1, 4, 5, 6}) {
    if (sp < kSpecies && st.variance[static_cast<Size>(sp)] >
                             st.variance[static_cast<Size>(ref)]) {
      ref = sp;
    }
  }
  const Real ref_var = st.variance[static_cast<Size>(ref)];
  if (!(ref_var > Real(0))) return st;

  // 水平相关（x 方向，lag 1..mlag）
  const Int mlag = std::min<Int>(8, std::max<Int>(1, nx / 2));
  st.horizontal_corr.assign(static_cast<Size>(mlag), Real(0));
  {
    Real norm = Real(0);
    for (const auto& s : samples) {
      const auto& f = s.field(static_cast<dyn::Species>(ref));
      for (Int k = 0; k < nz; ++k) {
        for (Int j = 0; j < ny; ++j) {
          for (Int i = 0; i < nx; ++i) {
            const Real v = f.at(i, j, k);
            norm += v * v;
          }
        }
      }
    }
    if (norm > Real(0)) {
      for (Int lag = 1; lag <= mlag; ++lag) {
        Real acc = Real(0);
        for (const auto& s : samples) {
          const auto& f = s.field(static_cast<dyn::Species>(ref));
          for (Int k = 0; k < nz; ++k) {
            for (Int j = 0; j < ny; ++j) {
              for (Int i = 0; i + lag < nx; ++i) {
                acc += f.at(i, j, k) * f.at(i + lag, j, k);
              }
            }
          }
        }
        st.horizontal_corr[static_cast<Size>(lag - 1)] = acc / norm;
      }
    }
    st.horizontal_length_scale = fit_length_scale(st.horizontal_corr, g.geom().dx);
  }

  // 垂直相关（lag 1..nz-1）
  if (nz > 1) {
    st.vertical_corr.assign(static_cast<Size>(nz - 1), Real(0));
    Real norm = Real(0);
    for (const auto& s : samples) {
      const auto& f = s.field(static_cast<dyn::Species>(ref));
      for (Int k = 0; k < nz; ++k) {
        for (Int j = 0; j < ny; ++j) {
          for (Int i = 0; i < nx; ++i) {
            const Real v = f.at(i, j, k);
            norm += v * v;
          }
        }
      }
    }
    if (norm > Real(0)) {
      for (Int lag = 1; lag < nz; ++lag) {
        Real acc = Real(0);
        for (const auto& s : samples) {
          const auto& f = s.field(static_cast<dyn::Species>(ref));
          for (Int k = 0; k + lag < nz; ++k) {
            for (Int j = 0; j < ny; ++j) {
              for (Int i = 0; i < nx; ++i) {
                acc += f.at(i, j, k) * f.at(i, j, k + lag);
              }
            }
          }
        }
        st.vertical_corr[static_cast<Size>(lag - 1)] = acc / norm;
      }
    }
    // 垂直层间距用模式顶/层数近似
    const Real dz = g.geom().z_top / static_cast<Real>(std::max<Int>(nz, 1));
    st.vertical_length_scale = fit_length_scale(st.vertical_corr, dz);
  }
  return st;
}

/// Gaspari-Cohn 局地化函数（[V20] 附录；紧支撑 C^2 核，|z| >= 2 时为零）
inline Real gaspari_cohn(Real z) {
  const Real az = std::abs(z);
  if (az >= Real(2)) return Real(0);
  if (az <= Real(1)) {
    return Real(1) - Real(5) / Real(3) * az * az + Real(5) / Real(8) * az * az * az +
           Real(1) / Real(2) * az * az * az * az - Real(1) / Real(4) * az * az * az * az * az;
  }
  const Real t = Real(2) - az;
  return Real(1) / Real(12) * t * t * t * t * t -
         Real(5) / Real(8) * t * t * t * t + Real(5) / Real(3) * t * t * t -
         Real(5) * t * t + Real(4) * t - Real(2) / Real(3);
}

}  // namespace

// ===========================================================================
// NMC
// ===========================================================================

ErrorStatistics NmcEstimator::estimate(const std::vector<dyn::State>& samples,
                                       const grid::Grid& g) {
  const SampleStats st = analyze_samples(samples, g, /*half=*/true);
  ErrorStatistics out;
  out.variance = st.variance;
  out.horizontal_length_scale = st.horizontal_length_scale;
  out.vertical_length_scale = st.vertical_length_scale;
  out.vertical_correlation = st.vertical_corr;
  out.samples = st.samples;
  return out;
}

std::vector<dyn::State> NmcEstimator::load_samples(const std::vector<std::string>& paths,
                                                   const grid::Grid& g) {
  std::vector<dyn::State> out;
  out.reserve(paths.size());
  VIBE_CHECK_MSG(g.nx() > 0 && g.ny() > 0 && g.nz() > 0,
                 "NmcEstimator::load_samples: 网格未初始化");
  for (const auto& p : paths) {
    dyn::State s(g);
    io::read_restart(p, s);   // 只覆盖预报变量；参考态由调用方注入
    out.push_back(std::move(s));
  }
  return out;
}

// ===========================================================================
// 集合
// ===========================================================================

ErrorStatistics EnsembleEstimator::estimate(const std::vector<dyn::State>& members,
                                            const grid::Grid& g) {
  // 先减去集合平均得到偏差样本
  std::vector<dyn::State> pert;
  if (members.empty()) return ErrorStatistics{};
  const Size n = members.size();
  dyn::State mean = members.front().clone();
  mean.scale(Real(1) / static_cast<Real>(n));
  for (Size i = 1; i < n; ++i) mean.add_scaled(Real(1) / static_cast<Real>(n), members[i]);

  pert.reserve(n);
  for (const auto& m : members) {
    dyn::State d = m.clone();
    d.add_scaled(Real(-1), mean);
    pert.push_back(std::move(d));
  }
  // 集合方差使用 1/(N-1) 归一（analyze_samples 用 1/N，这里做一次标定）
  const SampleStats st = analyze_samples(pert, g, /*half=*/false);
  const Real scale = (n > 1) ? static_cast<Real>(n) / static_cast<Real>(n - 1) : Real(1);

  ErrorStatistics out;
  out.variance = st.variance;
  for (auto& v : out.variance) v *= scale;
  out.horizontal_length_scale = st.horizontal_length_scale;
  out.vertical_length_scale = st.vertical_length_scale;
  out.vertical_correlation = st.vertical_corr;
  out.samples = static_cast<int>(n);
  return out;
}

void EnsembleEstimator::localize(std::vector<Real>& covariance, Real radius,
                                 Real length_scale) {
  if (covariance.empty()) return;
  VIBE_CHECK_MSG(radius > Real(0), "localize: radius 必须为正");
  const Real ls = (length_scale > Real(0)) ? length_scale : Real(1);
  const Size total = covariance.size();
  const Size n = static_cast<Size>(std::llround(std::sqrt(static_cast<double>(total))));
  if (n * n == total) {
    // 方阵：对 (i, j) 施加 Schur 乘积，索引距离按 length_scale 归一
    std::vector<Real> tmp = covariance;
    for (Size i = 0; i < n; ++i) {
      for (Size j = 0; j < n; ++j) {
        const Real dist = std::abs(static_cast<Real>(i) - static_cast<Real>(j)) * ls;
        const Real z = dist / radius;
        tmp[i * n + j] = covariance[i * n + j] * gaspari_cohn(z);
      }
    }
    covariance.swap(tmp);
    return;
  }
  // 非方阵：视作"逐 lag 的协方差"向量，lag 间距取 length_scale
  for (Size m = 0; m < total; ++m) {
    const Real dist = static_cast<Real>(m + 1) * ls;
    covariance[m] *= gaspari_cohn(dist / radius);
  }
}

// ===========================================================================
// Desroziers 诊断
// ===========================================================================

DiagnosticRatios diagnose_ob_stats(const std::vector<Real>& omb,
                                   const std::vector<Real>& oma,
                                   const std::vector<Real>& bma,
                                   const std::vector<Real>& sigma) {
  DiagnosticRatios r;
  VIBE_CHECK_MSG(omb.size() == oma.size() && omb.size() == sigma.size(),
                 "diagnose_ob_stats: O-B / O-A / sigma 长度必须一致");
  if (omb.empty()) return r;
  const Real n = static_cast<Real>(omb.size());
  Real s2_omb = Real(0), s2_oma = Real(0), s2_bma = Real(0), chi2 = Real(0);
  for (Size i = 0; i < omb.size(); ++i) {
    s2_omb += omb[i] * omb[i];
    s2_oma += oma[i] * oma[i];
    chi2 += omb[i] * omb[i] / std::max(sigma[i] * sigma[i], Real(1e-30));
  }
  for (Real v : bma) s2_bma += v * v;
  const Real n_bma = bma.empty() ? Real(1) : static_cast<Real>(bma.size());
  r.ob_minus_b_rms = std::sqrt(s2_omb / n);
  r.ob_minus_a_rms = std::sqrt(s2_oma / n);
  r.b_minus_a_rms = std::sqrt(s2_bma / n_bma);
  r.chi_square_per_obs = chi2 / n;
  r.gleit_ratio = (r.ob_minus_b_rms > Real(0)) ? r.ob_minus_a_rms / r.ob_minus_b_rms
                                               : Real(0);
  return r;
}

// ===========================================================================
// BackgroundErrorBuilder
// ===========================================================================

BackgroundErrorBuilder::BackgroundErrorBuilder(const grid::Grid& g,
                                               const dyn::ReferenceState& ref,
                                               const config::DaConfig& cfg)
    : grid_(&g), ref_(&ref), cfg_(cfg) {}

std::unique_ptr<ControlVariableTransform> BackgroundErrorBuilder::build() const {
  ControlVariableConfig cv;
  cv.balance = balance_from_string(cfg_.background.balance);
  cv.horizontal_length_scale = (stats_.horizontal_length_scale > Real(0))
                                   ? stats_.horizontal_length_scale
                                   : cfg_.background.horizontal_length_scale;
  cv.vertical_length_scale = (stats_.vertical_length_scale > Real(0))
                                 ? stats_.vertical_length_scale
                                 : cfg_.background.vertical_length_scale;
  cv.diffusion_order = cfg_.background.diffusion_order;
  cv.n_vertical_modes = std::max(1, cfg_.background.n_eofs);
  cv.use_ensemble_component = (cfg_.background.method == "ensemble" ||
                               cfg_.background.method == "hybrid");
  cv.hybrid_weight = cfg_.background.hybrid_weight;

  // 方差：若估计器给出逐物种方差，取前 6 个映射到 {u,v,w,theta,pi,qv}
  if (!stats_.variance.empty()) {
    cv.variance_unbalance.clear();
    const Size nv = std::min<Size>(stats_.variance.size(), 6);
    for (Size i = 0; i < nv; ++i) {
      cv.variance_unbalance.push_back(std::max(stats_.variance[i], Real(0)));
    }
  }
  cv.level_scale_unbalance = Real(1);
  return make_control_variable_transform(*grid_, *ref_, cfg_, cv);
}

}  // namespace vibe::da
