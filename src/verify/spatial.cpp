#include "vibe/verify/spatial.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "vibe/verify/contingency.hpp"

namespace vibe::verify {
namespace {

/// 场尺寸自检。
void check_field(const std::vector<Real>& f, int nx, int ny, const char* what) {
  if (nx <= 0 || ny <= 0) {
    throw DimensionError(std::string(what) + ": 网格尺寸必须为正 (nx=" +
                         std::to_string(nx) + ", ny=" + std::to_string(ny) + ")");
  }
  const std::size_t need = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
  if (f.size() != need) {
    throw DimensionError(std::string(what) + ": 场长度 " + std::to_string(f.size()) +
                         " 与 nx*ny=" + std::to_string(need) + " 不符");
  }
}

/// 二值化：field >= threshold -> 1，否则 0（NaN 记为 0）。
inline Real binarize(Real v, Real threshold) noexcept {
  if (is_missing(v)) return Real(0);
  return v >= threshold ? Real(1) : Real(0);
}

/// 2D 积分图（summed-area table），尺寸 (nx+1)x(ny+1)，行主序。
/// P[(i+1)*W + (j+1)] = b(i,j) + P[i*W+(j+1)] + P[(i+1)*W+j] - P[i*W+j]，
/// 从而任意矩形和可在 O(1) 内求得。构建复杂度 O(nx*ny)。
std::vector<Real> integral_image(const std::vector<Real>& f, int nx, int ny,
                                 Real threshold, int& stride_y) {
  stride_y = ny + 1;
  const int W = stride_y;
  std::vector<Real> P(static_cast<std::size_t>(nx + 1) * static_cast<std::size_t>(W),
                      Real(0));
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      const Real b = binarize(f[static_cast<std::size_t>(i) * static_cast<std::size_t>(ny) +
                               static_cast<std::size_t>(j)], threshold);
      P[static_cast<std::size_t>(i + 1) * static_cast<std::size_t>(W) +
        static_cast<std::size_t>(j + 1)] =
          b + P[static_cast<std::size_t>(i) * static_cast<std::size_t>(W) +
                static_cast<std::size_t>(j + 1)] +
          P[static_cast<std::size_t>(i + 1) * static_cast<std::size_t>(W) +
            static_cast<std::size_t>(j)] -
          P[static_cast<std::size_t>(i) * static_cast<std::size_t>(W) +
            static_cast<std::size_t>(j)];
    }
  }
  return P;
}

/// 矩形和：i in [i0,i1]，j in [j0,j1]（闭区间）。O(1)。
inline Real rect_sum(const std::vector<Real>& P, int W, int i0, int i1, int j0, int j1) noexcept {
  return P[static_cast<std::size_t>(i1 + 1) * static_cast<std::size_t>(W) +
           static_cast<std::size_t>(j1 + 1)] -
         P[static_cast<std::size_t>(i0) * static_cast<std::size_t>(W) +
           static_cast<std::size_t>(j1 + 1)] -
         P[static_cast<std::size_t>(i1 + 1) * static_cast<std::size_t>(W) +
           static_cast<std::size_t>(j0)] +
         P[static_cast<std::size_t>(i0) * static_cast<std::size_t>(W) +
           static_cast<std::size_t>(j0)];
}

/// DCT-II 的归一化系数：c_0 = sqrt(1/n)，c_k = sqrt(2/n)。
inline Real dct_coeff(int k, int n) noexcept {
  const Real fn = static_cast<Real>(n);
  return k == 0 ? std::sqrt(Real(1) / fn) : std::sqrt(Real(2) / fn);
}

}  // namespace

// ---------------------------------------------------------------------------
// 邻域分数
// ---------------------------------------------------------------------------

// 见 spatial.hpp：积分图实现，O(nx*ny)，与 radius 无关。
std::vector<Real> neighborhood_fractions(const std::vector<Real>& field, int nx, int ny,
                                         Real threshold, int radius) {
  check_field(field, nx, ny, "neighborhood_fractions");
  const int r = radius > 0 ? radius : 0;
  int W = 0;
  const std::vector<Real> P = integral_image(field, nx, ny, threshold, W);
  std::vector<Real> out(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny), Real(0));
  for (int i = 0; i < nx; ++i) {
    const int i0 = std::max(0, i - r);
    const int i1 = std::min(nx - 1, i + r);
    for (int j = 0; j < ny; ++j) {
      const int j0 = std::max(0, j - r);
      const int j1 = std::min(ny - 1, j + r);
      const Real sum = rect_sum(P, W, i0, i1, j0, j1);
      const Real cnt = static_cast<Real>((i1 - i0 + 1) * (j1 - j0 + 1));
      out[static_cast<std::size_t>(i) * static_cast<std::size_t>(ny) +
          static_cast<std::size_t>(j)] = sum / cnt;
    }
  }
  return out;
}

// 见 spatial.hpp：FSS = 1 - sum (Ff-Fo)^2 / (sum Ff^2 + sum Fo^2)；O(nx*ny)。
Real fractions_skill_score(const std::vector<Real>& f, const std::vector<Real>& o,
                           int nx, int ny, Real threshold, int radius) {
  check_field(f, nx, ny, "fractions_skill_score");
  check_field(o, nx, ny, "fractions_skill_score");
  const std::vector<Real> ff = neighborhood_fractions(f, nx, ny, threshold, radius);
  const std::vector<Real> fo = neighborhood_fractions(o, nx, ny, threshold, radius);
  Real num = Real(0), den = Real(0);
  for (std::size_t k = 0; k < ff.size(); ++k) {
    const Real d = ff[k] - fo[k];
    num += d * d;
    den += ff[k] * ff[k] + fo[k] * fo[k];
  }
  if (den <= Real(0)) return kNaN;  // 两场全无事件：FSS 未定义
  return Real(1) - num / den;
}

std::vector<FssPoint> fss_curve(const std::vector<Real>& f, const std::vector<Real>& o,
                                int nx, int ny, Real threshold,
                                const std::vector<int>& radii, Real dx) {
  std::vector<FssPoint> out;
  out.reserve(radii.size());
  for (const int r : radii) {
    FssPoint p;
    p.radius = r;
    p.fss = fractions_skill_score(f, o, nx, ny, threshold, r);
    p.scale = Real(2) * static_cast<Real>(r) * dx;
    out.push_back(p);
  }
  return out;
}

std::vector<Real> fss_vs_scale(const std::vector<Real>& f, const std::vector<Real>& o,
                               int nx, int ny, Real threshold, int max_radius) {
  std::vector<Real> out;
  if (max_radius < 0) return out;
  out.reserve(static_cast<std::size_t>(max_radius) + 1);
  for (int r = 0; r <= max_radius; ++r) {
    out.push_back(fractions_skill_score(f, o, nx, ny, threshold, r));
  }
  return out;
}

Real minimum_resolvable_scale(const std::vector<Real>& f, const std::vector<Real>& o,
                              int nx, int ny, Real threshold, int max_radius,
                              Real fss_target, Real dx) {
  for (int r = 0; r <= max_radius; ++r) {
    const Real s = fractions_skill_score(f, o, nx, ny, threshold, r);
    if (!is_missing(s) && s >= fss_target) {
      return Real(2) * static_cast<Real>(r) * dx;
    }
  }
  return kNaN;
}

// ---------------------------------------------------------------------------
// 模糊检验
// ---------------------------------------------------------------------------

// 邻域概率场：语义包装（与邻域分数完全一致）。O(nx*ny)。
std::vector<Real> neighborhood_probability(const std::vector<Real>& field, int nx, int ny,
                                           Real threshold, int radius) {
  return neighborhood_fractions(field, nx, ny, threshold, radius);
}

// 邻域（模糊）列联表：只要半径内有事件即视为"预报命中"，见 [E8]。O(nx*ny)。
FuzzyScores fuzzy_scores(const std::vector<Real>& f, const std::vector<Real>& o,
                         int nx, int ny, Real threshold, int radius) {
  check_field(f, nx, ny, "fuzzy_scores");
  check_field(o, nx, ny, "fuzzy_scores");
  const std::vector<Real> ff = neighborhood_fractions(f, nx, ny, threshold, radius);
  const std::vector<Real> fo = neighborhood_fractions(o, nx, ny, threshold, radius);
  ContingencyTable t;
  Real sum_f = Real(0), sum_o = Real(0);
  for (std::size_t k = 0; k < ff.size(); ++k) {
    sum_f += ff[k];
    sum_o += fo[k];
    const bool fe = ff[k] > Real(0);
    const bool oe = fo[k] > Real(0);
    if (fe && oe) {
      t.hits += Real(1);
    } else if (!fe && !oe) {
      t.correct_negatives += Real(1);
    } else if (fe && !oe) {
      t.false_alarms += Real(1);
    } else {
      t.misses += Real(1);
    }
  }
  FuzzyScores s;
  s.radius = radius;
  s.n = ff.size();
  s.neighborhood_pod = t.pod();
  s.neighborhood_far = t.far();
  s.neighborhood_csi = t.csi();
  s.neighborhood_ets = t.ets();
  s.neighborhood_bias = t.frequency_bias();
  s.mean_fraction_f = ff.empty() ? kNaN : sum_f / static_cast<Real>(ff.size());
  s.mean_fraction_o = fo.empty() ? kNaN : sum_o / static_cast<Real>(fo.size());
  s.fss = fractions_skill_score(f, o, nx, ny, threshold, radius);
  return s;
}

// ---------------------------------------------------------------------------
// 双惩罚诊断
// ---------------------------------------------------------------------------

// 见 spatial.hpp：整数位移搜索 + 重叠区 MSE。O(nx*ny*(2s+1)^2)。
DisplacementEstimate best_displacement(const std::vector<Real>& f,
                                       const std::vector<Real>& o,
                                       int nx, int ny, int max_shift) {
  check_field(f, nx, ny, "best_displacement");
  check_field(o, nx, ny, "best_displacement");
  const int smax = max_shift > 0 ? max_shift : std::max(nx, ny) - 1;
  DisplacementEstimate best;
  best.mse_at_best = std::numeric_limits<Real>::max();
  best.mse_zero = kNaN;

  // 零位移：全域 MSE。
  Real mse0 = Real(0);
  std::size_t n0 = 0;
  for (std::size_t k = 0; k < f.size(); ++k) {
    if (is_missing(f[k]) || is_missing(o[k])) continue;
    const Real d = f[k] - o[k];
    mse0 += d * d;
    ++n0;
  }
  if (n0 > 0) best.mse_zero = mse0 / static_cast<Real>(n0);

  for (int dx = -smax; dx <= smax; ++dx) {
    for (int dy = -smax; dy <= smax; ++dy) {
      Real mse = Real(0), num = Real(0), nf2 = Real(0), no2 = Real(0);
      std::size_t cnt = 0;
      for (int i = 0; i < nx; ++i) {
        const int si = i + dx;
        if (si < 0 || si >= nx) continue;
        for (int j = 0; j < ny; ++j) {
          const int sj = j + dy;
          if (sj < 0 || sj >= ny) continue;
          const Real fv = f[static_cast<std::size_t>(si) * static_cast<std::size_t>(ny) +
                            static_cast<std::size_t>(sj)];
          const Real ov = o[static_cast<std::size_t>(i) * static_cast<std::size_t>(ny) +
                            static_cast<std::size_t>(j)];
          if (is_missing(fv) || is_missing(ov)) continue;
          const Real d = fv - ov;
          mse += d * d;
          num += fv * ov;
          nf2 += fv * fv;
          no2 += ov * ov;
          ++cnt;
        }
      }
      if (cnt == 0) continue;
      mse /= static_cast<Real>(cnt);
      if (mse < best.mse_at_best) {
        best.mse_at_best = mse;
        best.dx = dx;
        best.dy = dy;
        const Real denom = std::sqrt(nf2 * no2);
        best.overlap_fraction = denom > Real(0) ? num / denom : kNaN;
      }
    }
  }
  if (best.mse_at_best == std::numeric_limits<Real>::max()) best.mse_at_best = kNaN;
  return best;
}

// 幅度/位移误差分解，见 spatial.hpp 与文档 09_verification.md。O(nx*ny*(2s+1)^2)。
DoublePenaltyDiagnosis double_penalty_diagnosis(const std::vector<Real>& f,
                                                const std::vector<Real>& o,
                                                int nx, int ny, int max_shift) {
  check_field(f, nx, ny, "double_penalty_diagnosis");
  check_field(o, nx, ny, "double_penalty_diagnosis");
  DoublePenaltyDiagnosis d;
  Real sum_f = Real(0), sum_o = Real(0), sum_ff = Real(0), sum_oo = Real(0), sum_fo = Real(0);
  Real mse = Real(0);
  std::size_t n = 0;
  for (std::size_t k = 0; k < f.size(); ++k) {
    if (is_missing(f[k]) || is_missing(o[k])) continue;
    sum_f += f[k];
    sum_o += o[k];
    sum_ff += f[k] * f[k];
    sum_oo += o[k] * o[k];
    sum_fo += f[k] * o[k];
    const Real e = f[k] - o[k];
    mse += e * e;
    ++n;
  }
  d.n = n;
  if (n == 0) return d;
  const Real fn = static_cast<Real>(n);
  d.mse_total = mse / fn;
  d.amplitude_error = (sum_f - sum_o) / fn;
  const Real var_f = sum_ff - sum_f * sum_f / fn;
  const Real var_o = sum_oo - sum_o * sum_o / fn;
  if (var_f > Real(0) && var_o > Real(0)) {
    const Real cov = sum_fo - sum_f * sum_o / fn;
    const Real r = cov / std::sqrt(var_f * var_o);
    d.correlation = r < Real(-1) ? Real(-1) : (r > Real(1) ? Real(1) : r);
  } else {
    d.correlation = Real(0);
  }
  d.shift = best_displacement(f, o, nx, ny, max_shift);
  d.mse_displaced = d.shift.mse_at_best;
  if (!is_missing(d.mse_displaced)) {
    const Real residual = d.mse_displaced - d.amplitude_error * d.amplitude_error;
    d.displacement_error = residual > Real(0) ? std::sqrt(residual) : Real(0);
    d.double_penalty_index = d.mse_displaced > Real(0) ? d.mse_total / d.mse_displaced : Real(1);
  }
  return d;
}

// ---------------------------------------------------------------------------
// DCT 尺度分解
// ---------------------------------------------------------------------------

// 正交 DCT-II，可分离实现。O(nx*ny*(nx+ny))。见 spatial.hpp。
void dct2(const std::vector<Real>& in, int nx, int ny, std::vector<Real>& out) {
  check_field(in, nx, ny, "dct2");
  out.assign(in.size(), Real(0));
  std::vector<Real> tmp(in.size(), Real(0));
  const Real pi = Real(3.14159265358979323846);
  const int W = ny;
  // 沿 i（x）做 DCT-II。
  for (int k = 0; k < nx; ++k) {
    const Real ck = dct_coeff(k, nx);
    const Real scale = pi * static_cast<Real>(k) / static_cast<Real>(2 * nx);
    for (int j = 0; j < ny; ++j) {
      Real s = Real(0);
      for (int i = 0; i < nx; ++i) {
        s += in[static_cast<std::size_t>(i) * static_cast<std::size_t>(W) +
                static_cast<std::size_t>(j)] *
             std::cos(scale * static_cast<Real>(2 * i + 1));
      }
      tmp[static_cast<std::size_t>(k) * static_cast<std::size_t>(W) +
          static_cast<std::size_t>(j)] = s * ck;
    }
  }
  // 沿 j（y）做 DCT-II。
  for (int k = 0; k < nx; ++k) {
    for (int l = 0; l < ny; ++l) {
      const Real cl = dct_coeff(l, ny);
      const Real scale = pi * static_cast<Real>(l) / static_cast<Real>(2 * ny);
      Real s = Real(0);
      for (int j = 0; j < ny; ++j) {
        s += tmp[static_cast<std::size_t>(k) * static_cast<std::size_t>(W) +
                 static_cast<std::size_t>(j)] *
             std::cos(scale * static_cast<Real>(2 * j + 1));
      }
      out[static_cast<std::size_t>(k) * static_cast<std::size_t>(W) +
          static_cast<std::size_t>(l)] = s * cl;
    }
  }
}

// 正交 DCT-III（逆变换），可分离实现，满足 idct2(dct2(x)) = x。O(nx*ny*(nx+ny))。
void idct2(const std::vector<Real>& in, int nx, int ny, std::vector<Real>& out) {
  check_field(in, nx, ny, "idct2");
  out.assign(in.size(), Real(0));
  std::vector<Real> tmp(in.size(), Real(0));
  const Real pi = Real(3.14159265358979323846);
  const int W = ny;
  // 沿 l（y）求和（C_y 作用于右侧）。
  for (int k = 0; k < nx; ++k) {
    for (int j = 0; j < ny; ++j) {
      Real s = Real(0);
      for (int l = 0; l < ny; ++l) {
        const Real scale = pi * static_cast<Real>(l) / static_cast<Real>(2 * ny);
        s += in[static_cast<std::size_t>(k) * static_cast<std::size_t>(W) +
                static_cast<std::size_t>(l)] *
             dct_coeff(l, ny) * std::cos(scale * static_cast<Real>(2 * j + 1));
      }
      tmp[static_cast<std::size_t>(k) * static_cast<std::size_t>(W) +
          static_cast<std::size_t>(j)] = s;
    }
  }
  // 沿 k（x）求和（C_x^T 作用于左侧）。
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      Real s = Real(0);
      for (int k = 0; k < nx; ++k) {
        const Real scale = pi * static_cast<Real>(k) / static_cast<Real>(2 * nx);
        s += tmp[static_cast<std::size_t>(k) * static_cast<std::size_t>(W) +
                 static_cast<std::size_t>(j)] *
             dct_coeff(k, nx) * std::cos(scale * static_cast<Real>(2 * i + 1));
      }
      out[static_cast<std::size_t>(i) * static_cast<std::size_t>(W) +
          static_cast<std::size_t>(j)] = s;
    }
  }
}

// 能量按归一化径向波数分带。O(nx*ny*(nx+ny))。
ScaleSpectrum dct_scale_spectrum(const std::vector<Real>& f, int nx, int ny, int n_bands) {
  check_field(f, nx, ny, "dct_scale_spectrum");
  const int K = n_bands > 0 ? n_bands : 1;
  std::vector<Real> sanitized(f.size(), Real(0));
  for (std::size_t k = 0; k < f.size(); ++k) {
    sanitized[k] = is_missing(f[k]) ? Real(0) : f[k];  // 缺测按 0 处理（建议先做 QC）
  }
  std::vector<Real> coeff;
  dct2(sanitized, nx, ny, coeff);

  ScaleSpectrum sp;
  sp.nx = nx;
  sp.ny = ny;
  sp.band_wavenumber.assign(static_cast<std::size_t>(K), Real(0));
  sp.band_energy.assign(static_cast<std::size_t>(K), Real(0));
  std::vector<std::size_t> band_count(static_cast<std::size_t>(K), 0);
  const Real norm = std::sqrt(Real(2));
  Real total = Real(0);
  for (int k = 0; k < nx; ++k) {
    for (int l = 0; l < ny; ++l) {
      const Real kx = static_cast<Real>(k) / static_cast<Real>(nx);
      const Real ly = static_cast<Real>(l) / static_cast<Real>(ny);
      const Real kappa = std::sqrt(kx * kx + ly * ly) / norm;  // in [0,1]
      std::size_t b = static_cast<std::size_t>(kappa * static_cast<Real>(K));
      if (b >= static_cast<std::size_t>(K)) b = static_cast<std::size_t>(K) - 1;
      const Real c = coeff[static_cast<std::size_t>(k) * static_cast<std::size_t>(ny) +
                           static_cast<std::size_t>(l)];
      const Real e = c * c;
      sp.band_energy[b] += e;
      sp.band_wavenumber[b] += kappa;
      band_count[b] += 1;
      total += e;
    }
  }
  for (std::size_t b = 0; b < static_cast<std::size_t>(K); ++b) {
    if (band_count[b] > 0) {
      sp.band_wavenumber[b] /= static_cast<Real>(band_count[b]);
    }
  }
  sp.total_energy = total;
  // 直流分量（k = l = 0）单独记录，便于诊断模式整体偏差。
  const Real dc = coeff[0];
  sp.residual_energy = dc * dc;
  return sp;
}

// ---------------------------------------------------------------------------
// 批量空间评分
// ---------------------------------------------------------------------------

Scores compute_fractional_skill(const std::vector<Real>& f, const std::vector<Real>& o,
                                int nx, int ny, Real threshold,
                                int neighborhood_radius) {
  Scores s = zero_scores(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
  s.fss = fractions_skill_score(f, o, nx, ny, threshold, neighborhood_radius);
  return s;
}

}  // namespace vibe::verify
