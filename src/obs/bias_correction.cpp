/// @file bias_correction.cpp
/// @brief 变分偏差订正（VarBC）：预报因子、分组系数与 Gauss-Newton 更新。
///
/// 代价函数（按分组求解，[V16][V17]）
/// ---------------------------------
/// 对某个 (ObsType, channel) 分组，记 d_i = y_i - H_i(x)（新息），
/// 预报因子行向量 X_i（长度 p），系数 beta：
///
///     J(beta) = 1/2 sum_i (d_i - X_i beta)^2 / sigma_i^2
///             + 1/2 (beta - beta^b)^T B_beta^{-1} (beta - beta^b)
///
/// 梯度与 Gauss-Newton 步（A = sum X_i^T X_i / sigma^2，b = sum X_i^T d_i / sigma^2）：
///
///     g(beta)  = -(b - A beta) + B_beta^{-1} (beta - beta^b)
///     Delta beta = (A + B_beta^{-1})^{-1} (-g) = (A + B_beta^{-1})^{-1} (b - A beta - B_beta^{-1} beta)
///     beta <- beta + relaxation * Delta beta
///
/// 本实现用标量对角先验 B_beta = background_variance * I，于是
/// (A + I/sigma_b^2) 是 p x p 对称正定矩阵，用 Cholesky 求解（p = 7）。
///
/// 复杂度：update O(n_obs p^2)；predict O(p)。
///
/// 文献：[V16] Dee (2005)；[V17] Auligné et al. (2007)；[O8] Lorenc et al. (2000)。

#include "vibe/obs/bias_correction.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "vibe/common/error.hpp"

namespace vibe::obs {

// ===========================================================================
// BiasPredictor
// ===========================================================================

void BiasPredictor::expand(Real* x) const noexcept {
  x[0] = Real(1);
  x[1] = scan_angle;
  x[2] = scan_angle * scan_angle;
  x[3] = layer_thickness;
  x[4] = surface_type;
  x[5] = surface_pressure;
  x[6] = cloud_amount;
}

std::vector<Real> BiasPredictor::to_vector() const {
  std::vector<Real> x(static_cast<Size>(kNumPredictors), Real(0));
  expand(x.data());
  return x;
}

BiasPredictor BiasPredictor::from_observation(const Observation& o) noexcept {
  BiasPredictor p;
  // 缺少扫描几何/下垫面信息时的保守默认：用观测高度作为层厚代理，
  // 其余因子置零（零系数的因子不贡献偏差）。
  p.scan_angle = Real(0);
  p.layer_thickness = o.z;
  p.surface_type = Real(0);
  p.surface_pressure = Real(0);
  p.cloud_amount = Real(0);
  return p;
}

// ===========================================================================
// BiasCoefficientSet
// ===========================================================================

Real BiasCoefficientSet::predict(const BiasPredictor& p, int n_predictors) const noexcept {
  const int np = std::min(n_predictors, static_cast<int>(beta.size()));
  if (np <= 0) return Real(0);
  Real x[BiasPredictor::kNumPredictors];
  p.expand(x);
  Real b = Real(0);
  for (int j = 0; j < np; ++j) b += beta[static_cast<Size>(j)] * x[j];
  return b;
}

// ===========================================================================
// VariationalBiasCorrection
// ===========================================================================

namespace {

/// 组键：(ObsType, channel)
using GroupKey = std::pair<int, int>;

GroupKey key_of(ObsType t, int channel) {
  return {static_cast<int>(t), channel};
}

/// 小型对称正定线性方程组求解（Cholesky，带对角回退）
bool solve_spd(std::vector<Real>& A, std::vector<Real>& rhs, int n,
               std::vector<Real>& x) {
  // Cholesky: A = L L^T
  std::vector<Real> L(static_cast<Size>(n) * static_cast<Size>(n), Real(0));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      Real s = A[static_cast<Size>(i) * static_cast<Size>(n) + static_cast<Size>(j)];
      for (int k = 0; k < j; ++k) {
        s -= L[static_cast<Size>(i) * static_cast<Size>(n) + static_cast<Size>(k)] *
             L[static_cast<Size>(j) * static_cast<Size>(n) + static_cast<Size>(k)];
      }
      if (i == j) {
        if (!(s > Real(0))) return false;
        L[static_cast<Size>(i) * static_cast<Size>(n) + static_cast<Size>(i)] = std::sqrt(s);
      } else {
        const Real d = L[static_cast<Size>(j) * static_cast<Size>(n) + static_cast<Size>(j)];
        if (!(std::abs(d) > Real(0))) return false;
        L[static_cast<Size>(i) * static_cast<Size>(n) + static_cast<Size>(j)] = s / d;
      }
    }
  }
  // 前代 L y = rhs
  std::vector<Real> yv(static_cast<Size>(n), Real(0));
  for (int i = 0; i < n; ++i) {
    Real s = rhs[static_cast<Size>(i)];
    for (int k = 0; k < i; ++k) {
      s -= L[static_cast<Size>(i) * static_cast<Size>(n) + static_cast<Size>(k)] *
           yv[static_cast<Size>(k)];
    }
    yv[static_cast<Size>(i)] = s / L[static_cast<Size>(i) * static_cast<Size>(n) + static_cast<Size>(i)];
  }
  // 回代 L^T x = y
  x.assign(static_cast<Size>(n), Real(0));
  for (int i = n - 1; i >= 0; --i) {
    Real s = yv[static_cast<Size>(i)];
    for (int k = i + 1; k < n; ++k) {
      s -= L[static_cast<Size>(k) * static_cast<Size>(n) + static_cast<Size>(i)] *
           x[static_cast<Size>(k)];
    }
    x[static_cast<Size>(i)] = s / L[static_cast<Size>(i) * static_cast<Size>(n) + static_cast<Size>(i)];
  }
  return true;
}

}  // namespace

BiasCoefficientSet& VariationalBiasCorrection::add_group(ObsType t, int channel) {
  for (auto& s : sets_) {
    if (s.type == t && s.channel == channel) {
      s.beta.assign(static_cast<Size>(BiasPredictor::kNumPredictors), Real(0));
      s.n_observations = 0;
      s.cost = Real(0);
      return s;
    }
  }
  BiasCoefficientSet s;
  s.type = t;
  s.channel = channel;
  s.beta.assign(static_cast<Size>(BiasPredictor::kNumPredictors), Real(0));
  sets_.push_back(std::move(s));
  return sets_.back();
}

const BiasCoefficientSet* VariationalBiasCorrection::find(ObsType t, int channel) const noexcept {
  for (const auto& s : sets_) {
    if (s.type == t && s.channel == channel) return &s;
  }
  return nullptr;
}

BiasCoefficientSet* VariationalBiasCorrection::find(ObsType t, int channel) noexcept {
  for (auto& s : sets_) {
    if (s.type == t && s.channel == channel) return &s;
  }
  return nullptr;
}

Real VariationalBiasCorrection::predict(const Observation& o, const BiasPredictor& p) const noexcept {
  if (!cfg_.enabled) return Real(0);
  const auto* s = find(o.type, o.channel);
  if (s == nullptr) return Real(0);
  return s->predict(p, cfg_.n_predictors);
}

Real VariationalBiasCorrection::predict(const Observation& o) const noexcept {
  return predict(o, BiasPredictor::from_observation(o));
}

Real VariationalBiasCorrection::correct(const Observation& o, const BiasPredictor& p) const noexcept {
  return o.value - predict(o, p);
}

Real VariationalBiasCorrection::correct(const Observation& o) const noexcept {
  return o.value - predict(o);
}

void VariationalBiasCorrection::correct_inplace(Observation& o, const BiasPredictor& p) const noexcept {
  // 累加到已估偏差：unbiased() = value - bias 即为订正后的观测
  o.bias += predict(o, p);
}

void VariationalBiasCorrection::update(const ObsSpace& obs,
                                       const std::vector<Real>& innovations,
                                       const std::vector<BiasPredictor>* predictors) {
  VIBE_CHECK_MSG(innovations.size() == obs.obs.size(),
                 "VarBC::update: innovations 长度与观测数不一致");
  if (predictors != nullptr) {
    VIBE_CHECK_MSG(predictors->size() == obs.obs.size(),
                   "VarBC::update: predictors 长度与观测数不一致");
  }
  if (!cfg_.enabled || obs.obs.empty()) return;

  const int np = std::min(cfg_.n_predictors, BiasPredictor::kNumPredictors);
  if (np <= 0) return;

  // 逐 (type, channel) 分组累计正规方程
  struct Accum {
    std::vector<Real> A;    // np x np
    std::vector<Real> b;    // np
    Real cost = Real(0);
    Size n = 0;
  };
  std::map<GroupKey, Accum> accs;
  // 确保所有出现的分组都存在系数集
  for (Size i = 0; i < obs.obs.size(); ++i) {
    const auto& o = obs.obs[i];
    if (!o.usable()) continue;
    const GroupKey key = key_of(o.type, o.channel);
    if (accs.find(key) == accs.end()) {
      Accum a;
      a.A.assign(static_cast<Size>(np) * static_cast<Size>(np), Real(0));
      a.b.assign(static_cast<Size>(np), Real(0));
      accs.emplace(key, std::move(a));
      add_group(o.type, o.channel);
    }
  }

  for (Size i = 0; i < obs.obs.size(); ++i) {
    const auto& o = obs.obs[i];
    if (!o.usable()) continue;
    const BiasPredictor p = predictors != nullptr ? (*predictors)[i]
                                                  : BiasPredictor::from_observation(o);
    Real x[BiasPredictor::kNumPredictors];
    p.expand(x);
    const Real sigma = std::max(o.sigma, cfg_.min_sigma);
    const Real w = Real(1) / (sigma * sigma);
    const Real d = innovations[i];
    auto& a = accs[key_of(o.type, o.channel)];
    for (int r = 0; r < np; ++r) {
      a.b[static_cast<Size>(r)] += w * x[r] * d;
      for (int c = 0; c < np; ++c) {
        a.A[static_cast<Size>(r) * static_cast<Size>(np) + static_cast<Size>(c)] +=
            w * x[r] * x[c];
      }
    }
    a.n += 1;
  }

  const Real prior_var = std::max(cfg_.background_variance, Real(1e-12));
  const Real inv_prior = Real(1) / prior_var;

  for (auto& kv : accs) {
    const GroupKey key = kv.first;
    Accum& a = kv.second;
    BiasCoefficientSet* set = find(static_cast<ObsType>(key.first), key.second);
    if (set == nullptr) continue;
    set->beta.resize(static_cast<Size>(np), Real(0));
    set->n_observations = a.n;

    for (int iter = 0; iter < std::max(1, cfg_.max_iterations); ++iter) {
      // 残差代价与新息项
      Real cost = Real(0);
      std::vector<Real> g(static_cast<Size>(np), Real(0));
      for (int r = 0; r < np; ++r) {
        Real Ab = Real(0);
        for (int c = 0; c < np; ++c) {
          Ab += a.A[static_cast<Size>(r) * static_cast<Size>(np) + static_cast<Size>(c)] *
                set->beta[static_cast<Size>(c)];
        }
        g[static_cast<Size>(r)] = -(a.b[static_cast<Size>(r)] - Ab) +
                                  inv_prior * set->beta[static_cast<Size>(r)];
      }
      // 先验项代价（信息量，不含常数）
      for (int r = 0; r < np; ++r) {
        cost += Real(0.5) * inv_prior * set->beta[static_cast<Size>(r)] *
                set->beta[static_cast<Size>(r)];
      }
      set->cost = cost;

      // (A + I/prior_var) Delta = -g
      std::vector<Real> M = a.A;
      for (int r = 0; r < np; ++r) {
        M[static_cast<Size>(r) * static_cast<Size>(np) + static_cast<Size>(r)] += inv_prior;
      }
      std::vector<Real> rhs(static_cast<Size>(np), Real(0));
      for (int r = 0; r < np; ++r) rhs[static_cast<Size>(r)] = -g[static_cast<Size>(r)];
      std::vector<Real> delta;
      if (!solve_spd(M, rhs, np, delta)) {
        // 回退：对角预条件的梯度下降
        delta.assign(static_cast<Size>(np), Real(0));
        for (int r = 0; r < np; ++r) {
          const Real dg = M[static_cast<Size>(r) * static_cast<Size>(np) + static_cast<Size>(r)];
          if (dg > Real(0)) delta[static_cast<Size>(r)] = -g[static_cast<Size>(r)] / dg;
        }
      }
      for (int r = 0; r < np; ++r) {
        set->beta[static_cast<Size>(r)] +=
            cfg_.relaxation * delta[static_cast<Size>(r)];
      }
    }
  }
}

void VariationalBiasCorrection::set_coefficients(ObsType t, int channel,
                                                 const std::vector<Real>& beta) {
  VIBE_CHECK_MSG(beta.size() <= static_cast<Size>(BiasPredictor::kNumPredictors),
                 "set_coefficients: beta 长度超过预报因子数");
  BiasCoefficientSet& s = add_group(t, channel);
  s.beta.assign(static_cast<Size>(BiasPredictor::kNumPredictors), Real(0));
  for (Size j = 0; j < beta.size(); ++j) s.beta[j] = beta[j];
}

void VariationalBiasCorrection::reset_coefficients() {
  for (auto& s : sets_) {
    s.beta.assign(static_cast<Size>(BiasPredictor::kNumPredictors), Real(0));
    s.n_observations = 0;
    s.cost = Real(0);
  }
}

std::string VariationalBiasCorrection::describe() const {
  std::ostringstream os;
  os << "VarBC[groups=" << sets_.size() << " enabled=" << (cfg_.enabled ? "yes" : "no")
     << " predictors=" << cfg_.n_predictors
     << " prior_var=" << cfg_.background_variance << "]";
  for (const auto& s : sets_) {
    Real amax = Real(0), anorm = Real(0);
    for (Real v : s.beta) {
      amax = std::max(amax, std::abs(v));
      anorm += v * v;
    }
    os << "\n  " << to_string(s.type) << " ch=" << s.channel << " n=" << s.n_observations
       << " |beta|max=" << amax << " |beta|=" << std::sqrt(anorm) << " cost=" << s.cost;
  }
  return os.str();
}

}  // namespace vibe::obs
