#pragma once
/// @file bias_correction.hpp
/// @brief 变分偏差订正（VarBC）与偏差预报因子。
///
/// 背景
/// ----
/// 卫星辐射率、AMV、散射计等观测存在**系统性偏差**，其来源包括仪器定标、
/// 观测几何（扫描角）、气团状态与下垫面。若把偏差当作观测误差处理，会
/// 在同化中引入有偏的分析增量。VarBC（[V16][V17]）把偏差写成一组
/// **预报因子**的线性组合，并把系数与状态一起在变分框架里求解：
///
///     y - H(x) - b(x, beta)  ~ N(0, R)
///     b(x, beta) = sum_{j} X_j(x) * beta_j = X beta
///
/// 同化代价函数中加入系数的惩罚（或先验）项：
///
///     J(beta) = 1/2 sum_i [ y_i - H_i(x) - X_i beta ]^2 / sigma_i^2
///             + 1/2 (beta - beta^b)^T B_beta^{-1} (beta - beta^b)
///
/// 对 beta 的梯度（记为 d = y - H(x) - X beta 为去偏后的新息）：
///
///     dJ/dbeta = - sum_i X_i^T d_i / sigma_i^2 + B_beta^{-1} (beta - beta^b)
///
/// 本实现采用**分组**结构：每个 (ObsType, channel) 各自一套系数，
/// 组内用一次 Gauss-Newton（正规方程）迭代更新，避免跨通道的相互污染。
///
/// 预报因子（本实现的基函数，长度 kNumPredictors = 7）
/// --------------------------------------------------
///     X_0 = 1                      常数项（仪器定标偏差）
///     X_1 = scan_angle             扫描角（一阶，临边效应）
///     X_2 = scan_angle^2           扫描角（二阶）
///     X_3 = layer_thickness        层厚（路径长度代理，[V17] 的 "thickness" 预报因子）
///     X_4 = surface_type           下垫面类型（0 海 / 1 陆 / 2 冰）
///     X_5 = surface_pressure       地面气压（地形效应）
///     X_6 = cloud_amount           云量（云污染的经验订正）
///
/// 复杂度：predict 为 O(kNumPredictors)；update 为 O(n_obs * kNumPredictors^2)
/// （只解一个 7x7 正规方程）。
///
/// 文献：[V16] Dee (2005)；[V17] Auligné et al. (2007)；[O8] Lorenc et al. (2000)。

#include <string>
#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/obs/observations.hpp"

namespace vibe::obs {

/// 偏差预报因子的取值（一次观测对应的 X 向量）
struct BiasPredictor {
  Real scan_angle = Real(0);         ///< 扫描角（弧度；天底 = 0）
  Real layer_thickness = Real(0);    ///< 层厚（m 或 hPa，量纲由调用方统一）
  Real surface_type = Real(0);       ///< 0 = 海洋, 1 = 陆地, 2 = 海冰
  Real surface_pressure = Real(0);   ///< Pa
  Real cloud_amount = Real(0);       ///< 0 - 1

  /// 预报因子个数（固定基函数表）
  static constexpr int kNumPredictors = 7;

  /// 把预报因子展开到长度为 kNumPredictors 的数组（row 为输出缓冲）
  void expand(Real* x) const noexcept;
  /// 展开为 std::vector（长度 kNumPredictors）
  std::vector<Real> to_vector() const;
  /// 由模式/观测几何做一次保守的默认填充（缺少扫描几何的观测）
  static BiasPredictor from_observation(const Observation& o) noexcept;
};

/// 变分偏差订正的配置
struct BiasCorrectionConfig {
  bool enabled = true;
  int  n_predictors = BiasPredictor::kNumPredictors;  ///< 使用的预报因子个数
  Real background_variance = Real(1.0);  ///< beta 先验方差（标量对角 B_beta）
  Real relaxation = Real(1.0);           ///< 更新步长松弛因子（1 = 全 Gauss-Newton 步）
  int  max_iterations = 1;               ///< 每次 update 的迭代次数
  Real min_sigma = Real(1e-6);           ///< sigma 下限，避免除零
};

/// 一个 (ObsType, channel) 分组的系数
struct BiasCoefficientSet {
  ObsType type = ObsType::Radiance;
  int     channel = -1;
  std::vector<Real> beta;      ///< 系数，长度 kNumPredictors
  Size    n_observations = 0;  ///< 参与过更新的观测数（诊断）
  Real    cost = Real(0);      ///< 上一次更新的代价（诊断）

  /// 预报：h^T beta
  Real predict(const BiasPredictor& p, int n_predictors) const noexcept;
};

/// 变分偏差订正
class VariationalBiasCorrection {
 public:
  VariationalBiasCorrection() = default;
  explicit VariationalBiasCorrection(const BiasCorrectionConfig& cfg) : cfg_(cfg) {}

  void configure(const BiasCorrectionConfig& cfg) { cfg_ = cfg; }
  const BiasCorrectionConfig& config() const noexcept { return cfg_; }

  /// 显式加入一个分组（重复加入同一 (type, channel) 时只重置系数）
  BiasCoefficientSet& add_group(ObsType t, int channel);
  /// 查找分组；不存在返回 nullptr
  const BiasCoefficientSet* find(ObsType t, int channel) const noexcept;
  BiasCoefficientSet* find(ObsType t, int channel) noexcept;

  /// 偏差预报 b = X beta；无对应分组时返回 0
  Real predict(const Observation& o, const BiasPredictor& p) const noexcept;
  /// 用由观测本身推断的默认预报因子做预报
  Real predict(const Observation& o) const noexcept;

  /// 订正后的观测值 y - b（不修改输入）
  Real correct(const Observation& o, const BiasPredictor& p) const noexcept;
  Real correct(const Observation& o) const noexcept;
  /// 就地订正：把 b 累加进 o.bias（可逆：set_coefficients(0) 后恢复）
  void correct_inplace(Observation& o, const BiasPredictor& p) const noexcept;

  /// 用新息向量做一次变分更新（Gauss-Newton 迭代）
  ///   @param obs          观测空间（与 innovations 等长、同序）
  ///   @param innovations  d = y - H(x) - b（已去偏，长度 = obs.size()）
  ///   @param predictors   可选：逐观测的预报因子；为空时用默认推断
  void update(const ObsSpace& obs, const std::vector<Real>& innovations,
              const std::vector<BiasPredictor>* predictors = nullptr);

  const std::vector<BiasCoefficientSet>& coefficients() const noexcept { return sets_; }
  void set_coefficients(ObsType t, int channel, const std::vector<Real>& beta);
  /// 全部归零（诊断/测试用）
  void reset_coefficients();

  Size n_groups() const noexcept { return sets_.size(); }
  std::string describe() const;

 private:
  BiasCorrectionConfig cfg_{};
  std::vector<BiasCoefficientSet> sets_;
};

}  // namespace vibe::obs
