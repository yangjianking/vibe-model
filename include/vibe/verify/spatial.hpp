#pragma once
/// @file spatial.hpp
/// @brief 空间/邻域检验：FSS、模糊检验、双惩罚诊断、DCT 尺度分解。
///
/// 动机（[E7] Roberts & Lean 2008；[E8] Ebert 2008；[E13] Casati et al. 2008）
/// ------------------------------------------------------------------------
/// 高分辨率模式对对流单体的**位移误差**在逐点检验中会被记为"全空 + 全错"
/// （double penalty），即使预报的空间形态与量级都合理。邻域法先在
/// (2r+1)x(2r+1) 窗口内把二值场平滑成"事件分数"，再比较分数场，
/// 从而把"位置略有偏差"与"完全不报"区分开。
///
/// 邻域分数（fractions）
/// ---------------------
///     二值化：I(x) = 1 if field(x) >= threshold else 0
///     分数：  F(x) = (1/|N_r(x)|) sum_{y in N_r(x)} I(y)
///     N_r(x) = { y : |i_y - i_x| <= r, |j_y - j_x| <= r } ∩ 网格
///     |N_r(x)| 为窗口内实际格点数（边界处自动收缩，避免虚报零分数）
///
/// FSS（[E7] 式 (4)，[E14] Mittermaier & Roberts 2010）
/// ---------------------------------------------------
///     FSS = 1 - sum_x (F_f(x) - F_o(x))^2
///               / [ sum_x F_f(x)^2 + sum_x F_o(x)^2 ]
///     分母为 0（两个场全无事件）时返回 kNaN（未定义）。
///
///     极限性质：
///       r -> 0  : F 退化为 I，FSS = 1 - (M+FA)/(2H+M+FA) = 2H/(2H+M+FA)
///                 （逐点"点数比"；注意这与 CSI = H/(H+M+FA) 不同，[E7] 第 3 节）
///       r -> oo : 两场分数都变成常数（域平均覆盖率）a、b，
///                 FSS -> 1 - (a-b)^2/(a^2+b^2)；
///                 当覆盖率相等（a = b）时趋于 1。
///
/// 实现
/// ----
///   邻域和用 2D 积分图（summed-area table）一次前缀和 + O(1) 查询，
///   每个半径的总代价 O(nx*ny)，与 r 无关（朴素实现为 O(nx*ny*r^2)）。
///
/// 模糊检验（[E8]）
/// ----------------
///   在邻域尺度上对"平滑后的分数场"再阈值化，得到邻域 POD/FAR/CSI/ETS，
///   以及邻域概率场；最小可分辨尺度 = FSS 首次达到 fss_target（默认 0.5）
///   的邻域半径（换算为物理尺度 2r*dx）。
///
/// 双惩罚诊断（[E8][E13]）
/// ----------------------
///   在整数位移搜索空间内找 MSE 最小的分量：
///     (dx*, dy*) = argmin_{|dx|,|dy| <= s} (1/N) sum (f(x+d) - o(x))^2
///     幅度误差   amplitude_error   = mean(f) - mean(o)                 （条件偏差）
///     位移误差   displacement_error = sqrt( max(MSE* - amp^2, 0) )
///     总误差     mse_total
///     双惩罚指数 dpi = mse_total / max(MSE*, eps) >= 1
///   其中 MSE* 为最优位移下的 MSE。dpi 接近 1 表明误差主要来自位置而非量级。
///
/// 尺度分解（可选，简化 2D DCT-II）
/// --------------------------------
///     正交 DCT-II 把场分解为 cos 基，能量按径向波数分带，
///     用于给出"误差能量谱"。正反变换满足 idct2(dct2(x)) = x（机器精度）。
///
/// 复杂度：FSS 单半径 O(nx*ny)；FSS 全尺度 O(nx*ny * r_max)；
///         双惩罚 O(nx*ny * (2s+1)^2)；DCT2 O(nx*ny*(nx+ny))。
///
/// 文献：[E7] Roberts & Lean (2008)；[E8] Ebert (2008)；[E13] Casati et al. (2008)；
///       [E14] Mittermaier & Roberts (2010)。

#include <cstddef>
#include <string>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/common/types.hpp"
#include "vibe/verify/scores.hpp"

namespace vibe::verify {

// ---------------------------------------------------------------------------
// 邻域分数与 FSS
// ---------------------------------------------------------------------------

/// 邻域事件分数场（行主序，长度 nx*ny）。O(nx*ny)，与 radius 无关。
///
/// @param field     扁平场（行主序：index = i*ny + j，i 沿 x，j 沿 y）
/// @param nx, ny     网格尺寸
/// @param threshold 事件阈值（field >= threshold 记为 1）
/// @param radius    邻域半径 r >= 0
/// @throws DimensionError 当 field.size() != nx*ny 或 nx/ny <= 0
std::vector<Real> neighborhood_fractions(const std::vector<Real>& field, int nx, int ny,
                                         Real threshold, int radius);

/// 邻域法 FSS（架构契约签名，冻结）。
///
/// @return 1 - sum (F_f - F_o)^2 / (sum F_f^2 + sum F_o^2)；分母为 0 时返回 kNaN
/// @throws DimensionError 两场长度不等于 nx*ny
///
/// 复杂度 O(nx*ny)（积分图）。
Real fractions_skill_score(const std::vector<Real>& f, const std::vector<Real>& o,
                           int nx, int ny, Real threshold, int radius);

/// 邻域法评分的多尺度结果点。
struct FssPoint {
  int radius = 0;      ///< 邻域半径（格点数）
  Real fss = kNaN;     ///< 该尺度的 FSS
  Real scale = kNaN;   ///< 物理尺度 2*radius*dx（dx 为网格距，缺省 1）
};

/// 给定半径列表的 FSS 曲线。O(nx*ny * |radii|)。
std::vector<FssPoint> fss_curve(const std::vector<Real>& f, const std::vector<Real>& o,
                                int nx, int ny, Real threshold,
                                const std::vector<int>& radii, Real dx = Real(1));

/// 多尺度 FSS：返回 radius = 0,1,...,max_radius 的 FSS 值。
/// 元素 k 对应 radius = k；radius=0 为逐点点数比，radius 覆盖全域时趋近 1。
/// 复杂度 O(nx*ny * max_radius)。
std::vector<Real> fss_vs_scale(const std::vector<Real>& f, const std::vector<Real>& o,
                               int nx, int ny, Real threshold, int max_radius);

/// 最小可分辨尺度：首个使 FSS >= fss_target 的半径（换算为物理尺度 2r*dx）。
/// 若 max_radius 内始终达不到目标，返回 kNaN。O(nx*ny * max_radius)。
Real minimum_resolvable_scale(const std::vector<Real>& f, const std::vector<Real>& o,
                              int nx, int ny, Real threshold, int max_radius,
                              Real fss_target = Real(0.5), Real dx = Real(1));

// ---------------------------------------------------------------------------
// 模糊检验
// ---------------------------------------------------------------------------

/// 邻域（模糊）检验评分。所有分数基于半径 radius 的邻域分数场。
struct FuzzyScores {
  int radius = 0;
  Real neighborhood_pod = kNaN;   ///< 邻域 POD：平滑预报 > 0 且观测有事件
  Real neighborhood_far = kNaN;   ///< 邻域 FAR
  Real neighborhood_csi = kNaN;   ///< 邻域 CSI
  Real neighborhood_ets = kNaN;   ///< 邻域 ETS（随机命中订正）
  Real neighborhood_bias = kNaN;  ///< 邻域频率偏差
  Real mean_fraction_f = kNaN;    ///< 预报场平均事件分数
  Real mean_fraction_o = kNaN;    ///< 观测场平均事件分数
  Real fss = kNaN;                ///< 同一尺度的 FSS
  std::size_t n = 0;
};

/// 计算指定半径的模糊（邻域）检验评分。O(nx*ny)。
FuzzyScores fuzzy_scores(const std::vector<Real>& f, const std::vector<Real>& o,
                         int nx, int ny, Real threshold, int radius);

/// 邻域概率场：与 neighborhood_fractions 相同，但语义为"该点邻域内出现事件的概率"。
/// 便于与集合概率一起画可靠性图。O(nx*ny)。
std::vector<Real> neighborhood_probability(const std::vector<Real>& field, int nx, int ny,
                                           Real threshold, int radius);

// ---------------------------------------------------------------------------
// 双惩罚诊断
// ---------------------------------------------------------------------------

/// 最优整数位移搜索结果。
struct DisplacementEstimate {
  int dx = 0;
  int dy = 0;
  Real mse_at_best = kNaN;   ///< 最优位移下的 MSE
  Real mse_zero = kNaN;      ///< 无位移（dx=dy=0）时的 MSE
  Real overlap_fraction = kNaN;  ///< 事件重叠率（二值化后 Jaccard 指数）
};

/// 在全场范围（或 max_shift 内）搜索使 MSE 最小的整数位移。
/// @param max_shift <= 0 表示搜索到全网格范围。O(nx*ny*(2s+1)^2)。
DisplacementEstimate best_displacement(const std::vector<Real>& f,
                                       const std::vector<Real>& o,
                                       int nx, int ny, int max_shift = -1);

/// 双惩罚误差分解结果。
struct DoublePenaltyDiagnosis {
  Real mse_total = kNaN;         ///< (1/N) sum (f - o)^2
  Real mse_displaced = kNaN;     ///< 最优位移后的 MSE
  Real amplitude_error = kNaN;   ///< mean(f) - mean(o)
  Real displacement_error = kNaN;///< sqrt(max(MSE* - amplitude^2, 0))
  Real double_penalty_index = kNaN;  ///< MSE_total / MSE* >= 1
  Real correlation = kNaN;       ///< 零位移 Pearson 相关
  DisplacementEstimate shift;    ///< 最优位移
  std::size_t n = 0;
};

/// 双惩罚（位移/幅度）误差分解。见文件头公式。O(nx*ny*(2s+1)^2)。
DoublePenaltyDiagnosis double_penalty_diagnosis(const std::vector<Real>& f,
                                                const std::vector<Real>& o,
                                                int nx, int ny, int max_shift = -1);

// ---------------------------------------------------------------------------
// DCT 尺度分解（简化实现）
// ---------------------------------------------------------------------------

/// 正交 2D DCT-II：X[k,l] = sum_{i,j} x[i,j] c_k c_l
///                        cos(pi k (2i+1)/(2nx)) cos(pi l (2j+1)/(2ny))
/// 其中 c_k = sqrt(1/nx) (k=0), sqrt(2/nx) (k>0)。O(nx*ny*(nx+ny))。
void dct2(const std::vector<Real>& in, int nx, int ny, std::vector<Real>& out);

/// 正交 2D DCT-III（逆变换），满足 idct2(dct2(x)) = x。O(nx*ny*(nx+ny))。
void idct2(const std::vector<Real>& in, int nx, int ny, std::vector<Real>& out);

/// 尺度能量谱。
struct ScaleSpectrum {
  int nx = 0;
  int ny = 0;
  std::vector<Real> band_wavenumber;  ///< 每个带的平均归一化径向波数
  std::vector<Real> band_energy;      ///< 每个带的能量（DCT 系数平方和）
  Real total_energy = kNaN;
  Real residual_energy = kNaN;        ///< 直流分量（k=l=0）能量，反映域平均偏差
};

/// 按归一化径向波数 kappa = sqrt((k/nx)^2 + (l/ny)^2)/sqrt(2) 把
/// DCT 能量分成 n_bands 个等宽带。O(nx*ny*(nx+ny))。
ScaleSpectrum dct_scale_spectrum(const std::vector<Real>& f, int nx, int ny,
                                 int n_bands = 8);

// ---------------------------------------------------------------------------
// 批量接口（架构契约签名，冻结）
// ---------------------------------------------------------------------------

/// 邻域法（空间）评分。
///
/// @return 填充 fss/n；n = nx*ny
/// @throws DimensionError 尺寸不一致
///
/// 复杂度 O(nx*ny)。
Scores compute_fractional_skill(const std::vector<Real>& f, const std::vector<Real>& o,
                                int nx, int ny, Real threshold,
                                int neighborhood_radius);

}  // namespace vibe::verify
