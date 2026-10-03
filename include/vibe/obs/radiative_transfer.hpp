#pragma once
/// @file radiative_transfer.hpp
/// @brief 面向数据同化的快速辐射传输算子及其雅可比。
///
/// 目的
/// ----
/// 卫星辐射率是当前同化系统中信息量最大、也最需要伴随的观测。
/// 本模块提供一个**可读的**快速辐射传输参数化（RTTOV 的教学级简化），
/// 它的价值在于：
///   1. 显式的、可微的透射率参数化，便于写切线性与伴随；
///   2. 完整的雅可比 `dBT/dT`、`dBT/dq`、`dBT/dPs`、`dBT/dO3`；
///   3. 可直接插入 4D-Var 的观测算子。
///
/// 模型
/// ----
/// 垂直分层（n_layers = 模式层数），考虑 n_channels 个通道。
/// 对每个通道 c：
///   * 预报因子 `predictors` 由层平均温度、水汽、臭氧、天顶角以及它们
///     与层间差分的乘积构成（RTTOV 的 "predictor" 体系，[O1]）；
///   * 光学厚度
///         tau_c = sum_k sum_j a_{c,k,j} * X_{k,j}
///     其中 a 为回归系数（本实现用解析生成的教学系数）；
///   * 透射率 transmittance = exp(-tau)；
///   * 对红外通道使用逐层积分形式的辐射传输方程
///         I = (1 - t_s) B(T_s) + sum_k B(T_k) (t_{k+1} - t_k)
///     其中 B 为 Planck 函数（由波数中心与温度解析给出）；
///   * 对微波通道使用类似的吸收模型（水汽与氧气吸收线）。
///
/// 雅可比
/// ------
/// 对回归形式，`dI/dX_{k,j}` 可以解析写出，避免了自动微分的开销；
/// 这也是同类系统中"Jacobian 计算"往往占据辐射传输模块大部分代码的原因。
///
/// 文献：[O1] Saunders et al. (1999)；[O2] RTTOV v11 Users Guide；
///       [O3] Eyre (1990)；[O10] Rodgers (2000) 第 9 章。

#include <string>
#include <vector>

#include "vibe/common/types.hpp"

namespace vibe::obs {

/// 通道的谱域
enum class SpectralBand { Infrared, Microwave, Visible, Count };

/// 单个通道定义
struct Channel {
  int id = 0;
  double wavenumber = Real(0);     ///< cm^-1（红外）或 GHz 的等效波数
  SpectralBand band = SpectralBand::Infrared;
  Real solar_contribution = Real(0);   ///< 短波贡献（可见/近红外）
  bool surface_sensitive = false;
  int  pe = 0;                         ///< 峰值能量层（诊断）
};

/// 一层的大气剖面
struct LayerProfile {
  Real pressure = Real(0);       ///< Pa
  Real temperature = Real(0);    ///< K
  Real qv = Real(0);             ///< kg/kg
  Real ozone = Real(0);          ///< kg/kg
  Real cloud_water = Real(0);
  Real cloud_ice = Real(0);
  Real zenith_angle_deg = Real(0);
  Real surface_temperature = Real(0);
  Real surface_pressure = Real(0);
  Real surface_emissivity = Real(1);
  Real skin_temperature = Real(0);
};

/// 单通道正演结果
struct RadianceResult {
  Real brightness_temperature = Real(0);  ///< K
  Real radiance = Real(0);                ///< mW/(m^2 sr cm^-1)
  Real transmittance_to_surface = Real(0);
  std::vector<Real> layer_optical_depth;  ///< 长度 n_layers
  std::vector<Real> weighting_function;   ///< d t / d ln p，长度 n_layers
};

/// 雅可比：对每层每个预报变量
struct RadianceJacobian {
  std::vector<Real> temperature;   ///< dBT/dT_k  (n_layers)
  std::vector<Real> humidity;      ///< dBT/dqv_k
  std::vector<Real> ozone;
  Real surface_temperature = Real(0);
  Real surface_emissivity = Real(0);
};

/// 快速辐射传输
class RadiativeTransfer {
 public:
  RadiativeTransfer() = default;
  explicit RadiativeTransfer(const std::vector<Channel>& channels);

  /// 内置的典型通道组（HIRS/AMSU-A/MHS/SEVIRI 采样）
  static RadiativeTransfer builtin_channels(const std::string& set_name = "amsua");

  Size n_channels() const noexcept { return channels_.size(); }
  const std::vector<Channel>& channels() const noexcept { return channels_; }
  const Channel& channel(int c) const { return channels_[static_cast<Size>(c)]; }

  /// 某个通道的正演
  RadianceResult forward(int channel, const LayerProfile& profile) const;

  /// 某个通道的雅可比（解析）
  RadianceJacobian jacobian(int channel, const LayerProfile& profile) const;

  /// 计算通道的光学厚度与透射率
  std::vector<Real> optical_depth(int channel, const LayerProfile& profile) const;

  /// 设定/读取回归系数（供训练或替换）
  void set_coefficients(const std::vector<std::vector<Real>>& coefs);
  const std::vector<std::vector<Real>>& coefficients() const noexcept { return coefs_; }

  /// 描述（日志）
  std::string describe() const;

 private:
  /// 预报因子构造
  std::vector<Real> predictors(const LayerProfile& profile, int channel) const;
  /// Planck 函数（由等效波数与温度算辐射率）
  Real planck(double wavenumber, Real temperature) const;
  /// 逆 Planck：由辐射率反算亮温
  Real inverse_planck(double wavenumber, Real radiance) const;

  std::vector<Channel> channels_;
  std::vector<std::vector<Real>> coefs_;   ///< [channel][predictor]
};

}  // namespace vibe::obs
