#pragma once
/// @file obs_operators.hpp
/// @brief 各类常规与卫星观测算子的具体声明。
///
/// 每个算子都实现 `apply / applyTL / applyAD`，并在头文件注释里给出
/// 具体的观测方程与线性化形式。
///
/// 文献：[O1][O3][O4][O5][O6][O7][O8]。

#include <memory>
#include <vector>

#include "vibe/obs/obs_operator.hpp"
#include "vibe/obs/radiative_transfer.hpp"

namespace vibe::obs {

/// 探空/飞机/廓线：u, v, T, q, p 在任意高度
///   H_u(x) = u(x_obs),  H_T(x) = T(pi, theta),  H_q(x) = qv
/// 切线性：dT = (dT/dpi) dpi + (dT/dtheta) dtheta，系数冻结在基础态。
class SoundingOperator final : public ObservationOperator {
 public:
  explicit SoundingOperator(const grid::Grid& g);
  const char* name() const noexcept override { return "sounding"; }
  ObsType type() const noexcept override { return ObsType::Radiosonde; }
  void apply(const ModelStateView& x, const ObsSpace& obs,
             std::vector<Real>& y) const override;
  void applyTL(const ModelStateView& x, const dyn::State& dx,
               const ObsSpace& obs, std::vector<Real>& dy) const override;
  void applyAD(const ModelStateView& x, const std::vector<Real>& dy,
               const ObsSpace& obs, dyn::State& dx) const override;

 private:
  const grid::Grid* grid_;
};

/// 地面：2m 温度/湿度、10m 风、地面气压
/// 使用 Monin-Obukhov 相似理论从最低模式层外推到 2m/10m（[P9][P10]）。
class SurfaceOperator final : public ObservationOperator {
 public:
  explicit SurfaceOperator(const grid::Grid& g);
  const char* name() const noexcept override { return "surface"; }
  ObsType type() const noexcept override { return ObsType::Surface; }
  void apply(const ModelStateView& x, const ObsSpace& obs,
             std::vector<Real>& y) const override;
  void applyTL(const ModelStateView& x, const dyn::State& dx,
               const ObsSpace& obs, std::vector<Real>& dy) const override;
  void applyAD(const ModelStateView& x, const std::vector<Real>& dy,
               const ObsSpace& obs, dyn::State& dx) const override;

 private:
  const grid::Grid* grid_;
};

/// 卫星导风 / 散射计：单层风矢量
class WindOperator final : public ObservationOperator {
 public:
  WindOperator(const grid::Grid& g, ObsType t);
  const char* name() const noexcept override { return "wind"; }
  ObsType type() const noexcept override { return type_; }
  void apply(const ModelStateView& x, const ObsSpace& obs,
             std::vector<Real>& y) const override;
  void applyTL(const ModelStateView& x, const dyn::State& dx,
               const ObsSpace& obs, std::vector<Real>& dy) const override;
  void applyAD(const ModelStateView& x, const std::vector<Real>& dy,
               const ObsSpace& obs, dyn::State& dx) const override;

 private:
  const grid::Grid* grid_;
  ObsType type_;
};

/// GNSS 掩星：折射率
///   N = 77.6 p / T + 3.73e5 q_v p / T^2
/// 切线性对该式求导，是典型的"多变量耦合"观测（[O4][O5]）。
class GnssRoOperator final : public ObservationOperator {
 public:
  explicit GnssRoOperator(const grid::Grid& g);
  const char* name() const noexcept override { return "gnssro"; }
  ObsType type() const noexcept override { return ObsType::GnssRo; }
  void apply(const ModelStateView& x, const ObsSpace& obs,
             std::vector<Real>& y) const override;
  void applyTL(const ModelStateView& x, const dyn::State& dx,
               const ObsSpace& obs, std::vector<Real>& dy) const override;
  void applyAD(const ModelStateView& x, const std::vector<Real>& dy,
               const ObsSpace& obs, dyn::State& dx) const override;

  /// 折射率正演：N(p, T, qv)
  static Real refractivity(Real pressure, Real temperature, Real qv);

 private:
  const grid::Grid* grid_;
};

/// 雷达反射率（[O6] Sun & Crook 1997）
class RadarOperator final : public ObservationOperator {
 public:
  explicit RadarOperator(const grid::Grid& g);
  const char* name() const noexcept override { return "radar"; }
  ObsType type() const noexcept override { return ObsType::RadarReflectivity; }
  void apply(const ModelStateView& x, const ObsSpace& obs,
             std::vector<Real>& y) const override;
  void applyTL(const ModelStateView& x, const dyn::State& dx,
               const ObsSpace& obs, std::vector<Real>& dy) const override;
  void applyAD(const ModelStateView& x, const std::vector<Real>& dy,
               const ObsSpace& obs, dyn::State& dx) const override;

 private:
  const grid::Grid* grid_;
};

/// 卫星辐射率（快速辐射传输 [O1][O2]）
class RadianceOperator final : public ObservationOperator {
 public:
  RadianceOperator(const grid::Grid& g, const RadiativeTransfer& rt);
  const char* name() const noexcept override { return "radiance"; }
  ObsType type() const noexcept override { return ObsType::Radiance; }
  void apply(const ModelStateView& x, const ObsSpace& obs,
             std::vector<Real>& y) const override;
  void applyTL(const ModelStateView& x, const dyn::State& dx,
               const ObsSpace& obs, std::vector<Real>& dy) const override;
  void applyAD(const ModelStateView& x, const std::vector<Real>& dy,
               const ObsSpace& obs, dyn::State& dx) const override;

 private:
  const grid::Grid* grid_;
  RadiativeTransfer rt_;
};

}  // namespace vibe::obs
