#pragma once
/// @file cumulus.hpp
/// @brief 积云对流参数化：Kain-Fritsch 质量通量方案 [P11] 与 Grell-Devenyi 集合框架 [P12]。
///
/// 框架
/// ----
///   1. 触发：在最低 ~300 hPa 内选取湿静力能最大的源层，抬升气块（干燥绝热 + 湿绝热），
///      要求 CAPE > CAPE_min、云厚 > 4 km 且气块能够到达自由对流高度。
///   2. 卷入/卷出 plume 模型（[P11] 式 (1)-(4)，含冰相）：
///         d m/dz = (eps - delta) m,   theta_e 守恒的饱和抬升
///   3. CAPE 消耗闭合（[P11] 第 3 节）：
///         m_b = rho_b CAPE / (tau Integral (g/(cp theta_ve)) |d s_v/dz| dz)
///   4. 环境响应：补偿下沉 + 卷出 + 下沉气流与再蒸发
///         dtheta/dt = -(1/rho) m_b dtheta_p/dz + D (theta_p - theta_env)/(rho dz)
///   5. 对流降水： P = eps_p L，其中 L 为整列水物质净减少率，eps_p 为降水效率
///      （按陆/海取值），未降水部分 (1-eps_p) L 作为下沉气流再蒸发返回云下层，
///      从而保证水物质收支闭合。
///
/// 文献：[P11] Kain & Fritsch (1990)；[P12] Grell & Devenyi (2002)。

#include <memory>
#include <string>
#include <vector>

#include "vibe/common/constants.hpp"
#include "vibe/common/types.hpp"
#include "vibe/physics/physics_types.hpp"

namespace vibe::physics {

// ---------------------------------------------------------------------------
/// 积云对流诊断
struct ConvectionDiagnostics {
  Real cloud_base_height = Real(0);   ///< m
  Real cloud_top_height = Real(0);    ///< m
  Real cape = Real(0);                ///< J/kg
  Real cin = Real(0);                 ///< J/kg
  Real mass_flux_base = Real(0);      ///< kg/(m^2 s)
  Real updraft_velocity = Real(0);    ///< 云底垂直速度估计 m/s
  Real precipitation = Real(0);       ///< mm/s
  Real downdraft_mass_flux = Real(0);
  Real heating_max = Real(0);         ///< K/s
  Real moistening_max = Real(0);      ///< kg/kg/s
  Real water_residual = Real(0);      ///< 水物质相对残差
  Int source_level = -1;
  Int cloud_base_level = -1;
  Int cloud_top_level = -1;
  Int lfc = -1, el = -1;
  bool triggered = false;
};

/// 上升气块（plume）廓线：长度均为 nz
struct PlumeProfile {
  std::vector<Real> mass_flux;      ///< 归一化质量通量 m(z)/m_b
  std::vector<Real> theta_e;        ///< 气块相当位温 (K)
  std::vector<Real> theta;          ///< 气块位温 (K)
  std::vector<Real> qv;             ///< 气块水汽 (kg/kg)
  std::vector<Real> qc;             ///< 气块凝结态水 (kg/kg)
  std::vector<Real> tv;             ///< 气块虚温 (K)
  std::vector<Real> buoyancy;       ///< 浮力加速度 (m/s^2)
  std::vector<Real> condensate;     ///< 该层净凝结量 (kg/kg)
  std::vector<Real> detrain;        ///< 该层卷出比例 (1/m)
  Int cloud_base_level = -1;
  Int cloud_top_level = -1;
  Int source_level = -1;
  Real cape = Real(0);
  Real cin = Real(0);
  bool valid = false;

  void resize(Int nz) {
    const Size n = static_cast<Size>(nz > 0 ? nz : 0);
    mass_flux.assign(n, Real(0));
    theta_e.assign(n, Real(0));
    theta.assign(n, Real(0));
    qv.assign(n, Real(0));
    qc.assign(n, Real(0));
    tv.assign(n, Real(0));
    buoyancy.assign(n, Real(0));
    condensate.assign(n, Real(0));
    detrain.assign(n, Real(0));
  }
  Int nz() const noexcept { return static_cast<Int>(mass_flux.size()); }
};

// ===========================================================================
/// 积云方案抽象接口
class CumulusBase {
 public:
  virtual ~CumulusBase() = default;
  virtual CumulusScheme scheme() const noexcept = 0;
  virtual const char* name() const noexcept = 0;

  // -------------------------------------------------------------------------
  /// 单列对流参数化
  /// @return 对流降水率 (mm/s)；加热/增湿/动量倾向累加到 tend
  virtual Real step_column(PhysicsColumn& col, const SurfaceState& sfc, Real dt,
                           ColumnTendency& tend, PhysicsDiagnostics& diag) = 0;

  virtual std::string describe() const;
  /// 上一次调用的对流诊断
  const ConvectionDiagnostics& last_diagnostics() const noexcept { return last_; }

 protected:
  explicit CumulusBase(const PhysicsOptions& opt) : opt_(opt) {}
  const PhysicsOptions& opt_;
  mutable ConvectionDiagnostics last_;
};

// ===========================================================================
/// Kain-Fritsch 质量通量方案 [P11]
/// ===========================================================================
class KainFritschCumulus final : public CumulusBase {
 public:
  explicit KainFritschCumulus(const PhysicsOptions& opt) : CumulusBase(opt) {}

  CumulusScheme scheme() const noexcept override { return CumulusScheme::KainFritsch; }
  const char* name() const noexcept override { return "Kain-Fritsch 质量通量 [P11]"; }

  Real step_column(PhysicsColumn& col, const SurfaceState& sfc, Real dt, ColumnTendency& tend,
                   PhysicsDiagnostics& diag) override;

  // ---- 各步骤（公开以便单元测试与文档对照） ----
  /// 触发判据：源层选择 + 气块抬升 + CAPE/云厚检查
  bool trigger(const PhysicsColumn& col, const SurfaceState& sfc, ParcelAscent& ascent,
               Int& source_level) const;

  /// 卷入/卷出 plume 积分
  ///   d m/dz = (eps - delta) m
  ///   theta_e 守恒的饱和抬升（Newton 反解 T）
  ///   浮力 b = g (Tv_p - Tv_e)/Tv_e；b < 0 持续两层则云顶
  void plume_ascent(const PhysicsColumn& col, Int source_level, Int cloud_base, PlumeProfile& plume) const;

  /// CAPE 消耗闭合
  ///   S = Integral_{z_b}^{z_t} (g/(cp theta_ve)) |d s_v/dz| dz
  ///   m_b = rho_b CAPE / (tau S)
  Real closure_mass_flux(const PhysicsColumn& col, const PlumeProfile& plume) const noexcept;

  /// 降水效率：浅对流 0.2，深对流按陆/海取值（[P11] 第 4 节）
  Real precipitation_efficiency(Real cloud_depth, bool over_land) const noexcept;

  /// 云层内最小相当位温所在层（下沉气流起始层）
  Int downdraft_origin(const PhysicsColumn& col, const PlumeProfile& plume) const noexcept;

  std::string describe() const override;
};

// ===========================================================================
/// Grell-Devenyi 集合式框架 [P12]
/// ===========================================================================
class GrellDevenyiEnsemble final : public CumulusBase {
 public:
  explicit GrellDevenyiEnsemble(const PhysicsOptions& opt) : CumulusBase(opt) {}

  CumulusScheme scheme() const noexcept override { return CumulusScheme::GrellDevenyi; }
  const char* name() const noexcept override { return "Grell-Devenyi 集合框架 [P12]"; }

  Real step_column(PhysicsColumn& col, const SurfaceState& sfc, Real dt, ColumnTendency& tend,
                   PhysicsDiagnostics& diag) override;

  /// 成员扰动（[P12] 表 1）：卷入率、卷出率、闭合时间、降水效率
  void perturbed_options(Int member, PhysicsOptions& out) const;

  /// 成员数（受 gd_ensemble_size 与 kMaxMembers 限制）
  Int member_count() const noexcept;

  static constexpr Int kMaxMembers = 32;

  std::string describe() const override;
};

/// 积云方案工厂：None -> nullptr
std::unique_ptr<CumulusBase> make_cumulus(CumulusScheme scheme, const PhysicsOptions& opt);

}  // namespace vibe::physics
