#pragma once
/// @file physics_driver.hpp
/// @brief 物理过程总驱动：按固定顺序调度辐射、陆面/边界层、微物理、积云与次网格动量通量，
///        并把单列倾向散射回 dyn::PhysicsTendency。
///
/// 调用顺序（与 docs/design/08_physics.md 第 3 节一致）
///   1. 辐射（按 radiation_cadence，两步之间保持上一次加热率 = 加热率时间平流）
///   2. 地面层（Monin-Obukhov）-> 陆面/海面（表皮能量平衡、五层土壤）
///   3. 边界层（YSU 或 MYJ）
///   4. 微物理（Kessler / Thompson）
///   5. 积云（Kain-Fritsch / Grell-Devenyi）
///   6. 次网格动量通量（Smagorinsky 水平扩散 + PBL 动量通量辐合）
///
/// 持久状态（跨时间步）
///   * 地表状态（土壤温度/湿度、表皮温度、粗糙度）：std::vector<SurfaceState>
///   * 湍流动能 e = q^2/2（MYJ 需要）：std::vector<Real>，长度 nx*ny*nz
///   * 上一次辐射加热率（时间平滑）：std::vector<Real>，长度 nx*ny*nz
///
/// 文献：[P1]-[P18]；调度与时间分裂见 [D5][D16]。

#include <memory>
#include <string>
#include <vector>

#include "vibe/config/config.hpp"
#include "vibe/dyn/reference_state.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/dyn/tendency.hpp"
#include "vibe/grid/geometry.hpp"
#include "vibe/physics/cumulus.hpp"
#include "vibe/physics/microphysics.hpp"
#include "vibe/physics/pbl.hpp"
#include "vibe/physics/physics_types.hpp"
#include "vibe/physics/radiation.hpp"
#include "vibe/physics/surface.hpp"

namespace vibe::physics {

// ---------------------------------------------------------------------------
/// 由模式配置构造物理选项。
///   * 方案字符串 -> 枚举（microphysics/radiation/pbl/surface/cumulus）
///   * radiation_cadence 单位为**秒**，转换为步数 round(radiation_cadence/dt)
///   * 使用 use_cloud_fraction / co2_ppm / aerosol_optical_depth / soil_layers
///   * 由 time.start_time 解析年积日与 UTC 小时（用于太阳几何）
/// 复杂度 O(1)。
PhysicsOptions options_from_config(const config::ModelConfig& cfg);

/// 把预报状态的一列填入 PhysicsColumn（z 由网格几何给出，p 由参考态 Exner 得到）
void state_to_column(const dyn::State& s, const dyn::ReferenceState& ref, const grid::Grid& g,
                     Int i, Int j, PhysicsColumn& col);

// ===========================================================================
/// 物理过程总驱动
class PhysicsDriver {
 public:
  PhysicsDriver(const PhysicsOptions& opt, const grid::Grid& g);

  // -------------------------------------------------------------------------
  /// **主入口**：把 dt 内的物理倾向累加到 out（不重置 out）。
  void step(dyn::State& s, const grid::Grid& g, const dyn::ReferenceState& ref,
            dyn::PhysicsTendency& out, Real dt);

  /// 单列调试入口（供单元测试与个例诊断）
  void step_column(Int i, Int j, dyn::State& s, const grid::Grid& g,
                   const dyn::ReferenceState& ref, dyn::PhysicsTendency& out, Real dt);

  // ---- 调度 ----
  const PhysicsOptions& options() const noexcept { return opt_; }
  void set_options(const PhysicsOptions& opt);
  void set_step_index(int step) noexcept { step_index_ = step; }
  int step_index() const noexcept { return step_index_; }
  /// 本步是否需要调用辐射（step % radiation_cadence == 0）
  bool call_radiation_step() const noexcept;
  Int radiation_cadence_steps() const noexcept { return opt_.radiation_cadence; }
  /// 由 time.dt 与 radiation_cadence(秒) 推导的步数

  // ---- 诊断 ----
  const PhysicsDiagnostics& diagnostics() const noexcept { return diag_; }
  PhysicsDiagnostics& diagnostics() noexcept { return diag_; }
  void reset_diagnostics() { diag_.reset(); }
  /// 物理 CFL 数（水平/落速/扩散）
  PhysicsCfl cfl_physics(const dyn::State& s, const dyn::ReferenceState& ref, Real dt) const;
  /// 稳定诊断文字报告
  std::string stability_report() const;
  /// 上一次 step 的最大 CFL
  const PhysicsCfl& last_cfl() const noexcept { return cfl_last_; }

  // ---- 子方案访问（可为 nullptr） ----
  MicrophysicsBase* microphysics() noexcept { return microphysics_.get(); }
  RadiationDriver* radiation() noexcept { return radiation_.get(); }
  PblBase* pbl() noexcept { return pbl_.get(); }
  MoninObukhov* surface_layer() noexcept { return surface_layer_.get(); }
  LandSurface* land_surface() noexcept { return land_.get(); }
  SeaSurface* sea_surface() noexcept { return sea_.get(); }
  CumulusBase* cumulus() noexcept { return cumulus_.get(); }

  // ---- 持久状态 ----
  std::vector<SurfaceState>& surface_states() noexcept { return surface_; }
  const std::vector<Real>& tke_storage() const noexcept { return tke_; }
  std::vector<Real>& tke_storage() noexcept { return tke_; }

  /// 方案摘要（日志用）
  std::string describe() const;

 private:
  void ensure_allocated(const grid::Grid& g);
  /// 单列物理（内部实现，被 step 与 step_column 共用）
  void column_physics(Int i, Int j, dyn::State& s, const grid::Grid& g,
                      const dyn::ReferenceState& ref, Real dt, Real cos_zenith,
                      Real sun_earth_factor, bool call_radiation, ColumnTendency& td,
                      SurfaceFluxes& flx_out);
  /// 次网格动量通量（Smagorinsky 水平扩散），i/j 为物理列索引
  void subgrid_momentum_tendency(const dyn::State& s, const grid::Grid& g, Int i, Int j,
                                 const PhysicsColumn& col, ColumnTendency& td) const;
  /// 把单列倾向散射到 PhysicsTendency 的 (i,j) 列
  void scatter(Int i, Int j, const ColumnTendency& td, dyn::PhysicsTendency& out) const;

  PhysicsOptions opt_;
  std::unique_ptr<MicrophysicsBase> microphysics_;
  std::unique_ptr<RadiationDriver> radiation_;
  std::unique_ptr<PblBase> pbl_;
  std::unique_ptr<MoninObukhov> surface_layer_;
  std::unique_ptr<LandSurface> land_;
  std::unique_ptr<SeaSurface> sea_;
  std::unique_ptr<CumulusBase> cumulus_;

  std::vector<SurfaceState> surface_;   ///< nx*ny
  const grid::Grid* grid_ = nullptr;    ///< 网格（CFL 诊断用）
  std::vector<Real> tke_;               ///< nx*ny*nz（MYJ 的 e）
  std::vector<Real> rad_theta_tend_;    ///< nx*ny*nz（辐射加热率时间平滑）
  PhysicsDiagnostics diag_;
  Int nx_ = 0, ny_ = 0, nz_ = 0;
  int step_index_ = 0;
  PhysicsCfl cfl_last_{};
  mutable std::vector<Real> lwp_work_, iwp_work_;  ///< 工作数组
};

// ---------------------------------------------------------------------------
/// **driver 层唯一的构造入口**：依据配置创建物理过程驱动；
/// 未启用任何方案时返回一个空实现（step 为 no-op，仅做诊断与守恒检查）。
/// 复杂度 O(1)（不分配网格）
std::unique_ptr<PhysicsDriver> make_physics_driver(const grid::Grid& g,
                                                   const config::ModelConfig& cfg);

}  // namespace vibe::physics
