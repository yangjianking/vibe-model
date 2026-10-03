#pragma once
/// @file tangent_linear.hpp
/// @brief 切线性模式（TL）与伴随模式（AD）。
///
/// 离散伴随的构造规则（[A1] Giering & Kaminski 1998）
/// ------------------------------------------------
///   1. 把前向代码分解为**基本语句**序列；
///   2. TL 代码：对每个语句做前向微分，用基础态保存的中间量；
///   3. AD 代码：把 TL 语句**逆序**，每个语句替换为其转置；
///   4. AD 需要基础态在每个时间步的中间量（trajectory），要么存储、
///      要么重算（checkpointing）。本实现采用**存储 + 分层检查点**。
///
/// 关键实现原则
/// ------------
///   * **对偶性**：每个 TL 例程都有唯一的 AD 例程，命名后缀 `_tl` / `_ad`；
///   * **无状态**：TL/AD 例程只读基础态，把结果写入显式输出；
///   * **线性性**：TL/AD 代码中不允许出现非线性运算（除了对基础态的读取）；
///   * **守恒性**：AD 必须保持 TL 的转置结构，因此不能"顺手"加平滑。
///
/// 不连续过程
/// ----------
/// 饱和调整、对流触发、云顶夹卷等不连续过程在基础态处不可微。
/// 处理方式（[A6][A8]）：
///   * 冻结开关（frozen switch）：用基础态的开关状态，导数为 0 或 1；
///   * 光滑化：用连续函数近似阶跃（会引入偏差）；
///   * 简化物理（simplified physics）：在 TL/AD 中只用部分物理过程。
///
/// 文献：[A1][A2][A3][A4][A5][A6][A8]。

#include <memory>
#include <string>
#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/da/da_types.hpp"
#include "vibe/dyn/equations.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/grid/geometry.hpp"
#include "vibe/time/integrator.hpp"

namespace vibe::config { struct ModelConfig; }

namespace vibe::da {

/// TL/AD 的物理过程处理方式
enum class PhysicsLinearization {
  Adiabatic,          ///< 完全不含物理（标准选择，[A5]）
  Simplified,         ///< 只含可微的简化物理（[A8]）
  FrozenSwitch,       ///< 含物理但用冻结开关（[A6]）
  FullPhysics,        ///< 含全部物理（一般不推荐）
};

const char* to_string(PhysicsLinearization p) noexcept;

/// 检查点策略
enum class CheckpointStrategy {
  StoreAll,      ///< 存储全部时间步（内存换时间）
  Revolve,       ///< Griewank-Walther revolve 算法（[A2] 第 12 章）
  Multilevel,    ///< 分层检查点
  Recompute      ///< 完全重算（时间换内存）
};

/// 一条模式轨迹（基础态 + 中间量）
class Trajectory {
 public:
  Trajectory() = default;
  void reserve(Size n_steps, const dyn::State& prototype);
  void push_back(const dyn::State& s, const dyn::Tendency& slow,
                 const dyn::Tendency& acoustic);
  Size size() const noexcept { return states_.size(); }
  const dyn::State& state(Size i) const { return states_[i]; }
  const dyn::Tendency& slow(Size i) const { return slow_[i]; }
  const dyn::Tendency& acoustic(Size i) const { return acoustic_[i]; }
  void clear();
  /// 内存占用（字节）
  Size memory_bytes() const;

 private:
  std::vector<dyn::State> states_;
  std::vector<dyn::Tendency> slow_;
  std::vector<dyn::Tendency> acoustic_;
};

/// 切线性模式
class TangentLinearModel {
 public:
  TangentLinearModel(const grid::Grid& g, const dyn::ReferenceState& ref,
                     const config::ModelConfig& cfg,
                     const config::DaConfig& da_cfg);

  /// 从基础态 xb 出发，推进 dx 共 dt 秒（n_steps 步）
  void propagate(const dyn::State& xb, const dyn::State& dx0, Real dt,
                 int n_steps, dyn::State& dx_out);

  /// 单步切线性推进
  void step(const dyn::State& xb, const dyn::State& dx, Real dt,
            dyn::State& dx_out);

  /// 记录轨迹（供伴随使用）
  void record_trajectory(const dyn::State& xb, Real dt, int n_steps);

  /// 轨迹访问
  const Trajectory& trajectory() const noexcept { return traj_; }
  Trajectory& trajectory() noexcept { return traj_; }

  /// 配置
  void set_physics_linearization(PhysicsLinearization p) noexcept { physics_ = p; }
  PhysicsLinearization physics_linearization() const noexcept { return physics_; }
  void set_checkpointing(CheckpointStrategy c) noexcept { checkpoint_ = c; }

  /// 线性稳定性检查：||dx|| 是否随时间指数增长超过阈值
  Real linear_growth_rate(const dyn::State& xb, const dyn::State& dx, Real dt,
                          int n_steps) const;

 private:
  const grid::Grid* grid_;
  const dyn::ReferenceState* ref_;
  dyn::Equations* eq_;
  Trajectory traj_;
  PhysicsLinearization physics_ = PhysicsLinearization::Adiabatic;
  CheckpointStrategy checkpoint_ = CheckpointStrategy::StoreAll;
  timeint::StepContext ctx_{};
};

/// 伴随模式
class AdjointModel {
 public:
  AdjointModel(const grid::Grid& g, const dyn::ReferenceState& ref,
               const config::ModelConfig& cfg, const config::DaConfig& da_cfg);

  /// 由末端扰动 dy 反向传播到初始时刻，得到初始扰动伴随 dx0
  void propagate(const dyn::State& xb_trajectory_end, const dyn::State& dy,
                 Real dt, int n_steps, dyn::State& dx0_adjoint);

  /// 单步伴随
  void step_adjoint(const dyn::State& xb, const dyn::State& dx_next,
                    Real dt, dyn::State& dx_prev);

  /// 使用已记录的轨迹（推荐）
  void propagate_with_trajectory(const Trajectory& traj, const dyn::State& dy,
                                 dyn::State& dx0_adjoint);

  void set_physics_linearization(PhysicsLinearization p) noexcept { physics_ = p; }

 private:
  const grid::Grid* grid_;
  const dyn::ReferenceState* ref_;
  dyn::Equations* eq_;
  Trajectory own_traj_;
  PhysicsLinearization physics_ = PhysicsLinearization::Adiabatic;
  timeint::StepContext ctx_{};
};

/// 点积检验工具（[A3] Sirkes & Tziperman 1997）
struct AdjointCheckResult {
  Real lhs = Real(0);            ///< < M dx, dy >
  Real rhs = Real(0);            ///< < dx, M^T dy >
  Real relative_error = Real(0);
  bool passed = false;
  int  n_steps = 0;
  Real dt = Real(0);
  std::string describe() const;
};

/// 对时间推进算子做伴随点积检验
AdjointCheckResult check_adjoint(const TangentLinearModel& tl,
                                 const AdjointModel& ad,
                                 const dyn::State& xb, Real dt, int n_steps,
                                 unsigned seed = 1, Real tolerance = Real(1e-8));

/// 切线性一致性检验（有限差分）：比较 TL 结果与 (M(x+eps dx) - M(x))/eps
struct TangentCheckResult {
  Real tl_norm = Real(0);
  Real fd_norm = Real(0);
  Real relative_error = Real(0);
  Real optimal_eps = Real(0);
  bool passed = false;
  std::string describe() const;
};
TangentCheckResult check_tangent(
    const std::function<void(const dyn::State&, dyn::State&, Real, int)>& model,
    const TangentLinearModel& tl, const dyn::State& xb,
    const dyn::State& dx, Real dt, int n_steps,
    const std::vector<Real>& epsilons = {Real(1e-2), Real(1e-3), Real(1e-4),
                                         Real(1e-5), Real(1e-6)});

}  // namespace vibe::da
