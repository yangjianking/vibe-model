#pragma once
/// @file kernels.hpp
/// @brief GPU/CPU 数值内核的**唯一公开接口**（签名与架构第 7 节一致）。
///
/// 组织方式
/// --------
/// 本文件只做**声明**。实现分两类：
///
///   * CPU：\c src/gpu/cpu/kernels_cpu.cpp —— 串行 + OpenMP，参考实现，
///     结果可与解析解逐位比对（tests/unit/test_gpu_precision.cpp）；
///   * CUDA：\c src/gpu/cuda/kernels_{advection,diffusion,tridiagonal,
///     helmholtz,interp,reduction}.cu —— 设备内核 + 主机侧启动包装。
///
/// **链接契约（重要）**：两种实现定义的是同一组 \c extern 函数，
/// 一次链接只能包含其中一组。CPU-only 构建只编译 \c kernels_cpu.cpp；
/// CUDA 构建只编译 \c src/gpu/cuda/*.cu。二者不可同时出现在同一链接单元里。
/// 这是"不修改构建系统"约束下唯一可行的组织方式（见交付总结中的接口偏差说明）。
///
/// 数据布局
/// --------
/// 内核操作 \c grid::Field<Real> 的**裸指针 + 网格步长**（架构第 4 节铁律 3：
/// 裸指针只出现在 common/gpu 层）。\c Field 为行主序、i 最快、含 halo：
///
/// \f[ \text{off}(i,j,k) = \big((k+h)\,n_{sy} + (j+h)\big)\,n_{sx} + (i+h) \f]
///
/// 因此所有内核都接受 \c (ptr, nsx, nsy, nsz, halo) 这一组"降级描述"，
/// 而不是 \c Field 本身——这样 GPU 侧不需要 Field 的模板实例，
/// CPU 侧也能直接在观测矢量上工作。
///
/// ============================ 内核一览 ============================
///   * 平流：\ref advect_scalar / \ref advect_momentum（2/4 阶中心，WENO5 可选）
///   * 扩散：\ref diffusion（Laplacian / biharmonic）
///   * 三对角：\ref tridiagonal_solve（批量 Thomas）
///   * Helmholtz：\ref helmholtz_residual / \ref helmholtz_jacobi
///   * 嵌套：\ref restrict_2to1 / \ref prolong_1to2
///   * 插值：\ref bilinear_interp / \ref trilinear_interp（观测算子）
///   * 归约：\ref reduce_sum / \ref reduce_max（补偿求和）
///
/// 文献：[D3] Arakawa & Lamb (1977)；[D6][D7][D8] 守恒离散与 WENO；
///       [T10] Thomas (1949)；[T13] Briggs et al. (2000) 多重网格；
///       [N4][N5] 嵌套限制/延拓；[O8] Lorenc et al. (2000) 观测算子插值；
///       [G4] Ogita, Rump & Oishi (2005) 精确求和；[G10] CUDA C++ Programming Guide。

#include <cstddef>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/common/types.hpp"
#include "vibe/gpu/backend.hpp"
#include "vibe/gpu/launch.hpp"
#include "vibe/gpu/memory.hpp"
#include "vibe/gpu/precision.hpp"
#include "vibe/grid/field.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::gpu {

// ===========================================================================
// 0. 几何描述与选项
// ===========================================================================
/// 网格的"裸描述"：内核只需要分辨率、halo 与（可选的）变分辨率度量。
/// 它由 \c grid::Grid 派生，但**不依赖** Grid 的实现细节，
/// 因此 GPU 侧可以自由构造测试用例。
struct KernelGeometry {
  Int nx = 0, ny = 0, nz = 0;  ///< 内部点数（不含 halo）
  Int halo = 0;                ///< halo 宽度
  Real dx = Real(1);           ///< 名义 x 分辨率（米）
  Real dy = Real(1);           ///< 名义 y 分辨率（米）
  Real dz = Real(1);           ///< 垂直层厚（米），或 \c zeta 层厚×z_top
  /// 逐列 x 单元宽度（长度 nx，可空 -> 用 dx）。变分辨率网格 [N5][N6]
  const Real* dx_cell = nullptr;
  /// 逐行 y 单元宽度（长度 ny，可空 -> 用 dy）
  const Real* dy_cell = nullptr;
  /// 地形追随坐标的雅可比 \f$ G^{1/2} = (H-z_s)/H \f$（长度 nx*ny，可空 -> 1）
  const Real* jacobian = nullptr;
  /// 周期水平边界（理想试验用）
  bool periodic_x = true;
  bool periodic_y = true;

  /// 该点（内部索引 i）的 x 单元宽度
  VIBE_HD Real dx_at(Int i) const noexcept {
    return (dx_cell != nullptr) ? dx_cell[i] : dx;
  }
  VIBE_HD Real dy_at(Int j) const noexcept {
    return (dy_cell != nullptr) ? dy_cell[j] : dy;
  }
  VIBE_HD Real inv_dx_at(Int i) const noexcept { return Real(1) / dx_at(i); }
  VIBE_HD Real inv_dy_at(Int j) const noexcept { return Real(1) / dy_at(j); }
  /// 单元体积 \f$ \Delta V = dx\,dy\,G^{1/2}\,dz \f$
  VIBE_HD Real cell_volume(Int i, Int j) const noexcept {
    const Real g = (jacobian != nullptr) ? jacobian[j * nx + i] : Real(1);
    return dx_at(i) * dy_at(j) * g * dz;
  }

  /// 由 \c grid::Grid 构造（若 \c dx_cell/dy_cell 为空则使用标称分辨率）
  static KernelGeometry from_grid(const grid::Grid& g);
};

/// 场在内存中的形状（含 halo）。所有内核的第一个参数组都用它。
struct FieldShape {
  Int nsx = 0, nsy = 0, nsz = 0;  ///< 含 halo 的存储维度
  Int halo = 0;                   ///< halo 宽度

  /// 由 \c Field 推导
  template <class T>
  static FieldShape from_field(const grid::FieldT<T>& f) noexcept {
    FieldShape s;
    s.nsx = f.nsx();
    s.nsy = f.nsy();
    s.nsz = f.nsz();
    s.halo = f.halo();
    return s;
  }
  /// 展平偏移（与 field.hpp 的 offset 完全一致）
  VIBE_HD std::size_t offset(Int i, Int j, Int k) const noexcept {
    return static_cast<std::size_t>(((k + halo) * nsy + (j + halo)) * nsx + (i + halo));
  }
  std::size_t size() const noexcept {
    return static_cast<std::size_t>(nsx) * static_cast<std::size_t>(nsy) *
           static_cast<std::size_t>(nsz);
  }
};

/// 平流离散选项
struct AdvectionOptions {
  int order = 2;             ///< 中心差分阶数：2 或 4
  bool weno5 = false;        ///< 用 WENO5 重建替代中心差分（[D7][D8]）
  bool flux_form = true;     ///< 守恒（通量差分）或平流（对流）形式
  Real divergence_damping = Real(0);  ///< 散度阻尼系数（[D12]，0 = 关闭）
};

/// 扩散选项
enum class DiffusionKind {
  Laplacian = 0,   ///< \f$ \nabla\cdot(\kappa\nabla q) \f$
  Biharmonic = 1   ///< \f$ -\nu\nabla^4 q \f$（尺度选择阻尼，[D12]）
};

// ===========================================================================
// 1. 平流（advection）
// ===========================================================================
/// 标量平流内核。
///
/// 数学形式（守恒形式，[D6] 有限体积）
/// -------------------------------------------------
/// \f[
///   \frac{\partial (\rho q)}{\partial t} + \nabla\cdot(\rho q \mathbf{u}) = 0
/// \f]
/// 写成趋势形式（\c dqdt 保存 \f$ \partial q/\partial t \f$）：
/// \f[
///   \frac{\partial q}{\partial t} =
///   -\frac{1}{\rho}\left[
///     \frac{\partial (\rho q u)}{\partial x} +
///     \frac{\partial (\rho q v)}{\partial y} +
///     \frac{\partial (\rho q w)}{\partial z} \right]
/// \f]
///
/// 离散化（2 阶中心，C-grid 交错）
/// ------------------------------------------------
/// 面通量 \f$ F^x_{i+1/2} = (\rho q)_f\,u_{i+1/2} \f$，其中
/// \f$ (\rho q)_f = \tfrac12[(\rho q)_i + (\rho q)_{i+1}] \f$；
/// 散度用相邻面通量差分：
/// \f[
///   \left(\frac{\partial (\rho q u)}{\partial x}\right)_i =
///   \frac{F^x_{i+1/2} - F^x_{i-1/2}}{\Delta x_i}
/// \f]
/// 该形式在周期边界下精确保持 \f$ \sum_i \rho_i q_i \Delta V_i \f$
/// （telescoping），误差仅在舍入层面。
///
/// 4 阶中心（非交错 5 点模板，[B2] 第 3 章）：
/// \f[
///   \frac{\partial q}{\partial x}\bigg|_i =
///   \frac{-q_{i+2} + 8q_{i+1} - 8q_{i-1} + q_{i-2}}{12\Delta x} + O(\Delta x^4)
/// \f]
///
/// WENO5 重建（\c weno5 = true，[D7][D8]）
/// ----------------------------------------
/// 对左侧通量用三个 3 阶候选多项式与非线性权 \f$ \omega_k \f$：
/// \f[
///   \hat{q}^{-}_{i+1/2} = \sum_{k=0}^{2} \omega_k q^{(k)}_{i+1/2},\qquad
///   \omega_k = \frac{\alpha_k}{\sum_j \alpha_j},\;
///   \alpha_k = \frac{d_k}{(\epsilon + \beta_k)^2}
/// \f]
/// 理想权 \f$ d = (0.3, 0.6, 0.1) \f$，光滑因子 \f$ \beta_k \f$ 为 [D8] 式 (2.6)；
/// \f$ \epsilon = 10^{-6} \f$ 防止除零。WENO5 的每点模板宽 5、算术量约为
/// 2 阶中心的 6~8 倍，但能抑制陡峭梯度处的振荡。
///
/// 线程映射与访存
/// --------------
///   1 个线程负责 1 个内部点 \c (i,j,k)；i 方向连续线程访问连续地址，
///   合并访存（coalesced）命中率最高。x 方向模板宽为 \c order（2 阶模板 3 点、
///   WENO5 为 5 点），故需要 halo >= order/2 + 1。
///
/// 共享内存：不显式使用（依赖 L1/L2 缓存）；若开 \c weno5 且要减少重复
/// 全局访存，可在 \c block 内用 \f$ (t_x+4)\times t_y \times t_z \f$ 的 tile。
///
/// 复杂度：\f$ O(N) \f$，N = 内部点数；每点 flops 约 18（2 阶）、
/// 约 60（4 阶）、约 120（WENO5）。访存 \f$ O(N) \f$（约 7 次读 + 1 次写）。
///
/// 精度策略：所有通量与差分在 \ref current_policy 的 \c compute 精度下计算，
/// 累加到 \c dqdt 时提升到 \c storage 精度（见文档第 3 节）。
///
/// 文献：[D3][D6][D7][D8][B2][T17][T18]。
///
/// @param q     标量场（含 halo）
/// @param u,v,w 风分量（C-grid 面错位，含 halo）
/// @param rho   密度（Cell，含 halo；\c flux_form = false 时可为 nullptr）
/// @param out   趋势 \c dqdt（Cell，含 halo），**先清零再写**
void advect_scalar(const Real* q, const Real* u, const Real* v, const Real* w, const Real* rho,
                   Real* out, const FieldShape& shape, const KernelGeometry& geom,
                   const AdvectionOptions& options = AdvectionOptions{});

/// 动量平流内核：对标量平流叠加"曲线坐标度量项 + 散度阻尼"。
///
/// \f[
///   \frac{\partial u}{\partial t} = -\mathbf{u}\cdot\nabla u
///   - \frac{1}{\rho}\frac{\partial p'}{\partial x} \;(\text{由 timeint 处理})
/// \f]
/// 本内核只负责第一项（平流），第二项（气压梯度）由半隐式求解器提供。
/// 对 u 分量使用 \f$ u\,\partial u/\partial x + v\,\partial u/\partial y
/// + w\,\partial u/\partial z \f$ 的对流形式，通量形式的散度与标量一致。
///
/// 变分辨率网格的度量项：\f$ \Delta x \to \Delta x_i \f$，
/// 由 \c geom.dx_at(i) 提供，因此 CPU/CUDA 代码都写作
/// \c (f(i+1)-f(i-1)) * 0.5 * geom.inv_dx_at(i)，不做 \c dx 的常量折叠。
///
/// 复杂度：\f$ O(N) \f$，每点约 3 个方向 × (2 阶 6 flops / 4 阶 12 flops)。
///
/// 线程映射与访存同 \ref advect_scalar；共享内存用量 0。
///
/// 文献：[D3][D4][D6][B2]。
void advect_momentum(const Real* u, const Real* v, const Real* w, const Real* rho, Real* du,
                     Real* dv, Real* dw, const FieldShape& shape, const KernelGeometry& geom,
                     const AdvectionOptions& options = AdvectionOptions{});

// ===========================================================================
// 2. 扩散与阻尼（diffusion）
// ===========================================================================
/// 扩散/超扩散内核。
///
/// Laplacian 形式（[B2] 第 2 章）
/// -----------------------------
/// \f[
///   \frac{\partial q}{\partial t} = \kappa\,\nabla^2 q,\qquad
///   (\nabla^2 q)_{i,j,k} = \frac{q_{i+1}-2q_i+q_{i-1}}{\Delta x_i^2}
///   + \frac{q_{j+1}-2q_j+q_{j-1}}{\Delta y_j^2}
///   + \frac{q_{k+1}-2q_k+q_{k-1}}{\Delta z^2}
/// \f]
///
/// Biharmonic 形式（尺度选择阻尼，[D12][D10]）
/// -------------------------------------------
/// \f[
///   \frac{\partial q}{\partial t} = -\nu\,\nabla^4 q = -\nu\,\nabla^2(\nabla^2 q)
/// \f]
/// 实现为**两次** Laplacian：中间量写入调用方提供的 \p work 缓冲；
/// 少一次全局往返（比两次内核启动省一次读写）。
///
/// 稳定条件（显式时间推进）：\f$ \kappa\Delta t \le \Delta x^2/6 \f$
/// （3D 中心差分的 von Neumann 条件，[B2] 第 3 章）。
///
/// 线程映射：1 线程 1 点，7 点模板；共享内存 0；
/// 复杂度 \f$ O(N) \f$；每点约 6 读 2 写（biharmonic 为 12 读 4 写）。
///
/// 精度策略：由 \ref current_policy 的 \c compute 决定中间量精度；
/// biharmonic 的中间量若用半精度会显著放大截断误差（误差按 \f$ \nabla^2 \f$ 放大
/// \f$ 1/\Delta x^2 \f$ 倍），因此强制 \c compute >= FP32（函数内检查）。
///
/// 文献：[B2][D12][G6]。
///
/// @param q     输入场（含 halo）
/// @param work  中间缓冲（biharmonic 需要；Laplacian 可为 nullptr）
/// @param kappa 扩散系数 \f$ \kappa \f$（\c Biharmonic 时解释为 \f$ \nu \f$）
void diffusion(const Real* q, Real* out, Real* work, const FieldShape& shape,
               const KernelGeometry& geom, Real kappa,
               DiffusionKind kind = DiffusionKind::Laplacian);

// ===========================================================================
// 3. 批量三对角求解（tridiagonal）
// ===========================================================================
/// 批量三对角求解（Thomas 算法 [T10]），沿垂直方向 k 求解，水平方向每列独立。
///
/// 方程
/// ----
/// \f[
///   a_k x_{k-1} + b_k x_k + c_k x_{k+1} = d_k,\qquad k = 0,\dots,n_z-1
/// \f]
/// 系数布局：\p a、\p b、\p c 的长度为 \f$ n_{\!sys}\times n_z \f$
/// （\c sys*nz + k），即多个**独立**三对角系统首尾相接；
/// \f$ a_0 \f$ 与 \f$ c_{n_z-1} \f$ 未使用。这样布局的好处是：
///   * CPU 侧可以连续访存（同一 k 层的所有列相邻）；
///   * GPU 侧一个线程（或一个 block）处理一个系统，寄存器/共享内存里
///     放 \f$ 3n_z \f$ 个系数（\f$ n_z \le 64 \f$ 时约 1.5 KB）。
///
/// 离散化：本内核不做离散化——它求解的是**时间推进器已经离散化好的**
/// 线性系统（例如半隐式声波-重力波方程，[T1][T2][T7][T8]）。
///
/// Thomas 算法（前向消元 + 回代）
/// -----------------------------
/// 前向：\f$ c'_k = \dfrac{c_k}{b_k - a_k c'_{k-1}},\;
/// d'_k = \dfrac{d_k - a_k d'_{k-1}}{b_k - a_k c'_{k-1}} \f$
/// 回代：\f$ x_k = d'_k - c'_k x_{k+1} \f$
/// 严格对角占优（\f$ |b_k| \ge |a_k|+|c_k| \f$）时无主元增长，
/// 数值稳定；否则返回 \ref NumericalError（不静默继续）。
///
/// 复杂度：\f$ O(n_{\!sys}\,n_z) \f$，约 \f$ 8 n_z \f$ flops/系统；
/// 访存 \f$ O(n_{\!sys}\,n_z) \f$（每列 2 遍）。
///
/// CUDA 实现：\c parallel cyclic reduction (PCR) 或 block-Thomas
/// （\c threadIdx.x \f$< n_z \f$ 且共享内存保存 \f$ c',d' \f$），
/// 见 src/gpu/cuda/kernels_tridiagonal.cu。
///
/// 文献：[T10] Thomas (1949)；[T1][T2][T7][T8] 半隐式离散；
///       [B11] Higham (2002) 第 4 章（前向误差与对角占优）。
///
/// @param n_sys 独立系统个数
/// @param nz    每个系统的长度
/// @param a,b,c,d 系数与右端项（就地修改用于前向消元）
/// @param x     解（长度 \f$ n_{\!sys}\times n_z \f$）
void tridiagonal_solve(Real* a, Real* b, Real* c, Real* d, Real* x,
                       std::size_t n_sys, std::size_t nz);

/// \c std::vector 便捷重载（单系统，与架构第 7 节的签名一致）。
/// 内部转发到上方的裸指针版本；系数按值传入不影响调用方。
void tridiagonal_solve(const std::vector<Real>& a, const std::vector<Real>& b,
                       const std::vector<Real>& c, const std::vector<Real>& d,
                       std::vector<Real>& x);

// ===========================================================================
// 4. Helmholtz 求解组件（helmholtz）
// ===========================================================================
/// Helmholtz 残差：\f$ r = f - (\nabla\cdot(a\nabla x) - b\,x) \f$。
///
/// 算子离散（3D 7 点，变系数）
/// --------------------------
/// \f[
///   \mathcal{L}x = \nabla\cdot(a\nabla x) - b x
///   = \frac{1}{\Delta x_i}\!\left[a_{i+1/2}\frac{x_{i+1}-x_i}{\Delta x_{i+1/2}}
///     - a_{i-1/2}\frac{x_i-x_{i-1}}{\Delta x_{i-1/2}}\right]
///   + (y,z\ \text{同理}) - b\,x
/// \f]
/// 面系数取相邻中心值的算术平均：\f$ a_{i+1/2} = \tfrac12(a_i + a_{i+1}) \f$
/// （对光滑 \f$ a \f$ 为 2 阶；变分辨率下用调和平均可保持通量守恒，
/// 见文档第 6 节讨论）。
///
/// 复杂度 \f$ O(N) \f$；每点 6 次邻居读 + 1 次写；共享内存 0。
///
/// 精度策略：残差必须比未知量**更精确**才能驱动迭代精化，
/// 因此本内核强制在 \c reduce（>= FP32，默认 FP64）精度下累加
/// （[G5] 迭代精化的残差计算要求，[B11] 第 4 章）。
///
/// 文献：[T1][T2][T13][G5]。
void helmholtz_residual(const Real* x, const Real* a, const Real* b, const Real* f, Real* r,
                        const FieldShape& shape, const KernelGeometry& geom);

/// Helmholtz 平滑子：Jacobi 或 Chebyshev 加权 Jacobi（[T13] 第 3 章）。
///
/// Jacobi 迭代
/// -----------
/// 记 \f$ D \f$ 为 \f$ \mathcal{L} \f$ 的对角（\f$ = -\sum_{\text{faces}} a/\Delta^2 - b \f$），
/// \f[ x^{(m+1)} = x^{(m)} + \frac{\omega}{D}\left(f - \mathcal{L}x^{(m)}\right) \f]
/// \f$ \omega = 2/3 \f$ 为 3D 7 点 Laplacian 的最优 Jacobi 权
/// （\f$ \omega_{opt} = 2/(2 + 2\cos(\pi h)) \to 2/3 \f$，[T13] 式 (3.24)）。
///
/// Chebyshev 加速：把 \f$ \omega \f$ 换成随迭代次数变化的
/// \f[ \omega_m = \frac{2}{2 - \rho^2 \omega_{m-1}},\quad \omega_0 = 1,\;
///    \rho = \rho(J) = \tfrac13(\cos(\pi/n_x)+\cos(\pi/n_y)+\cos(\pi/n_z)) \f]
/// Chebyshev 半迭代把谱半径 \f$ \rho \f$ 的有效误差降到
/// \f$ O(\rho^{1/2}) \f$，是多重网格中最常用的光滑子（[T13] 第 3.3 节）。
///
/// 迭代次数由参数 \p sweeps 给出；残差范数由调用方用
/// \ref reduce_sum 计算（保持内核"无全局同步"的设计）。
///
/// 线程映射：1 线程 1 点；**Jacobi 需要双缓冲**才能在 GPU 上并行，
/// 因此 \p x 与 \p scratch 必须不同（\p x 读、\p scratch 写），
/// 由调用方交换。共享内存：block 内 tile 可省 1/3 访存，本实现依赖 L1。
///
/// 复杂度：\f$ O(\text{sweeps}\cdot N) \f$。
///
/// 文献：[T13] Briggs, Henson & McCormick (2000)；[T2][T7]。
void helmholtz_jacobi(const Real* a, const Real* b, const Real* f, const Real* x, Real* out,
                      const FieldShape& shape, const KernelGeometry& geom, int sweeps = 1,
                      bool chebyshev = false);

// ===========================================================================
// 5. 嵌套限制与延拓（nest）
// ===========================================================================
/// 细网格 -> 粗网格：面积（体积）加权限制，细化比 \p ratio = r（通常 3）。
///
/// \f[
///   \phi^{\text{coarse}}_{I,J,K} =
///   \frac{1}{r^2}\sum_{p=0}^{r-1}\sum_{q=0}^{r-1} \phi^{\text{fine}}_{rI+p,\,rJ+q,\,K}
///   \quad\text{(水平)}
/// \f]
/// 质量类变量必须用 \f$ 1/r^2 \f$ 权重才能满足
/// \f$ \sum_{\text{children}} \rho_c \Delta V_c = \rho_p \Delta V_p \f$
/// （架构第 5.4 节守恒约束，[N4][N5]）。垂直方向不聚合（同一 k）。
///
/// 变分辨率情形：若粗网格单元面积不等于子单元之和，用面积比
/// \f$ \Delta A_{\text{coarse}}/\sum \Delta A_{\text{child}} \f$ 修正
/// （由 \c geom 的 \c dx_cell/dy_cell 提供）。
///
/// 复杂度 \f$ O(N_{\text{coarse}}\,r^2) \f$；单调、保正、保常数（见测试）。
///
/// 文献：[N3][N4][N5]（Skamarock & Klemp 1993；MPAS 2012）。
void restrict_2to1(const Real* fine, Real* coarse, const FieldShape& fine_shape,
                   const FieldShape& coarse_shape, Int ratio);

/// 粗网格 -> 细网格：双线性延拓（[N1][N5]）。
///
/// \f[
///   \phi^{\text{fine}}_{rI+p,\,rJ+q} = \text{bilin}\big(\phi^{\text{coarse}};\;
///   I + (p+0.5)/r,\; J + (q+0.5)/r\big)
/// \f]
/// 偏移 0.5 是因为子网格点是**单元中心**而非格点边界，缺了它会把
/// 常数场延拓成锯齿（见测试 "prolong 保持常数"）。
///
/// 边界：粗网格边缘外做**线性外推**（而非钳制），保证一阶导数连续；
/// 可选保正（\p enforce_positive）。
///
/// 复杂度 \f$ O(N_{\text{fine}}) \f$。
///
/// 文献：[N1] Davies (1976)；[N4][N5]；[D3]（错位网格插值的一致性）。
void prolong_1to2(const Real* coarse, Real* fine, const FieldShape& coarse_shape,
                  const FieldShape& fine_shape, Int ratio, bool enforce_positive = false);

// ===========================================================================
// 6. 观测算子插值（interp）
// ===========================================================================
/// 单点双线性插值（观测算子 [O8] 的基本构件）。
///
/// 设物理坐标 \f$ (x,y) \f$ 落在单元 \f$ (i,j) \f$ 内、
/// 局部坐标 \f$ \xi=(x-x_i)/\Delta x_i,\; \eta=(y-y_j)/\Delta y_j \f$，
/// 则
/// \f[
///   q(x,y) = (1-\xi)(1-\eta)\,q_{i,j} + \xi(1-\eta)\,q_{i+1,j}
///          + (1-\xi)\eta\,q_{i,j+1} + \xi\eta\,q_{i+1,j+1}
/// \f]
/// 误差 \f$ O(\Delta^2) \f$；在 C-grid 上需先把标量插到角点或把风分量
/// 插到中心（\c grid::stagger_to_cell，[D3]）。
///
/// 边界处理：坐标超出内部区域时钳制到最近的**内部**单元（不做外推），
/// 因为观测算子的外推会引入无法用背景误差协方差控制的误差。
///
/// 单点版本由主机调用：创建 1 线程索引空间，O(1) 复杂度。
/// 批量版本 \ref bilinear_interp_batch 才是 GPU 上的正确用法
/// （每线程一个观测，避免 O(n_obs) 次内核启动，[O8]）。
///
/// 文献：[O8] Lorenc et al. (2000)；[D3]；[B9] Wilks 第 3 章（插值误差）。
Real bilinear_interp(const Real* f, const FieldShape& shape, const KernelGeometry& geom, Int k,
                     Real x, Real y);

/// 批量双线性插值：n 个观测点，每个线程处理一个。
/// 复杂度 \f$ O(n) \f$；访存 \f$ O(4n) \f$（相邻观测通常命中同一缓存行）。
void bilinear_interp_batch(const Real* f, const FieldShape& shape, const KernelGeometry& geom,
                           Int k, const Real* x, const Real* y, Real* out, std::size_t n);

/// 单点三线性插值（与 \ref bilinear_interp 相同的约定，增加垂直权重 \f$ \zeta \f$）。
///
/// \f[
///   q(x,y,z) = \sum_{a,b,c\in\{0,1\}} w_a(\xi) w_b(\eta) w_c(\zeta)\, q_{i+a,j+b,k+c}
/// \f]
/// 其中 \f$ w_0(t) = 1-t,\; w_1(t) = t \f$。
/// 垂直坐标先由地形追随 \f$ \zeta \f$ 转成物理高度 \f$ z \f$
/// （\c grid::Grid::height，[D2]），再由调用方换算成 k 索引。
///
/// 复杂度 \f$ O(1) \f$；批量版本同 \ref bilinear_interp_batch。
///
/// 文献：[O8][D2][D3][B9]。
Real trilinear_interp(const Real* f, const FieldShape& shape, const KernelGeometry& geom, Real x,
                      Real y, Real z);

/// 批量三线性插值
void trilinear_interp_batch(const Real* f, const FieldShape& shape, const KernelGeometry& geom,
                            const Real* x, const Real* y, const Real* z, Real* out, std::size_t n);

// ===========================================================================
// 7. 归约（reduction）
// ===========================================================================
/// 精确求和：\f$ S = \sum_{i=0}^{n-1} x_i \f$，使用
/// \ref CompensatedSum（Kahan [G2] / Neumaier [G3]）。
///
/// 为什么必须补偿
/// --------------
/// 朴素左折叠在 \f$ n \f$ 项上的相对误差上界为 \f$ (n-1)u \f$
/// （[B11] 定理 4.2）；\f$ n = 10^8 \f$、\f$ u = 2^{-24} \f$ 时上界约 6，
/// 即结果可能完全失真。补偿求和的误差上界降到
/// \f$ (2u + O(nu^2))\sum|x_i| \f$（[B11] 式 (4.7)、[G4]），
/// 对 Kahan 为 \f$ 2u\sum|x_i| + O(nu^2) \f$，与 \f$ n \f$ 无关。
///
/// 归约结构（两级，与 [G4] 第 4 节一致）
/// -------------------------------------
///   1. 把 \f$ [0,n) \f$ 按 \ref LaunchConfig 的执行单元划成连续块，
///      每块用一个 \ref CompensatedSum 累加（并行、确定性）；
///   2. 按块顺序用 Neumaier 规则两两 \ref CompensatedSum::merge
///      （树形，避免长链），得到最终值。
///
/// 顺序确定性：块划分固定 ⇒ 同一输入在任何线程数下得到**逐位相同**的结果，
/// 这对 4D-Var 可复现性至关重要（[V3][A4]）。
///
/// 复杂度：\f$ O(n) \f$，额外寄存器 \f$ O(1) \f$；共享内存
/// \f$ O(\text{TPB}) \f$（GPU 版本块内先做 warp shuffle 再写共享内存，
/// 避免 \f$ O(\text{TPB}) \f$ 次原子操作）。
///
/// 文献：[G2] Kahan (1965)；[G3] Neumaier (1974)；[G4] Ogita et al. (2005)；
///       [B11] Higham (2002) 第 4 章。
Real reduce_sum(const Real* data, std::size_t n, SumAlgorithm algorithm = SumAlgorithm::Neumaier);

/// 带步伐的求和（用于按 stride 抽取同一列/同一层，例如垂直积分）
Real reduce_sum_strided(const Real* data, std::size_t n, std::size_t stride,
                        SumAlgorithm algorithm = SumAlgorithm::Neumaier);

/// 最大值归约（不需要补偿求和，但需要与 \ref reduce_sum 相同的分块顺序
/// 以保证确定性）。NaN 语义：NaN 参与比较时按 IEEE 的 \c fmax 语义被忽略。
Real reduce_max(const Real* data, std::size_t n);

/// 同时求最小/最大值（用于 CFL 与稳定性检查）
void reduce_minmax(const Real* data, std::size_t n, Real& vmin, Real& vmax);

/// 均值（= \ref reduce_sum / n，用补偿求和后一次除法，避免逐项除法的额外舍入）
Real reduce_mean(const Real* data, std::size_t n,
                 SumAlgorithm algorithm = SumAlgorithm::Neumaier);

}  // namespace vibe::gpu
