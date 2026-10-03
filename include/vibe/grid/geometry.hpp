#pragma once
/// @file geometry.hpp
/// @brief 网格几何、错位定义与度量项。
///
/// 坐标系
/// ------
/// 水平：Arakawa C-grid（[D3]），u 位于 x 面心，v 位于 y 面心，标量位于体心。
/// 垂直：Lorenz 错位，w 位于层界面，标量位于层中心（[D4] 能量守恒错位）。
/// 垂向坐标：地形追随高度坐标（[D2] Gal-Chen & Somerville 1975）
///
///     zeta = H * (z - z_s) / (H - z_s)
///
/// 由 (x, y, zeta) 到 (x, y, z) 的变换引入度量项；在有限体积离散下
/// 雅可比 G^{1/2} = dz/dzeta 与水平单元面积 (dx*dy) 一起构成体积元
///
///     dV = dx * dy * (dz/dzeta) * dzeta
///
/// 变分辨率网格（[N5][N6]）通过把 dx、dy 变成逐列/逐行的数组实现，
/// 动力学代码看到的接口与均匀网格完全一致（见 variable_resolution.hpp）。
///
/// 平滑坐标面（[D9][D10][D11]）用于抑制陡峭地形上的虚假气压梯度，
/// 由 `smooth_vertical_levels` 在网格生成阶段施加。

#include <cmath>
#include <string>
#include <vector>

#include "vibe/common/constants.hpp"
#include "vibe/common/error.hpp"
#include "vibe/common/types.hpp"
#include "vibe/grid/decomposition.hpp"

namespace vibe::grid {

/// 错位类型
enum class Stagger {
  Cell = 0,   ///< 标量：rho, theta, pi, 水物质
  FaceX = 1,  ///< u
  FaceY = 2,  ///< v
  FaceZ = 3,  ///< w
  Corner = 4  ///< 诊断/检验用角点场
};

inline const char* to_string(Stagger s) noexcept {
  switch (s) {
    case Stagger::Cell: return "cell";
    case Stagger::FaceX: return "face_x";
    case Stagger::FaceY: return "face_y";
    case Stagger::FaceZ: return "face_z";
    case Stagger::Corner: return "corner";
  }
  return "unknown";
}

struct Index3 {
  Int i = 0, j = 0, k = 0;
};

/// 网格几何：只描述几何与度量，不含任何物理量
struct Geometry {
  Int nx = 0, ny = 0, nz = 0;   ///< 全局内部点数
  Real x0 = Real(0), y0 = Real(0);
  Real z_top = Real(20000);      ///< 模式顶高度（平坦地面处）m
  Real dx = Real(1000), dy = Real(1000);  ///< 名义水平分辨率；变分辨率为最小值

  /// 垂直层界面坐标 (nz+1)，取值 [0, 1]；层中心 zeta_c(k) = 0.5*(zeta[k]+zeta[k+1])
  std::vector<Real> zeta;

  /// 全局地形高度 z_s(i,j)，长度 nx*ny（行主序，j 为慢变）；平坦地形时为空
  std::vector<Real> zs;

  /// 变分辨率：逐列 x 单元宽度（长度 nx，可空）；逐行 y 单元宽度（长度 ny，可空）
  std::vector<Real> dx_cell;
  std::vector<Real> dy_cell;

  bool variable_resolution = false;
  bool flat_terrain = true;
  std::string name = "d01";

  /// 垂直层数校验 + 缺省拉伸（近地层加密）
  static std::vector<Real> default_zeta_levels(Int nz, Real stretch = Real(1.0));

  bool valid() const noexcept {
    return nx > 0 && ny > 0 && nz > 0 && static_cast<Int>(zeta.size()) == nz + 1;
  }
};

/// 网格：几何 + 并行分解 + 预计算度量项
class Grid {
 public:
  Grid() = default;
  Grid(Geometry g, Decomposition d);

  const Geometry& geom() const noexcept { return geom_; }
  const Decomposition& decomp() const noexcept { return dec_; }
  const std::string& name() const noexcept { return geom_.name; }

  // ---- 维度 ---------------------------------------------------------------
  Int nx() const noexcept { return dec_.nx_local(); }
  Int ny() const noexcept { return dec_.ny_local(); }
  Int nz() const noexcept { return geom_.nz; }
  Int nx_global() const noexcept { return geom_.nx; }
  Int ny_global() const noexcept { return geom_.ny; }
  Int halo() const noexcept { return dec_.halo; }

  /// 该错位网格的本地内部点数（不含 halo）
  Int size(Stagger s) const noexcept;
  /// 含 halo 的存储点数
  Int storage_size(Stagger s) const noexcept;
  /// halo 之后、内部区域的起始偏移
  Index3 origin(Stagger s) const noexcept;

  /// 把 (i,j,k)（本地内部索引，从 0 开始）映射为含 halo 的存储下标
  Index3 store_index(Stagger s, Int i, Int j, Int k) const noexcept;
  /// 一维展平下标（行主序：i 最快，k 最慢）
  Index flatten(Stagger s, Int i, Int j, Int k) const noexcept;

  /// 判断是否为内部点
  bool interior(Stagger s, Int i, Int j, Int k) const noexcept;

  // ---- 度量项 -------------------------------------------------------------
  /// 本列 x 单元宽度（变分辨率）或 dx
  Real dx_at(Int i) const noexcept;
  /// 本行 y 单元宽度
  Real dy_at(Int j) const noexcept;
  /// 层 k 的 dzeta
  Real dzeta(Int k) const noexcept;
  /// 由 zeta 与地形得到的物理高度 z(i,j,k)，k 为层界面；k 可为半整数用 kk=2k+1 表示
  Real height(Int i, Int j, Real kk) const noexcept;
  /// 地形高度（本地 i,j）
  Real terrain(Int i, Int j) const noexcept;
  /// 雅可比 G^{1/2} = dz/dzeta = (H - z_s) / H
  Real jacobian(Int i, Int j) const noexcept;
  /// 垂直层中心对应的 z
  Real z_center(Int i, Int j, Int k) const noexcept {
    return height(i, j, Real(k) + Real(0.5));
  }
  /// 垂直层界面（w 层）对应的 z
  Real z_interface(Int i, Int j, Int k) const noexcept { return height(i, j, Real(k)); }

  /// 1/dz（层中心），用于垂直差分
  Real inv_dz(Int i, Int j, Int k) const noexcept;

  /// 单元体积（有限体积用）
  Real cell_volume(Int i, Int j, Int k) const noexcept {
    return dx_at(i) * dy_at(j) * jacobian(i, j) * dzeta(k) * geom_.z_top;
  }

  /// 输出网格摘要（日志用）
  std::string describe() const;

  /// 设置内部场（重建度量项缓存）
  void rebuild_metrics();

 private:
  Geometry geom_{};
  Decomposition dec_{};
  std::vector<Real> zs_local_;   ///< 本地地形切片 (nx_global x ny_local) 简化为本地
  std::vector<Real> zc_;         ///< 层中心 zeta (nz)
  std::vector<Real> dzi_;        ///< 1/(z_{k+1}-z_k)
};

/// 生成平滑垂直层：近地层厚约 20 m，向上几何拉伸，顶约 500 m
std::vector<Real> make_stretched_zeta(Int nz, Real first_thickness, Real stretch);

/// 对地形做多次 1-2-1 平滑，抑制陡坡上的截断误差（[D10][D11]）
void smooth_terrain(std::vector<Real>& zs, Int nx, Int ny, int passes, Real max_slope);

/// 生成孤立山（Witch of Agnesi，[D2] 经典理想试验）
std::vector<Real> witch_of_agnesi(Int nx, Int ny, Real dx, Real dy,
                                  Real h0, Real a, Real xc, Real yc);

/// 生成 Schär 型光滑山（[D9]）
std::vector<Real> schar_mountain(Int nx, Int ny, Real dx, Real dy,
                                 Real h0, Real lambda, Real xc, Real yc);

}  // namespace vibe::grid
