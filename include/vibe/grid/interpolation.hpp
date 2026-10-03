#pragma once
/// @file interpolation.hpp
/// @brief 错位网格之间的插值、水平/垂直插值与守恒重映射。
///
/// 用途
/// ----
///   * C-grid 错位 -> 体心（诊断、物理参数化、检验）
///   * 体心 -> 面（边界条件与嵌套边界值）
///   * 垂直插值到气压层/高度层（观测算子、检验、后处理）
///   * 嵌套的粗->细延拓（prolongation）与细->粗限制（restriction）
///
/// 文献：[D3] Arakawa & Lamb (1977) 错位网格插值；[N1] Davies (1976)；
///       [N5] MPAS 守恒重映射；[D16] WRF ARW 第 3 章。

#include <functional>
#include <memory>
#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/grid/field.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::grid {

/// 插值阶数
enum class InterpOrder { Linear = 2, Cubic = 4, Quintic = 6 };

/// 水平插值：错位 -> 体心（就地输出到 Cell 场，边界用单侧外推）
void stagger_to_cell(const Field<Real>& in, Field<Real>& out);

/// 体心 -> x 面（简单平均，保持二阶精度）
void cell_to_face_x(const Field<Real>& in, Field<Real>& out);
void cell_to_face_y(const Field<Real>& in, Field<Real>& out);
void cell_to_face_z(const Field<Real>& in, Field<Real>& out);

/// 体心 -> 角点（双线性）
void cell_to_corner(const Field<Real>& in, Field<Real>& out);

/// 一般水平双线性插值：物理坐标 (x, y) -> 该点标量值
Real bilinear(const Field<Real>& f, const Grid& g, Real x, Real y, Int k);

/// 一般水平三次插值（16 点模板，[T18] Rai & Moin 1991）
Real bicubic(const Field<Real>& f, const Grid& g, Real x, Real y, Int k);

/// 三线性插值：(x, y, z) -> 标量
Real trilinear(const Field<Real>& f, const Grid& g, Real x, Real y, Real z);

/// 垂直插值到指定高度（单调 PCHIP，避免过冲；[B9] Wilks 第 3 章）
Real interp_to_height(const std::vector<Real>& z, const std::vector<Real>& v, Real z_target);

/// 垂直插值到指定气压层（输入为层中心 p 与变量）
Real interp_to_pressure(const std::vector<Real>& p, const std::vector<Real>& v, Real p_target);

/// 水平方向的保守重映射（面积加权），用于细->粗
/// @param src 细网格（Cell）
/// @param dst 粗网格（Cell）
/// @param ratio 细化比 r
void conservative_restrict(const Field<Real>& src, Field<Real>& dst, Int ratio);

/// 粗->细延拓：双线性 + 边缘一致性（保正的可选开关）
void bilinear_prolong(const Field<Real>& src, Field<Real>& dst, Int ratio,
                      bool enforce_positive = false);

/// 通用可插拔插值器接口（供观测算子与嵌套选择不同实现）
class Interpolator {
 public:
  virtual ~Interpolator() = default;
  virtual Real operator()(const Field<Real>& f, const Grid& g, Real x, Real y,
                          Real z) const = 0;
  virtual const char* name() const noexcept = 0;
};

std::unique_ptr<Interpolator> make_interpolator(InterpOrder order);

}  // namespace vibe::grid
