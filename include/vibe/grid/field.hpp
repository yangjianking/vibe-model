#pragma once
/// @file field.hpp
/// @brief 网格量容器：单数组 + halo 布局。
///
/// 内存布局
/// --------
/// 行主序，i 变化最快、k 最慢：
///
///     offset(i, j, k) = ((k + halo) * ny_s + (j + halo)) * nx_s + (i + halo)
///
/// 其中 nx_s/ny_s 为含 halo 的维度。该布局对 CPU 的向量化与 GPU 的
/// 合并访存都友好（[G10] CUDA C++ Programming Guide，第 9 章）。
///
/// GPU 互操作
/// ----------
/// FieldT 只持有主机侧 std::vector；`data()` 返回的指针可直接传给 GPU 内核，
/// 前提是该内存由 gpu::DeviceBuffer 托管（统一内存或已映射）。设备镜像的
/// 生命周期管理在 vibe::gpu 层完成，FieldT 不引入 GPU 依赖。

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/common/types.hpp"
#include "vibe/grid/geometry.hpp"

namespace vibe::grid {

template <class T>
class FieldT {
 public:
  using value_type = T;

  FieldT() = default;

  FieldT(const Grid& g, Stagger s, std::string name)
      : grid_(&g), stagger_(s), name_(std::move(name)) {
    layout_.nx = g.nx();
    layout_.ny = g.ny();
    layout_.nz = g.nz();
    layout_.halo = g.halo();
    layout_.nsx = layout_.nx + 2 * layout_.halo;
    layout_.nsy = layout_.ny + 2 * layout_.halo;
    layout_.nsz = layout_.nz + 2 * layout_.halo;
    if (s == Stagger::FaceX) layout_.nsx += 1;
    if (s == Stagger::FaceY) layout_.nsy += 1;
    if (s == Stagger::FaceZ) layout_.nsz += 1;
    data_.assign(static_cast<Size>(layout_.nsx) * layout_.nsy * layout_.nsz, T(0));
  }

  // ---- 元信息 -------------------------------------------------------------
  const std::string& name() const noexcept { return name_; }
  void set_name(std::string n) { name_ = std::move(n); }
  Stagger stagger() const noexcept { return stagger_; }
  Int nx() const noexcept { return layout_.nx; }
  Int ny() const noexcept { return layout_.ny; }
  Int nz() const noexcept { return layout_.nz; }
  Int halo() const noexcept { return layout_.halo; }
  Int nsx() const noexcept { return layout_.nsx; }
  Int nsy() const noexcept { return layout_.nsy; }
  Int nsz() const noexcept { return layout_.nsz; }
  Size size() const noexcept { return data_.size(); }
  bool empty() const noexcept { return data_.empty(); }
  const Grid& grid() const { VIBE_CHECK(grid_ != nullptr); return *grid_; }

  // ---- 数据访问 -----------------------------------------------------------
  T* data() noexcept { return data_.data(); }
  const T* data() const noexcept { return data_.data(); }
  std::vector<T>& storage() noexcept { return data_; }
  const std::vector<T>& storage() const noexcept { return data_; }

  /// 本地内部索引 -> 展平偏移
  Size offset(Int i, Int j, Int k) const noexcept {
    return static_cast<Size>(((k + layout_.halo) * layout_.nsy + (j + layout_.halo)) *
                                 layout_.nsx +
                             (i + layout_.halo));
  }

  /// 存储索引（含 halo 偏移）访问
  T& at(Int i, Int j, Int k) noexcept { return data_[offset(i, j, k)]; }
  const T& at(Int i, Int j, Int k) const noexcept { return data_[offset(i, j, k)]; }

  T& operator()(Int i, Int j, Int k) noexcept { return data_[offset(i, j, k)]; }
  const T& operator()(Int i, Int j, Int k) const noexcept { return data_[offset(i, j, k)]; }

  /// 带越界钳制的访问（诊断/绘图用，禁止出现在时间步内）
  T clamp_at(Int i, Int j, Int k) const noexcept {
    const Int ii = clamp(i, Int(0), layout_.nx - 1);
    const Int jj = clamp(j, Int(0), layout_.ny - 1);
    const Int kk = clamp(k, Int(0), layout_.nz - 1);
    return data_[offset(ii, jj, kk)];
  }

  // ---- 批量操作 -----------------------------------------------------------
  void fill(T v) { std::fill(data_.begin(), data_.end(), v); }

  /// *this = a * x + b * (*this)   （BLAS axpy 语义，供 RK 与 4D-Var 使用）
  void axpy(T a, const FieldT& x, T b) {
    VIBE_CHECK(x.size() == data_.size());
    for (Size n = 0; n < data_.size(); ++n) data_[n] = a * x.data_[n] + b * data_[n];
  }

  /// *this = a * x + (*this)
  void add_scaled(T a, const FieldT& x) {
    VIBE_CHECK(x.size() == data_.size());
    for (Size n = 0; n < data_.size(); ++n) data_[n] += a * x.data_[n];
  }

  /// *this *= a
  void scale(T a) {
    for (auto& v : data_) v *= a;
  }

  FieldT& operator+=(const FieldT& o) { add_scaled(T(1), o); return *this; }
  FieldT& operator-=(const FieldT& o) { add_scaled(T(-1), o); return *this; }

  /// 欧氏范数（用于点积检验与收敛判据）
  Real norm2() const {
    Real s = Real(0);
    for (const auto& v : data_) s += static_cast<Real>(v) * static_cast<Real>(v);
    return std::sqrt(s);
  }

  /// 点积 <*this, o>
  Real dot(const FieldT& o) const {
    VIBE_CHECK(o.size() == data_.size());
    Real s = Real(0);
    for (Size n = 0; n < data_.size(); ++n) s += static_cast<Real>(data_[n]) * static_cast<Real>(o.data_[n]);
    return s;
  }

  /// 是否含非有限值（NaN/Inf）；用于数值发散检测
  bool has_nonfinite() const {
    for (const auto& v : data_) {
      if (!std::isfinite(static_cast<Real>(v))) return true;
    }
    return false;
  }

  /// 统计信息：{min, max, mean, rms}
  struct Stats { Real min, max, mean, rms; };
  Stats stats() const {
    Real lo = std::numeric_limits<Real>::max();
    Real hi = -lo;
    Real s1 = Real(0), s2 = Real(0);
    for (const auto& v : data_) {
      const Real x = static_cast<Real>(v);
      lo = std::min(lo, x);
      hi = std::max(hi, x);
      s1 += x;
      s2 += x * x;
    }
    const Real n = data_.empty() ? Real(1) : static_cast<Real>(data_.size());
    return {lo, hi, s1 / n, std::sqrt(s2 / n)};
  }

  /// 数组拷入/拷出（用于 IO 与 4D-Var 打包）
  void pack(std::vector<Real>& out) const {
    out.resize(data_.size());
    for (Size n = 0; n < data_.size(); ++n) out[n] = static_cast<Real>(data_[n]);
  }
  void unpack(const std::vector<Real>& in) {
    VIBE_CHECK(in.size() == data_.size());
    for (Size n = 0; n < data_.size(); ++n) data_[n] = static_cast<T>(in[n]);
  }

  /// 深拷贝（同布局）
  FieldT clone() const { return *this; }

 private:
  struct Layout {
    Int nx = 0, ny = 0, nz = 0, halo = 0;
    Int nsx = 0, nsy = 0, nsz = 0;
  };
  const Grid* grid_ = nullptr;
  Stagger stagger_ = Stagger::Cell;
  std::string name_;
  Layout layout_{};
  std::vector<T> data_;
};

template <class T> using Field = FieldT<T>;

/// 在内部点 (i,j,k) 处做 2 阶中心差分：d/dx
template <class T>
inline T ddx(const FieldT<T>& f, Int i, Int j, Int k, Real inv_dx) {
  return (f(i + 1, j, k) - f(i - 1, j, k)) * static_cast<T>(Real(0.5) * inv_dx);
}
template <class T>
inline T ddy(const FieldT<T>& f, Int i, Int j, Int k, Real inv_dy) {
  return (f(i, j + 1, k) - f(i, j - 1, k)) * static_cast<T>(Real(0.5) * inv_dy);
}
/// 在 w 层 (k) 处计算 d/dzeta（相邻为标量层中心）
template <class T>
inline T ddz_face(const FieldT<T>& f, Int i, Int j, Int k, Real inv_dz) {
  return (f(i, j, k) - f(i, j, k - 1)) * static_cast<T>(inv_dz);
}

/// 场之间的算术组合（就地）
template <class T>
inline void field_add(const FieldT<T>& a, const FieldT<T>& b, FieldT<T>& r) {
  r.fill(T(0));
  r.add_scaled(T(1), a);
  r.add_scaled(T(1), b);
}

}  // namespace vibe::grid
