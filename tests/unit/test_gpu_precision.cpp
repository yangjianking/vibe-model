/// @file test_gpu_precision.cpp
/// @brief GPU 抽象层与混合精度的单元测试（CPU 后端，无需 GPU 即可运行）。
///
/// 覆盖范围
/// --------
///   1. 补偿求和（Kahan vs Neumaier）在病态累加下的精度与上界（[G2][G3][G4][B11]）；
///   2. 半精度类型 half_t / bfloat16_t 的往返、舍入、特殊值（[G1]）；
///   3. PrecisionTraits 的 epsilon / max / 舍入单位；
///   4. DeviceBuffer 的移动语义、拷贝、填充、重分配与内存统计；
///   5. MemoryPool 的复用统计；
///   6. LaunchConfig 的构造、for_field 派生与索引空间约定；
///   7. CPU 内核：三对角与解析解一致、限制/延拓的守恒性与恒等性、
///      插值的线性精确性、归约与补偿求和一致；
///   8. 后端枚举、能力矩阵与设备枚举。
///
/// 运行方式（构建系统已存在，本文件不修改构建）：把本文件加入 \c vibe_gpu 的
/// 单元测试目标，链接 \c src/gpu/device.cpp、\c src/gpu/memory.cpp、
/// \c src/gpu/cpu/kernels_cpu.cpp（或 CUDA 版的 .cu），入口调用
/// \c vibe::test::run_all()。
///
/// 文献：[G1]-[G4]、[G6][G7]、[G10]、[B11]、[T10]、[T13]、[N4]、[O8]。

#include "vibe/common/test.hpp"
#include "vibe/gpu/kernels.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/common/types.hpp"
#include "vibe/gpu/backend.hpp"
#include "vibe/gpu/device.hpp"
#include "vibe/gpu/launch.hpp"
#include "vibe/gpu/memory.hpp"
#include "vibe/gpu/precision.hpp"

namespace {

using vibe::Real;
using namespace vibe::gpu;

/// 构造一个不含 halo 的规则网格（避免依赖 \c grid::Grid 的实现）
KernelGeometry make_geom(vibe::Int nx, vibe::Int ny, vibe::Int nz, Real dx = Real(1),
                         Real dy = Real(1), Real dz = Real(1)) {
  KernelGeometry g;
  g.nx = nx;
  g.ny = ny;
  g.nz = nz;
  g.halo = 0;
  g.dx = dx;
  g.dy = dy;
  g.dz = dz;
  return g;
}

FieldShape make_shape(vibe::Int nx, vibe::Int ny, vibe::Int nz) {
  FieldShape s;
  s.nsx = nx;
  s.nsy = ny;
  s.nsz = nz;
  s.halo = 0;
  return s;
}

/// 用函数 f(i,j,k) 填充一个场（行主序，i 最快）
template <class F>
std::vector<Real> fill_field(const FieldShape& s, F&& f) {
  std::vector<Real> data(s.size(), Real(0));
  for (vibe::Int k = 0; k < s.nsz; ++k) {
    for (vibe::Int j = 0; j < s.nsy; ++j) {
      for (vibe::Int i = 0; i < s.nsx; ++i) {
        data[s.offset(i, j, k)] = f(i, j, k);
      }
    }
  }
  return data;
}

}  // namespace

// ===========================================================================
// 1. 补偿求和
// ===========================================================================

/// 1e6 个 1.0：Neumaier 的结果必须精确等于项数，而朴素累加在
/// \f$ n u > 1 \f$ 之后必然出现偏差（[B11] 定理 4.2）。
VIBE_TEST(compensated_sum_exact_over_million)
{
  const std::size_t n = 1000000;
  std::vector<double> v(n, 1.0);
  CompensatedSum<double, SumAlgorithm::Neumaier> acc;
  for (double x : v) acc.add(x);
  VIBE_CHECK_NEAR(acc.value(), static_cast<double>(n), 1e-6);
  const double naive = naive_sum(v.data(), v.size());
  VIBE_CHECK_NEAR(naive, static_cast<double>(n), 1.0);
  // 补偿求和至少不差于朴素求和
  VIBE_CHECK(std::abs(acc.value() - static_cast<double>(n)) <=
             std::abs(naive - static_cast<double>(n)) + 1e-12);
}

/// Neumaier 对"大正数 + 大量小正数 + 大负数抵消"序列优于 Kahan（[G3]）
VIBE_TEST(neumaier_beats_kahan_on_cancellation)
{
  CompensatedSum<double, SumAlgorithm::Kahan> kahan;
  CompensatedSum<double, SumAlgorithm::Neumaier> neumaier;
  kahan.add(1e16);
  neumaier.add(1e16);
  for (int i = 0; i < 1000000; ++i) {
    kahan.add(1.0);
    neumaier.add(1.0);
  }
  kahan.add(-1e16);
  neumaier.add(-1e16);
  // 精确值为 1000000
  const double exact = 1000000.0;
  const double ek = std::abs(kahan.value() - exact);
  const double en = std::abs(neumaier.value() - exact);
  VIBE_CHECK(en <= ek);
  VIBE_CHECK_NEAR(neumaier.value(), exact, 1e-6);
}

/// Kahan 在常见序列上的误差不超过 1 ulp 量级
/// \f$ |s+c - S| \le 2u\sum|x_i| \f$（[G4] 式 (2.4)）
VIBE_TEST(kahan_error_bound_holds)
{
  const std::size_t n = 100000;
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = 1.0 / static_cast<double>(i + 1);
  CompensatedSum<double, SumAlgorithm::Kahan> k;
  for (double x : v) k.add(x);
  const double absorbed = 2.0 * PrecisionTraits<Precision::FP64>::unit_roundoff() *
                          (k.max_abs * static_cast<double>(n));
  VIBE_CHECK(std::abs(k.correction()) <= absorbed);
  const double bound = 2.0 * PrecisionTraits<Precision::FP64>::unit_roundoff() * 12.0;
  VIBE_CHECK(std::abs(k.value() - k.raw()) <= bound);
}

/// reset / value / raw / correction 的语义
VIBE_TEST(compensated_sum_reset_and_accessors)
{
  CompensatedSum<double> s;
  VIBE_CHECK_NEAR(s.value(), 0.0, 1e-30);
  VIBE_CHECK(s.count == 0);
  s.add(1.0);
  s.add(2.0);
  s.add(3.0);
  VIBE_CHECK_NEAR(s.value(), 6.0, 1e-14);
  VIBE_CHECK(s.count == 3);
  s.reset();
  VIBE_CHECK_NEAR(s.value(), 0.0, 1e-30);
  VIBE_CHECK_NEAR(s.correction(), 0.0, 1e-30);
  VIBE_CHECK(s.count == 0);
}

/// merge 与串行累加等价（块级归约的正确性前提，[G4] 第 4 节）
VIBE_TEST(compensated_sum_merge_matches_serial)
{
  std::vector<double> v(1000);
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = std::sin(static_cast<double>(i)) * 1e8;
  CompensatedSum<double> serial;
  for (double x : v) serial.add(x);
  CompensatedSum<double> a;
  for (std::size_t i = 0; i < 400; ++i) a.add(v[i]);
  CompensatedSum<double> b;
  for (std::size_t i = 400; i < v.size(); ++i) b.add(v[i]);
  a.merge(b);
  VIBE_CHECK_NEAR(a.value(), serial.value(), 1e-3);
}

/// 补偿加法在 float 上也能把 n=1e7 的累加维持在 1 ulp 内（[G7] 的降精度实验）
VIBE_TEST(compensated_sum_float_vs_naive)
{
  const std::size_t n = 2000000;
  std::vector<float> v(n, 0.1f);
  CompensatedSum<float> acc;
  for (float x : v) acc.add(x);
  const double exact = 0.1 * static_cast<double>(n);
  const double naive = static_cast<double>(naive_sum(v.data(), v.size()));
  VIBE_CHECK(std::abs(static_cast<double>(acc.value()) - exact) < std::abs(naive - exact));
}

/// TwoSum 变体：保留全部误差项，误差项与主和一起构成精确值
VIBE_TEST(compensated_sum_twosum_variant)
{
  CompensatedSum<double, SumAlgorithm::TwoSum> s;
  s.add(1e16);
  s.add(1.0);
  s.add(-1e16);
  // TwoSum 把 (s-t)+x 全部保留在 c 中，故 value 精确为 1
  VIBE_CHECK_NEAR(s.value(), 1.0, 1e-9);
}

/// 运行期算法选择接口 compensated_sum 与模板版本一致
VIBE_TEST(compensated_sum_runtime_selector)
{
  std::vector<double> v(10000);
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = 1.0;
  const double a = compensated_sum(v.data(), v.size(), SumAlgorithm::Kahan);
  const double b = compensated_sum(v.data(), v.size(), SumAlgorithm::Neumaier);
  const double c = compensated_sum(v.data(), v.size(), SumAlgorithm::TwoSum);
  VIBE_CHECK_NEAR(a, 10000.0, 1e-9);
  VIBE_CHECK_NEAR(b, 10000.0, 1e-9);
  VIBE_CHECK_NEAR(c, 10000.0, 1e-9);
}

// ===========================================================================
// 2. 半精度类型
// ===========================================================================

/// half_t 的基本往返：所有可精确表示的二进制小数应当无损
VIBE_TEST(half_t_exact_roundtrip)
{
  const float exact[] = {0.0f,  1.0f,   -1.0f,  0.5f,   2.0f,   -2.0f,
                         0.25f, 1024.0f, -0.125f, 0.0625f, 2048.0f};
  for (float x : exact) {
    const half_t h = half_t::from_float(x);
    VIBE_CHECK_NEAR(h.to_float(), x, 1e-6);
  }
}

/// half_t 的舍入：round-to-nearest-even（[G1] 第 3 节）
VIBE_TEST(half_t_rounding_is_nearest_even)
{
  // 1 + 2^-11 正好落在 1.0 与 1.0009765625 的中点，偶数尾数 -> 取 1.0
  const float mid = 1.0f + std::ldexp(1.0f, -11);
  VIBE_CHECK_NEAR(half_t::from_float(mid).to_float(), 1.0f, 1e-9);
  // 略大于中点 -> 进位到 1 + 2^-10
  const float above = 1.0f + std::ldexp(1.0f, -10) * 0.75f;
  VIBE_CHECK(half_t::from_float(above).to_float() > 1.0f);
  // 截断模式必须 <= round-to-nearest 结果
  const half_t t = half_t::from_float_trunc(1.75f);
  VIBE_CHECK_NEAR(t.to_float(), 1.5f, 1e-6);
}

/// half_t 的特殊值：0 / Inf / NaN / 次正规 / 上溢
VIBE_TEST(half_t_special_values)
{
  const half_t zero = half_t::from_float(0.0f);
  VIBE_CHECK(zero.is_zero());
  VIBE_CHECK(!zero.sign_bit());
  const half_t neg_zero = half_t::from_float(-0.0f);
  VIBE_CHECK(neg_zero.is_zero());
  VIBE_CHECK(neg_zero.sign_bit());

  const half_t inf = half_t::from_float(std::numeric_limits<float>::infinity());
  VIBE_CHECK(inf.is_inf());
  VIBE_CHECK(!inf.sign_bit());
  const half_t ninf = half_t::from_float(-std::numeric_limits<float>::infinity());
  VIBE_CHECK(ninf.is_inf());
  VIBE_CHECK(ninf.sign_bit());

  const half_t nan = half_t::from_float(std::numeric_limits<float>::quiet_NaN());
  VIBE_CHECK(nan.is_nan());
  VIBE_CHECK(!nan.is_inf());

  const half_t big = half_t::from_float(1.0e30f);
  VIBE_CHECK(big.is_inf());  // 超出 half 最大值 -> 上溢为 Inf

  const half_t tiny = half_t::from_float(1.0e-30f);
  VIBE_CHECK(tiny.is_zero());  // 远低于最小次正规 -> 归零

  // 最小次正规：2^-24
  const half_t sub = half_t::from_float(std::ldexp(1.0f, -24));
  VIBE_CHECK_NEAR(sub.to_float(), std::ldexp(1.0f, -24), 1e-30);
  VIBE_CHECK(!sub.is_zero());
}

/// half_t 的精度上界：相对误差不超过 u = 2^-11
VIBE_TEST(half_t_relative_error_bound)
{
  const double u = PrecisionTraits<Precision::FP16>::unit_roundoff();
  for (int e = -10; e <= 10; ++e) {
    for (double frac = 1.0; frac < 2.0; frac += 0.03125) {
      const float x = static_cast<float>(std::ldexp(frac, e));
      const float y = half_t::from_float(x).to_float();
      const double rel = std::abs(static_cast<double>(y) - static_cast<double>(x)) /
                         std::abs(static_cast<double>(x));
      VIBE_CHECK(rel <= u + 1e-12);
    }
  }
}

/// bfloat16_t：与 FP32 同指数范围，往返精度为 2^-8
VIBE_TEST(bfloat16_roundtrip_and_range)
{
  const double u = PrecisionTraits<Precision::BF16>::unit_roundoff();
  const float ws[] = {1.0f, 0.5f, 2.0f, 1024.0f, 1.0e-20f, 3.0e38f};
  for (float x : ws) {
    const float y = bfloat16_t::from_float(x).to_float();
    const double rel = std::abs(static_cast<double>(y) - static_cast<double>(x)) /
                       std::abs(static_cast<double>(x));
    VIBE_CHECK(rel <= u + 1e-12);
  }
  VIBE_CHECK(bfloat16_t::from_float(0.0f).is_zero());
  VIBE_CHECK(bfloat16_t::from_float(std::numeric_limits<float>::infinity()).is_inf());
  VIBE_CHECK(bfloat16_t::from_float(std::numeric_limits<float>::quiet_NaN()).is_nan());
  // 1e30 在 bfloat16 中仍然是有限数（不像 half 会溢出）
  VIBE_CHECK(!bfloat16_t::from_float(1.0e30f).is_inf());
}

/// demote / promote 的组合：降精度再升精度等价于一次降精度
VIBE_TEST(demote_promote_consistency)
{
  const float x = 3.14159265f;
  const half_t h = demote<float, half_t>(x);
  VIBE_CHECK_NEAR(promote<half_t, float>(h), h.to_float(), 1e-9);
  const bfloat16_t b = demote<float, bfloat16_t>(x);
  VIBE_CHECK_NEAR(promote<bfloat16_t, double>(b), static_cast<double>(b.to_float()), 1e-12);
  // double -> float -> double 的精度损失应为 0（x 本身是 float）
  VIBE_CHECK_NEAR(demote<double, float>(static_cast<double>(x)), x, 1e-12);
  // 截断版本的误差不小于舍入版本
  const half_t ht = demote_trunc<float, half_t>(x);
  VIBE_CHECK(std::abs(ht.to_float() - x) >= std::abs(h.to_float() - x) - 1e-6f);
}

// ===========================================================================
// 3. PrecisionTraits 与策略
// ===========================================================================

/// epsilon / max / 位宽与 IEEE-754 定义一致
VIBE_TEST(precision_traits_epsilon_and_max)
{
  VIBE_CHECK_NEAR(PrecisionTraits<Precision::FP16>::epsilon(), std::ldexp(1.0, -10), 1e-15);
  VIBE_CHECK_NEAR(PrecisionTraits<Precision::BF16>::epsilon(), std::ldexp(1.0, -7), 1e-15);
  VIBE_CHECK_NEAR(PrecisionTraits<Precision::FP32>::epsilon(),
                  static_cast<double>(std::numeric_limits<float>::epsilon()), 1e-20);
  VIBE_CHECK_NEAR(PrecisionTraits<Precision::FP64>::epsilon(),
                  std::numeric_limits<double>::epsilon(), 1e-30);

  VIBE_CHECK_NEAR(PrecisionTraits<Precision::FP16>::max(), 65504.0, 1e-6);
  VIBE_CHECK_NEAR(PrecisionTraits<Precision::FP32>::max(),
                  static_cast<double>(std::numeric_limits<float>::max()), 1e20);
  VIBE_CHECK_NEAR(PrecisionTraits<Precision::FP64>::max(),
                  std::numeric_limits<double>::max(), 1e280);

  VIBE_CHECK(precision_bits(Precision::FP16) == 16);
  VIBE_CHECK(precision_bits(Precision::BF16) == 16);
  VIBE_CHECK(precision_bits(Precision::FP32) == 32);
  VIBE_CHECK(precision_bits(Precision::FP64) == 64);
  VIBE_CHECK(precision_max(Precision::FP16, Precision::FP32) == Precision::FP32);
  VIBE_CHECK(PrecisionTraits<Precision::FP16>::is_half());
  VIBE_CHECK(!PrecisionTraits<Precision::FP64>::is_half());
  VIBE_CHECK(std::string(PrecisionTraits<Precision::BF16>::name()) == "BF16");
}

/// 策略的 three-role 约束与默认构造
VIBE_TEST(precision_policy_roles)
{
  const PrecisionPolicy def = PrecisionPolicy::default_policy();
  VIBE_CHECK(def.storage == Precision::FP64);
  VIBE_CHECK(def.compute == Precision::FP32);
  VIBE_CHECK(def.reduce == Precision::FP64);
  VIBE_CHECK(def.compensated_summation);
  VIBE_CHECK(def.iterative_refinement);
  VIBE_CHECK(def.valid());

  const PrecisionPolicy res = PrecisionPolicy::research();
  VIBE_CHECK(res.storage == Precision::FP64 && res.compute == Precision::FP64);
  VIBE_CHECK(res.valid());

  const PrecisionPolicy fast = PrecisionPolicy::fast();
  VIBE_CHECK(fast.storage == Precision::FP64);
  VIBE_CHECK(!fast.compensated_summation);
  VIBE_CHECK(fast.valid());

  // 违反铁律（存储非 FP64 且未显式允许）必须判为非法
  PrecisionPolicy bad;
  bad.storage = Precision::FP32;
  bad.compute = Precision::FP32;
  bad.reduce = Precision::FP32;
  VIBE_CHECK(!bad.valid());
  bad.allow_lossy_storage = true;
  VIBE_CHECK(bad.valid());
  // 归约精度低于计算精度同样非法
  bad.compute = Precision::FP64;
  VIBE_CHECK(!bad.valid());

  VIBE_CHECK(precision_from_string("fp32") == Precision::FP32);
  VIBE_CHECK(precision_from_string("double") == Precision::FP64);
  VIBE_CHECK(precision_from_string("half") == Precision::FP16);
  VIBE_CHECK(precision_from_string("bf16") == Precision::BF16);
}

/// PrecisionGuard 的 RAII 语义与线程局部性
VIBE_TEST(precision_guard_scoped_switch)
{
  const PrecisionPolicy before = current_policy();
  {
    PrecisionGuard guard(Precision::FP64, Precision::FP64, Precision::FP64);
    VIBE_CHECK(current_policy().compute == Precision::FP64);
    VIBE_CHECK(current_policy().storage == Precision::FP64);
    VIBE_CHECK(guard.active());
    PrecisionGuard inner(PrecisionPolicy::half_experiment());
    VIBE_CHECK(current_policy().compute == Precision::BF16);
  }
  VIBE_CHECK(current_policy().compute == before.compute);
  VIBE_CHECK(current_policy().storage == before.storage);

  PrecisionGuard g(PrecisionPolicy::fast());
  g.release();
  VIBE_CHECK(!g.active());
  VIBE_CHECK(current_policy().compute == before.compute);
}

// ===========================================================================
// 4. DeviceBuffer 与内存池
// ===========================================================================

/// 移动语义：移动后源为空、目标持有数据，且不重复释放
VIBE_TEST(device_buffer_move_semantics)
{
  DeviceBuffer<double> a(8, AllocKind::Device);
  VIBE_CHECK(a.size() == 8);
  VIBE_CHECK(a.device_ptr() != nullptr);
  VIBE_CHECK(a.host_ptr() != nullptr);
  double* raw = a.device_ptr();
  DeviceBuffer<double> b(std::move(a));
  VIBE_CHECK(a.empty());
  VIBE_CHECK(a.device_ptr() == nullptr);
  VIBE_CHECK(b.size() == 8);
  VIBE_CHECK(b.device_ptr() == raw);

  DeviceBuffer<double> c;
  c = std::move(b);
  VIBE_CHECK(b.empty());
  VIBE_CHECK(c.size() == 8);
  // CPU 后端下设备指针 == 主机指针（统一地址空间）
  VIBE_CHECK(c.device_ptr() == c.host_ptr());
  // 禁止拷贝：类型层面已 delete，这里用 static_assert 固化契约
  static_assert(!std::is_copy_constructible<DeviceBuffer<double>>::value,
                "DeviceBuffer 必须禁止拷贝");
}

/// fill / zero / copy / resize / assign 的正确性
VIBE_TEST(device_buffer_fill_copy_resize)
{
  DeviceBuffer<double> b(4, AllocKind::Device);
  b.fill(2.5);
  for (std::size_t i = 0; i < b.size(); ++i) VIBE_CHECK_NEAR(b.host_ptr()[i], 2.5, 1e-15);
  b.zero();
  for (std::size_t i = 0; i < b.size(); ++i) VIBE_CHECK_NEAR(b.host_ptr()[i], 0.0, 1e-30);

  const double src[] = {1.0, 2.0, 3.0, 4.0};
  b.assign(src, 4);
  b.copy_to_device();
  double dst[4] = {0, 0, 0, 0};
  b.download(dst, 4);
  for (int i = 0; i < 4; ++i) VIBE_CHECK_NEAR(dst[i], src[i], 1e-15);

  b.resize(16);
  VIBE_CHECK(b.size() == 16);
  b.fill(1.0);
  VIBE_CHECK_NEAR(b.host_ptr()[15], 1.0, 1e-15);
  b.release();
  VIBE_CHECK(b.empty());
  VIBE_CHECK(b.host_ptr() == nullptr);
}

/// 内存统计随分配/释放正确变化，峰值只增不减
VIBE_TEST(memory_stats_tracking)
{
  reset_memory_stats();
  const MemoryStats s0 = memory_stats();
  VIBE_CHECK(s0.current == 0);
  {
    DeviceBuffer<float> b(256, AllocKind::Device);  // 1 KiB 设备 + 1 KiB 主机镜像
    const MemoryStats s1 = memory_stats();
    VIBE_CHECK(s1.current >= 2048);
    VIBE_CHECK(s1.allocations >= 2);
    VIBE_CHECK(s1.peak >= s1.current);
    b.release();
    const MemoryStats s2 = memory_stats();
    VIBE_CHECK(s2.current < s1.current || s2.current == 0);
    VIBE_CHECK(s2.deallocations >= 2);
  }
  const MemoryStats s3 = memory_stats();
  VIBE_CHECK(s3.current == 0);
  VIBE_CHECK(!memory_stats_string().empty());
}

/// 内存池：acquire/release 复用同一块，命中次数增加
VIBE_TEST(memory_pool_reuse_statistics)
{
  MemoryPool pool(16u * 1024u * 1024u);
  pool.clear();
  pool.reset_stats();
  {
    DeviceBuffer<double> a = pool.acquire<double>(1024);
    VIBE_CHECK(a.size() >= 1024);
    void* first = a.device_ptr();
    pool.release(a);
    VIBE_CHECK(a.empty());
    VIBE_CHECK(pool.cached_blocks() >= 1);

    DeviceBuffer<double> b = pool.acquire<double>(1024);
    VIBE_CHECK(b.device_ptr() == first);  // 精确命中 -> 复用同一块
    const MemoryStats st = pool.stats();
    VIBE_CHECK(st.pool_hits >= 1);
    VIBE_CHECK(st.pool_misses >= 1);
    pool.release(b);
  }
  VIBE_CHECK(pool.cached_bytes() > 0);
  pool.clear();
  VIBE_CHECK(pool.cached_bytes() == 0);
  VIBE_CHECK(pool.cached_blocks() == 0);
}

/// FieldMirror 的按字节上传/下载
VIBE_TEST(field_mirror_roundtrip)
{
  const Real host[6] = {1, 2, 3, 4, 5, 6};
  FieldMirror m(sizeof(host));
  VIBE_CHECK(m.bytes() == sizeof(host));
  VIBE_CHECK(m.count<Real>() == 6);
  m.upload(host, sizeof(host));
  Real out[6] = {0, 0, 0, 0, 0, 0};
  m.download(out, sizeof(out));
  for (int i = 0; i < 6; ++i) VIBE_CHECK_NEAR(out[i], host[i], 1e-15);
  const Real* dev = m.device_ptr<Real>();
  VIBE_CHECK_NEAR(dev[5], Real(6), 1e-15);
}

// ===========================================================================
// 5. LaunchConfig
// ===========================================================================

/// 构造助手与派生量的语义
VIBE_TEST(launch_config_construction)
{
  const LaunchConfig c1 = make_1d(1000, 128);
  VIBE_CHECK(c1.grid.x == 1000 && c1.grid.y == 1 && c1.grid.z == 1);
  VIBE_CHECK(c1.block.x == 128);
  VIBE_CHECK(c1.total_grid() == 1000);
  VIBE_CHECK(c1.threads_per_block() == 128);
  VIBE_CHECK(c1.block_count() == (1000 + 127) / 128);
  VIBE_CHECK(c1.shared_bytes == 0);
  VIBE_CHECK(c1.stream == nullptr);

  const LaunchConfig c2 = make_2d(64, 32, 16, 8);
  VIBE_CHECK(c2.grid.x == 64 && c2.grid.y == 32 && c2.grid.z == 1);
  VIBE_CHECK(c2.block.x == 16 && c2.block.y == 8);
  VIBE_CHECK(c2.threads_per_block() == 128);

  const LaunchConfig c3 = make_3d(8, 8, 8, 4, 4, 4);
  VIBE_CHECK(c3.total_grid() == 512);
  VIBE_CHECK(c3.threads_per_block() == 64);

  const LaunchConfig c4 = make_1d_strided(100000, 256, 4);
  VIBE_CHECK(c4.block.x == 256);
  // 每个执行单元覆盖 256*4 个元素
  VIBE_CHECK(c4.block_count() == (100000 + 1023) / 1024);
  VIBE_CHECK(c4.consistent());
}

/// for_field 按含 halo 的存储维度生成配置，并按 warp 尺寸夹紧 block
VIBE_TEST(launch_config_for_field)
{
  const LaunchConfig c = for_field_dims(34, 18, 6, 128);
  VIBE_CHECK(c.grid.x == 34 && c.grid.y == 18 && c.grid.z == 6);
  VIBE_CHECK(c.block.x >= 32 && c.block.x <= 1024);
  // 索引空间线性化必须是 x 最快
  VIBE_CHECK(c.total_grid() == 34 * 18 * 6);
  const LaunchConfig huge = for_field_dims(100, 100, 100, 4096);
  VIBE_CHECK(huge.block.x <= 1024);
  const LaunchConfig tiny = for_field_dims(10, 10, 10, 1);
  VIBE_CHECK(tiny.block.x >= 32);  // 不小于一个 warp
}

/// for_each_index_3d 的线性化顺序与访问次数
VIBE_TEST(launch_index_iteration_orders)
{
  LaunchConfig cfg = make_1d(5 * 4 * 3, 8);
  cfg.grid = make_dim3(5, 4, 3);
  std::vector<int> order;
  order.reserve(60);
  for_each_index_3d(cfg, [&](unsigned int, unsigned int, unsigned int) { order.push_back(1); });
  VIBE_CHECK(order.size() == 60);
  // 单元素配置：验证 (i,j,k) 的展开顺序
  LaunchConfig one = make_1d(1, 1);
  one.grid = make_dim3(3, 1, 1);
  std::vector<int> idx;
  for_each_index_3d(one, [&](unsigned int i, unsigned int j, unsigned int k) {
    idx.push_back(static_cast<int>(i * 100 + j * 10 + k));
  });
  VIBE_CHECK(idx.size() == 3);
  VIBE_CHECK(idx[0] == 0 && idx[1] == 100 && idx[2] == 200);
}

// ===========================================================================
// 6. CPU 内核
// ===========================================================================

/// 常系数三对角：解必须与解析解一致（\f$ x_k = 1 \f$ 时构造右端项）
VIBE_TEST(tridiagonal_constant_coefficients)
{
  const std::size_t nz = 16;
  const Real a = -1.0, b = 2.0, c = -1.0;
  std::vector<Real> av(nz, a), bv(nz, b), cv(nz, c), dv(nz, 0.0), x;
  std::vector<Real> exact(nz);
  for (std::size_t k = 0; k < nz; ++k) {
    exact[k] = std::sin(0.1 * static_cast<double>(k + 1));
    Real rhs = b * exact[k];
    if (k > 0) rhs += a * exact[k - 1];
    if (k + 1 < nz) rhs += c * exact[k + 1];
    dv[k] = rhs;
  }
  tridiagonal_solve(av, bv, cv, dv, x);
  VIBE_CHECK(x.size() == nz);
  for (std::size_t k = 0; k < nz; ++k) VIBE_CHECK_NEAR(x[k], exact[k], 1e-10);
}

/// 批量三对角：多个系统必须互不干扰
VIBE_TEST(tridiagonal_batched_systems)
{
  const std::size_t n_sys = 3;
  const std::size_t nz = 4;
  // 三个系统分别对应解向量 x = (1,2,3,4) * s，s = 1,2,3
  const Real diag = 4.0, off = -1.0;
  std::vector<Real> a(n_sys * nz, off), b(n_sys * nz, diag), c(n_sys * nz, off);
  std::vector<Real> d(n_sys * nz, 0.0), x(n_sys * nz, 0.0);
  for (std::size_t s = 0; s < n_sys; ++s) {
    for (std::size_t k = 0; k < nz; ++k) {
      const Real val = static_cast<Real>(k + 1) * static_cast<Real>(s + 1);
      Real rhs = diag * val;
      if (k > 0) rhs += off * (static_cast<Real>(k) * static_cast<Real>(s + 1));
      if (k + 1 < nz) rhs += off * (static_cast<Real>(k + 2) * static_cast<Real>(s + 1));
      d[s * nz + k] = rhs;
    }
  }
  tridiagonal_solve(a.data(), b.data(), c.data(), d.data(), x.data(), n_sys, nz);
  for (std::size_t s = 0; s < n_sys; ++s) {
    for (std::size_t k = 0; k < nz; ++k) {
      VIBE_CHECK_NEAR(x[s * nz + k], static_cast<Real>(k + 1) * static_cast<Real>(s + 1), 1e-10);
    }
  }
}

/// 非对角占优的系统必须显式报错，而不是给出不可信的解（[B11] 第 4 章）
VIBE_TEST(tridiagonal_rejects_non_dominant)
{
  std::vector<Real> a = {0.0, 5.0}, b = {1.0, 1.0}, c = {5.0, 0.0}, d = {1.0, 1.0}, x(2, 0.0);
  VIBE_CHECK_THROWS(tridiagonal_solve(a.data(), b.data(), c.data(), d.data(), x.data(), 1, 2));
}

/// 限制与延拓的守恒性：\f$ \sum \text{coarse} = \frac{1}{r^2}\sum \text{fine} \f$
/// 且常数场在 restrict/prolong 下保持不变（[N4][N5]）
VIBE_TEST(restrict_prolong_conservation)
{
  const vibe::Int r = 3;
  const vibe::Int nxc = 4, nyc = 3, nzc = 2;
  const FieldShape cf = make_shape(nxc, nyc, nzc);
  const FieldShape ff = make_shape(nxc * r, nyc * r, nzc);
  std::vector<Real> fine(ff.size(), 0.0), coarse(cf.size(), 0.0);

  // 常数场：restrict 必须给出同一个常数（权重 1/r^2 之和为 1）
  std::fill(fine.begin(), fine.end(), 7.5);
  restrict_2to1(fine.data(), coarse.data(), ff, cf, r);
  for (std::size_t i = 0; i < coarse.size(); ++i) VIBE_CHECK_NEAR(coarse[i], 7.5, 1e-12);

  // 任意场：总量守恒
  for (std::size_t i = 0; i < fine.size(); ++i) fine[i] = std::sin(0.3 * static_cast<double>(i));
  std::fill(coarse.begin(), coarse.end(), 0.0);
  restrict_2to1(fine.data(), coarse.data(), ff, cf, r);
  const Real sum_coarse = reduce_sum(coarse.data(), coarse.size());
  const Real sum_fine = reduce_sum(fine.data(), fine.size());
  VIBE_CHECK_NEAR(sum_coarse, sum_fine / static_cast<Real>(r * r), 1e-10);

  // 延拓保持常数（单元中心偏移 0.5/r 的正确性检验）
  std::fill(coarse.begin(), coarse.end(), -2.25);
  std::vector<Real> back(ff.size(), 0.0);
  prolong_1to2(coarse.data(), back.data(), cf, ff, r);
  for (std::size_t i = 0; i < back.size(); ++i) VIBE_CHECK_NEAR(back[i], -2.25, 1e-12);
}

/// restrict(prolong(x)) == x：对光滑场误差应为高阶小量
VIBE_TEST(restrict_of_prolong_is_identity_on_smooth)
{
  const vibe::Int r = 3;
  const vibe::Int nxc = 4, nyc = 3, nzc = 1;
  const FieldShape cf = make_shape(nxc, nyc, nzc);
  const FieldShape ff = make_shape(nxc * r, nyc * r, nzc);
  std::vector<Real> coarse(cf.size(), 0.0), fine(ff.size(), 0.0), roundtrip(cf.size(), 0.0);
  for (vibe::Int j = 0; j < nyc; ++j) {
    for (vibe::Int i = 0; i < nxc; ++i) {
      coarse[cf.offset(i, j, 0)] = 1.0 + 0.5 * static_cast<Real>(i) - 0.25 * static_cast<Real>(j);
    }
  }
  prolong_1to2(coarse.data(), fine.data(), cf, ff, r);
  restrict_2to1(fine.data(), roundtrip.data(), ff, cf, r);
  for (std::size_t i = 0; i < coarse.size(); ++i) VIBE_CHECK_NEAR(roundtrip[i], coarse[i], 1e-10);
}

/// 双线性插值对线性场精确
VIBE_TEST(bilinear_interp_is_exact_on_linear_field)
{
  const vibe::Int nx = 10, ny = 8, nz = 2;
  const KernelGeometry g = make_geom(nx, ny, nz, 100.0, 100.0, 10.0);
  const FieldShape s = make_shape(nx, ny, nz);
  // f = 2 + 0.01*x + 0.02*y（以米为单位，格距 100 m）
  auto field = fill_field(s, [&](vibe::Int i, vibe::Int j, vibe::Int) {
    return Real(2) + Real(0.01) * (static_cast<Real>(i) * g.dx) +
           Real(0.02) * (static_cast<Real>(j) * g.dy);
  });
  for (Real x : {250.0, 123.4, 0.0, 400.0}) {
    for (Real y : {50.0, 333.3, 700.0}) {
      const Real expect = Real(2) + Real(0.01) * x + Real(0.02) * y;
      // 允许单元边界处的钳制（x=400 落在内部）
      VIBE_CHECK_NEAR(bilinear_interp(field.data(), s, g, 0, x, y), expect, 0.5);
    }
  }
  // 格点上的值必须精确复现
  for (vibe::Int i = 1; i < nx - 1; ++i) {
    VIBE_CHECK_NEAR(bilinear_interp(field.data(), s, g, 1, static_cast<Real>(i) * g.dx,
                                    static_cast<Real>(2) * g.dy),
                    field[s.offset(i, 2, 1)], 1e-9);
  }
}

/// 三线性插值：与双线性在 k 层上的结果一致（当 z 正好落在层中心时）
VIBE_TEST(trilinear_interp_consistency)
{
  const vibe::Int nx = 6, ny = 6, nz = 4;
  const KernelGeometry g = make_geom(nx, ny, nz, 10.0, 10.0, 5.0);
  const FieldShape s = make_shape(nx, ny, nz);
  auto field = fill_field(s, [&](vibe::Int i, vibe::Int j, vibe::Int k) {
    return Real(i) + Real(10) * static_cast<Real>(j) + Real(100) * static_cast<Real>(k);
  });
  const Real x = static_cast<Real>(2) * g.dx;
  const Real y = static_cast<Real>(3) * g.dy;
  const Real z = static_cast<Real>(2) * g.dz;
  VIBE_CHECK_NEAR(trilinear_interp(field.data(), s, g, x, y, z), field[s.offset(2, 3, 2)], 1e-9);
  // z 取层界面时是上下层的平均
  VIBE_CHECK_NEAR(trilinear_interp(field.data(), s, g, x, y, z + Real(0.5) * g.dz),
                  Real(0.5) * (field[s.offset(2, 3, 2)] + field[s.offset(2, 3, 3)]), 1e-9);
}

/// 归约：与补偿求和一致，且对 1e6 个 1.0 精确
VIBE_TEST(reduce_sum_matches_compensated)
{
  const std::size_t n = 1000000;
  std::vector<Real> v(n, Real(1));
  const Real s = reduce_sum(v.data(), n);
  VIBE_CHECK_NEAR(s, static_cast<Real>(n), 1e-6);
  const Real sk = reduce_sum(v.data(), n, SumAlgorithm::Kahan);
  VIBE_CHECK_NEAR(sk, static_cast<Real>(n), 1e-6);
  VIBE_CHECK_NEAR(reduce_mean(v.data(), n), Real(1), 1e-12);
  VIBE_CHECK_NEAR(reduce_max(v.data(), n), Real(1), 1e-15);

  // 病态序列：大数 + 大量小数
  std::vector<Real> w(n + 2);
  w[0] = Real(1e16);
  for (std::size_t i = 1; i <= n; ++i) w[i] = Real(1);
  w[n + 1] = Real(-1e16);
  VIBE_CHECK_NEAR(reduce_sum(w.data(), w.size()), static_cast<Real>(n), 1e-3);
}

/// 步伐归约与 min/max
VIBE_TEST(reduce_strided_and_minmax)
{
  std::vector<Real> v = {1, 2, 3, 4, 5, 6, 7, 8};
  VIBE_CHECK_NEAR(reduce_sum_strided(v.data(), 4, 2), Real(1 + 3 + 5 + 7), 1e-12);
  Real lo = 0, hi = 0;
  reduce_minmax(v.data(), v.size(), lo, hi);
  VIBE_CHECK_NEAR(lo, Real(1), 1e-15);
  VIBE_CHECK_NEAR(hi, Real(8), 1e-15);
  // 空输入
  VIBE_CHECK_NEAR(reduce_sum(v.data(), 0), Real(0), 1e-30);
  VIBE_CHECK(reduce_max(v.data(), 0) < Real(0));
}

/// 平流内核：常速度场对常标量场的平流趋势必须为零（保常数性）
VIBE_TEST(advect_scalar_preserves_constant)
{
  const vibe::Int nx = 8, ny = 8, nz = 4;
  const KernelGeometry g = make_geom(nx, ny, nz, 100.0, 100.0, 20.0);
  const FieldShape s = make_shape(nx, ny, nz);
  // 内部点常值，halo 用周期填充（用平铺复制模拟）
  auto q = fill_field(s, [&](vibe::Int, vibe::Int, vibe::Int) { return Real(3.5); });
  auto u = fill_field(s, [&](vibe::Int, vibe::Int, vibe::Int) { return Real(12.0); });
  auto v = fill_field(s, [&](vibe::Int, vibe::Int, vibe::Int) { return Real(-4.0); });
  auto w = fill_field(s, [&](vibe::Int, vibe::Int, vibe::Int) { return Real(0.5); });
  auto rho = fill_field(s, [&](vibe::Int, vibe::Int, vibe::Int) { return Real(1.0); });
  std::vector<Real> out(s.size(), Real(0));
  AdvectionOptions opt;
  opt.order = 2;
  advect_scalar(q.data(), u.data(), v.data(), w.data(), rho.data(), out.data(), s, g, opt);
  for (std::size_t i = 0; i < out.size(); ++i) VIBE_CHECK_NEAR(out[i], Real(0), 1e-12);

  // 4 阶与 WENO5 同样保常数
  opt.order = 4;
  advect_scalar(q.data(), u.data(), v.data(), w.data(), rho.data(), out.data(), s, g, opt);
  for (std::size_t i = 0; i < out.size(); ++i) VIBE_CHECK_NEAR(out[i], Real(0), 1e-10);
}

/// 扩散：Laplacian 对二次场给出常数二阶导
VIBE_TEST(diffusion_laplacian_of_quadratic)
{
  const vibe::Int nx = 9, ny = 9, nz = 9;
  const Real dx = 2.0;
  const KernelGeometry g = make_geom(nx, ny, nz, dx, dx, dx);
  const FieldShape s = make_shape(nx, ny, nz);
  // q = x^2 -> 3D Laplacian = 2（仅 x 方向贡献）
  auto q = fill_field(s, [&](vibe::Int i, vibe::Int, vibe::Int) {
    const Real x = static_cast<Real>(i) * dx;
    return x * x;
  });
  std::vector<Real> out(s.size(), Real(0));
  diffusion(q.data(), out.data(), nullptr, s, g, Real(1), DiffusionKind::Laplacian);
  // 内部点（1..nx-2）处必须是 2
  for (vibe::Int k = 1; k < nz - 1; ++k) {
    for (vibe::Int j = 1; j < ny - 1; ++j) {
      for (vibe::Int i = 1; i < nx - 1; ++i) {
        VIBE_CHECK_NEAR(out[s.offset(i, j, k)], Real(2), 1e-9);
      }
    }
  }
}

/// Helmholtz 残差：x 为算子精确解时残差为零
VIBE_TEST(helmholtz_residual_of_exact_solution)
{
  const vibe::Int nx = 6, ny = 6, nz = 6;
  const KernelGeometry g = make_geom(nx, ny, nz, 1.0, 1.0, 1.0);
  const FieldShape s = make_shape(nx, ny, nz);
  // 常系数 a = 1、b = 2，解 x = 常数 -> L x = -b x
  auto a = fill_field(s, [&](vibe::Int, vibe::Int, vibe::Int) { return Real(1.0); });
  auto b = fill_field(s, [&](vibe::Int, vibe::Int, vibe::Int) { return Real(2.0); });
  auto x = fill_field(s, [&](vibe::Int, vibe::Int, vibe::Int) { return Real(3.0); });
  auto f = fill_field(s, [&](vibe::Int, vibe::Int, vibe::Int) { return Real(-6.0); });
  std::vector<Real> r(s.size(), Real(123));
  helmholtz_residual(x.data(), a.data(), b.data(), f.data(), r.data(), s, g);
  for (vibe::Int k = 1; k < nz - 1; ++k) {
    for (vibe::Int j = 1; j < ny - 1; ++j) {
      for (vibe::Int i = 1; i < nx - 1; ++i) VIBE_CHECK_NEAR(r[s.offset(i, j, k)], Real(0), 1e-10);
    }
  }
}

// ===========================================================================
// 7. 后端与设备
// ===========================================================================

/// 后端枚举的字符串互转（大小写不敏感 + 别名）
VIBE_TEST(backend_string_roundtrip)
{
  VIBE_CHECK(VIBE_BACKEND_CPU + VIBE_BACKEND_CUDA + VIBE_BACKEND_HIP + VIBE_BACKEND_SYCL == 1);
  for (Backend b : {Backend::CPU, Backend::CUDA, Backend::HIP, Backend::SYCL}) {
    bool ok = false;
    const Backend parsed = from_string(to_string(b), &ok);
    VIBE_CHECK(ok);
    VIBE_CHECK(parsed == b);
  }
  bool ok = false;
  from_string("nonsense", &ok);
  VIBE_CHECK(!ok);
  VIBE_CHECK(compiled_backend() == active_backend());
  VIBE_CHECK(is_compiled(compiled_backend()));
  // CSV 构建下 CPU 必须是编译后端
#if VIBE_BACKEND_CPU
  VIBE_CHECK(active_backend() == Backend::CPU);
  VIBE_CHECK(!is_accelerator(Backend::CPU));
#endif
}

/// 后端能力矩阵的保守性
VIBE_TEST(backend_caps_matrix)
{
  const BackendCaps cpu = backend_caps(Backend::CPU);
  VIBE_CHECK(!cpu.accelerator);
  VIBE_CHECK(cpu.unified_memory);       // 同一地址空间
  VIBE_CHECK(cpu.atomic_float);
  VIBE_CHECK(!cpu.tensor_core);
  VIBE_CHECK(!cpu.events_timing);

  const BackendCaps cuda = backend_caps(Backend::CUDA);
  VIBE_CHECK(cuda.accelerator);
  VIBE_CHECK(cuda.async_streams);
  VIBE_CHECK(cuda.events_timing);
  VIBE_CHECK(cuda.pinned_memory);

  const BackendCaps hip = backend_caps(Backend::HIP);
  VIBE_CHECK(hip.accelerator && hip.unified_memory);
  const BackendCaps sycl = backend_caps(Backend::SYCL);
  VIBE_CHECK(sycl.accelerator && sycl.unified_memory);
}

/// 设备枚举：CPU 后端恰好一个伪设备，字段自洽
VIBE_TEST(device_enumeration)
{
  const auto devices = enumerate_devices();
  VIBE_CHECK(!devices.empty());
  VIBE_CHECK(device_count() == static_cast<int>(devices.size()));
  const DeviceInfo& d0 = devices.front();
  VIBE_CHECK(d0.backend == compiled_backend());
  VIBE_CHECK(d0.id == 0);
  VIBE_CHECK(d0.memory_bytes > 0);
  VIBE_CHECK(!d0.name.empty());
  VIBE_CHECK(!d0.describe().empty());
  VIBE_CHECK(d0.compute_capability() == d0.compute_major * 10 + d0.compute_minor);

  Device dev;
  VIBE_CHECK(dev.valid());
  VIBE_CHECK(dev.id() == 0);
  VIBE_CHECK(dev.warp_size() >= 1);
  VIBE_CHECK(dev.max_threads_per_block() >= 32);
  dev.synchronize();
  dev.make_current();
  const double occ = dev.occupancy(32, 0, 256);
  VIBE_CHECK(occ >= 0.0 && occ <= 1.0 + 1e-12);
  VIBE_CHECK_THROWS(Device(active_backend(), 9999));
}

/// 内核计时表：按名字聚合
VIBE_TEST(kernel_timings_registry)
{
  KernelTimings& t = kernel_timings();
  t.reset();
  const LaunchConfig cfg = make_1d(1024, 256);
  t.record("advect_scalar", 1.5, cfg);
  t.record("advect_scalar", 2.5, cfg);
  t.record("diffusion", 0.75, cfg);
  VIBE_CHECK(t.entries.size() == 2);
  const auto& adv = t.get("advect_scalar");
  VIBE_CHECK(adv.calls == 2);
  VIBE_CHECK_NEAR(adv.elapsed_ms, 4.0, 1e-12);
  VIBE_CHECK_NEAR(adv.mean_ms(), 2.0, 1e-12);
  VIBE_CHECK_NEAR(t.total_ms(), 4.75, 1e-12);
  VIBE_CHECK(!t.report().empty());
  const auto sorted = t.sorted();
  VIBE_CHECK(sorted.size() == 2);
  VIBE_CHECK(sorted[0].elapsed_ms >= sorted[1].elapsed_ms);
  t.reset();
  VIBE_CHECK(t.entries.empty());
}

/// time_kernel 在关闭计时时不产生记录，开启时记录一次
VIBE_TEST(time_kernel_helper)
{
  KernelTimings& t = kernel_timings();
  t.reset();
  const LaunchConfig cfg = make_1d(16, 16);
  const double ms = time_kernel("noop", cfg, []() {});
  if (timing_enabled()) {
    VIBE_CHECK(t.get("noop").calls == 1);
    VIBE_CHECK(ms >= 0.0);
  } else {
    VIBE_CHECK(t.entries.empty());
  }
}
