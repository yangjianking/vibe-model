#pragma once
/// @file observations.hpp
/// @brief 观测的数据模型、观测空间与观测文件读写。
///
/// 约定
/// ----
///   * 时间统一为"相对同化窗口起点的秒数"，避免时区与日历问题；
///   * 位置为模式的物理坐标（米），由 `obs::Locator` 负责经纬度->投影坐标；
///   * 每个观测携带 `sigma`（观测误差标准差）与 `qc_flag`；
///   * 观测误差协方差 R 假定对角（对角近似），因此只存 1/sigma^2。
///
/// 观测类型与变量对应（[O8][O9]）
/// ------------------------------
///   探空 Radiosonde      : U, V, T, Q, P
///   地面 Surface         : PS, T2m, Q2m, U10, V10
///   飞机 Aircraft        : U, V, T
///   卫星导风 AMV         : U, V
///   散射计 Scatterometer : U10, V10
///   掩星 GnssRo          : Refractivity（折射率）
///   卫星辐射率 Radiance  : Radiance（通道亮度温度）
///   雷达 RadarReflectivity: Reflectivity
///
/// 文献：[O8] Lorenc et al. (2000)；[O9] WMO-No.8；[V14] Ide et al. (1997) 统一记号。

#include <algorithm>
#include <string>
#include <vector>

#include "vibe/common/types.hpp"
#include "vibe/config/config.hpp"

namespace vibe::obs {

/// 观测类型
enum class ObsType {
  Radiosonde = 0, Surface, Aircraft, AMV, Scatterometer,
  GnssRo, Radiance, RadarReflectivity, Profiler, Count
};

/// 观测变量种类
enum class VarKind {
  U = 0, V, W, T, Q, PS, P, Radiance, Refractivity, Reflectivity, Count
};

const char* to_string(ObsType t) noexcept;
const char* to_string(VarKind v) noexcept;
ObsType obs_type_from_string(const std::string& s);
VarKind var_kind_from_string(const std::string& s);

/// 单条观测
struct Observation {
  ObsType type = ObsType::Radiosonde;
  Real    time = Real(0);        ///< 相对窗口起点的秒数
  Real    x = Real(0), y = Real(0), z = Real(0);
  Real    lon = Real(0), lat = Real(0);   ///< 原始经纬度（诊断/输出用）
  VarKind variable = VarKind::T;
  int     channel = -1;          ///< 卫星通道号（非辐射率为 -1）
  Real    value = Real(0);
  Real    sigma = Real(1);       ///< 观测误差标准差
  Real    bias = Real(0);        ///< 已估计的系统偏差（O-B 的系统部分）
  int     qc_flag = 0;           ///< 0 可用；>0 为剔除原因码
  Index   station_id = -1;
  Index   record_id = -1;        ///< 同一次探空的记录序号
  int     level_type = 0;        ///< 0 = 高度, 1 = 气压层

  /// 有效观测：已通过 QC 且 sigma > 0
  bool usable() const noexcept { return qc_flag == 0 && sigma > Real(0); }

  /// 去偏后的观测值
  Real unbiased() const noexcept { return value - bias; }
};

/// 观测空间
class ObsSpace {
 public:
  std::vector<Observation> obs;
  Real window_start = Real(0);
  Real window_length = Real(0);
  std::string source;                 ///< 文件名/说明
  std::string valid_time;

  Size size() const noexcept { return obs.size(); }
  /// 可用观测数
  Size usable_count() const noexcept;
  /// 按类型分组计数
  std::vector<Size> count_by_type() const;
  /// 按变量分组计数
  std::vector<Size> count_by_variable() const;

  /// 构造对角 R^{-1}（只含可用观测）
  std::vector<Real> inverse_variance() const;

  /// 只保留满足谓词的观测，返回保留数量
  template <class Pred>
  Size filter(Pred p) {
    const Size before = obs.size();
    obs.erase(std::remove_if(obs.begin(), obs.end(), [&](const Observation& o) { return !p(o); }),
              obs.end());
    return before - obs.size();
  }

  /// 按类型抽取子集
  ObsSpace subset(ObsType t) const;
  /// 按时间时隙抽取（同化 4D-Var 的分组）
  std::vector<ObsSpace> by_slot(int n_slots) const;

  std::string describe() const;
};

/// 观测文件读写
///   文本格式：CSV，表头 type,time,x,y,z,var,channel,value,sigma,qc
class ObsReader {
 public:
  static ObsSpace read_csv(const std::string& path, Real window_length = Real(0));
  static ObsSpace read_binary(const std::string& path);
  /// 依据扩展名自动选择
  static ObsSpace read(const std::string& path, Real window_length = Real(0));
};

void write_csv(const std::string& path, const ObsSpace& os);

/// 生成合成观测（OSE / 理想试验用）：从模式状态用观测算子采样，加上随机误差
struct SyntheticObsOptions {
  std::vector<ObsType> types{ObsType::Radiosonde, ObsType::Surface};
  Real density_fraction = Real(0.01);   ///< 抽稀比例
  bool add_noise = true;
  unsigned seed = 42;
  Real time = Real(0);
};
ObsSpace generate_synthetic_observations(const struct ObservationOperator& op,
                                         const struct ModelStateView& view,
                                         const SyntheticObsOptions& opt);

}  // namespace vibe::obs
