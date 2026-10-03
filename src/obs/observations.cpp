/// @file observations.cpp
/// @brief ObsSpace 统计、时间分箱、CSV/二进制读写与合成观测生成。
///
/// 设计要点
/// --------
///   * 时间统一为"相对同化窗口起点的秒数"，见 observations.hpp；
///   * CSV 采用显式表头 (type,time,x,y,z,var,channel,value,sigma,qc)，
///     写出时对含分隔符/引号/换行的字段做 RFC4180 转义（[O9] WMO-No.8
///     只规定观测内容，文件布局由本项目约定）；
///   * 合成观测用固定种子 std::mt19937，保证 OSE 试验可复现。
///
/// 复杂度：统计 O(n)；by_slot O(n + n_slots)；CSV 读写 O(n)。
///
/// 文献：[O8] Lorenc et al. (2000)；[O9] WMO-No.8；[V14] Ide et al. (1997)。

#include "vibe/obs/observations.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/common/logging.hpp"
#include "vibe/dyn/state.hpp"
#include "vibe/grid/geometry.hpp"
#include "vibe/obs/obs_operator.hpp"

namespace vibe::obs {

// ===========================================================================
// 枚举 <-> 字符串
// ===========================================================================

const char* to_string(ObsType t) noexcept {
  switch (t) {
    case ObsType::Radiosonde:        return "radiosonde";
    case ObsType::Surface:           return "surface";
    case ObsType::Aircraft:          return "aircraft";
    case ObsType::AMV:               return "amv";
    case ObsType::Scatterometer:     return "scatterometer";
    case ObsType::GnssRo:            return "gnssro";
    case ObsType::Radiance:          return "radiance";
    case ObsType::RadarReflectivity: return "radar";
    case ObsType::Profiler:          return "profiler";
    case ObsType::Count:             return "count";
  }
  return "unknown";
}

const char* to_string(VarKind v) noexcept {
  switch (v) {
    case VarKind::U:            return "u";
    case VarKind::V:            return "v";
    case VarKind::W:            return "w";
    case VarKind::T:            return "t";
    case VarKind::Q:            return "q";
    case VarKind::PS:           return "ps";
    case VarKind::P:            return "p";
    case VarKind::Radiance:     return "radiance";
    case VarKind::Refractivity: return "refractivity";
    case VarKind::Reflectivity: return "reflectivity";
    case VarKind::Count:        return "count";
  }
  return "unknown";
}

namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

/// 去掉大小写、'_'、'-'、空格后比较，容忍命名差异
std::string canonical(const std::string& s) {
  std::string x;
  for (char c : lower(s)) {
    if (c != '_' && c != '-' && c != ' ') x.push_back(c);
  }
  return x;
}

}  // namespace

ObsType obs_type_from_string(const std::string& s) {
  for (int i = 0; i < static_cast<int>(ObsType::Count); ++i) {
    const auto t = static_cast<ObsType>(i);
    if (canonical(s) == canonical(to_string(t))) return t;
  }
  const std::string c = canonical(s);
  if (c == "sonde" || c == "raob" || c == "sounding" || c == "temp") return ObsType::Radiosonde;
  if (c == "synop" || c == "station" || c == "sfc") return ObsType::Surface;
  if (c == "satob" || c == "wind" || c == "atovs") return ObsType::AMV;
  if (c == "scat") return ObsType::Scatterometer;
  if (c == "ro" || c == "gpsro" || c == "gnss" || c == "gnssro") return ObsType::GnssRo;
  if (c == "rad" || c == "sat" || c == "bt" || c == "radiance") return ObsType::Radiance;
  if (c == "dbz" || c == "reflectivity" || c == "radarreflectivity") {
    return ObsType::RadarReflectivity;
  }
  VIBE_CHECK_MSG(false, "未知观测类型: " + s);
  return ObsType::Radiosonde;
}

VarKind var_kind_from_string(const std::string& s) {
  for (int i = 0; i < static_cast<int>(VarKind::Count); ++i) {
    const auto v = static_cast<VarKind>(i);
    if (canonical(s) == canonical(to_string(v))) return v;
  }
  const std::string c = canonical(s);
  if (c == "temperature" || c == "temp" || c == "ta") return VarKind::T;
  if (c == "humidity" || c == "qv" || c == "sh") return VarKind::Q;
  if (c == "uwnd" || c == "windu") return VarKind::U;
  if (c == "vwnd" || c == "windv") return VarKind::V;
  if (c == "pressure" || c == "pres") return VarKind::P;
  if (c == "surfacepressure" || c == "sp" || c == "psfc") return VarKind::PS;
  if (c == "n" || c == "ref" || c == "bending") return VarKind::Refractivity;
  if (c == "z" || c == "dbz") return VarKind::Reflectivity;
  VIBE_CHECK_MSG(false, "未知观测变量: " + s);
  return VarKind::T;
}

// ===========================================================================
// ObsSpace
// ===========================================================================

Size ObsSpace::usable_count() const noexcept {
  Size n = 0;
  for (const auto& o : obs) {
    if (o.usable()) ++n;
  }
  return n;
}

std::vector<Size> ObsSpace::count_by_type() const {
  std::vector<Size> c(static_cast<Size>(ObsType::Count), 0);
  for (const auto& o : obs) {
    const auto i = static_cast<Size>(o.type);
    if (i < c.size()) ++c[i];
  }
  return c;
}

std::vector<Size> ObsSpace::count_by_variable() const {
  std::vector<Size> c(static_cast<Size>(VarKind::Count), 0);
  for (const auto& o : obs) {
    const auto i = static_cast<Size>(o.variable);
    if (i < c.size()) ++c[i];
  }
  return c;
}

std::vector<Real> ObsSpace::inverse_variance() const {
  std::vector<Real> r(obs.size(), Real(0));
  for (Size i = 0; i < obs.size(); ++i) {
    const auto& o = obs[i];
    r[i] = o.usable() ? Real(1) / (o.sigma * o.sigma) : Real(0);
  }
  return r;
}

ObsSpace ObsSpace::subset(ObsType t) const {
  ObsSpace out;
  out.window_start = window_start;
  out.window_length = window_length;
  out.source = source;
  out.valid_time = valid_time;
  for (const auto& o : obs) {
    if (o.type == t) out.obs.push_back(o);
  }
  return out;
}

std::vector<ObsSpace> ObsSpace::by_slot(int n_slots) const {
  VIBE_CHECK(n_slots > 0);
  std::vector<ObsSpace> slots(static_cast<Size>(n_slots));
  for (auto& s : slots) {
    s.window_start = window_start;
    s.window_length = window_length;
    s.source = source;
    s.valid_time = valid_time;
  }
  if (obs.empty()) return slots;

  // 时隙宽度：优先使用同化窗口长度；未设置时回退到观测时间跨度
  Real span = window_length;
  if (!(span > Real(0))) {
    Real tmin = obs.front().time, tmax = obs.front().time;
    for (const auto& o : obs) {
      tmin = std::min(tmin, o.time);
      tmax = std::max(tmax, o.time);
    }
    span = tmax - tmin;
  }
  if (!(span > Real(0))) {
    for (const auto& o : obs) slots[0].obs.push_back(o);
    return slots;
  }

  const Real dt = span / static_cast<Real>(n_slots);
  for (const auto& o : obs) {
    int bin = static_cast<int>(std::floor((o.time - window_start) / dt));
    bin = clamp(bin, 0, n_slots - 1);
    slots[static_cast<Size>(bin)].obs.push_back(o);
  }
  return slots;
}

std::string ObsSpace::describe() const {
  std::ostringstream os;
  os << "ObsSpace[" << source << "] n=" << obs.size()
     << " usable=" << usable_count() << " window=[" << window_start << ", "
     << (window_start + window_length) << "] s";
  if (!valid_time.empty()) os << " valid=" << valid_time;

  const auto bt = count_by_type();
  os << " types={";
  bool first = true;
  for (Size i = 0; i + 1 < bt.size(); ++i) {
    if (bt[i] == 0) continue;
    os << (first ? "" : ", ") << to_string(static_cast<ObsType>(i)) << ':' << bt[i];
    first = false;
  }
  os << "} vars={";
  const auto bv = count_by_variable();
  first = true;
  for (Size i = 0; i + 1 < bv.size(); ++i) {
    if (bv[i] == 0) continue;
    os << (first ? "" : ", ") << to_string(static_cast<VarKind>(i)) << ':' << bv[i];
    first = false;
  }
  os << '}';

  if (!obs.empty()) {
    Real tmin = obs.front().time, tmax = obs.front().time;
    Real smin = obs.front().sigma, smax = obs.front().sigma;
    Size bad = 0;
    for (const auto& o : obs) {
      tmin = std::min(tmin, o.time);
      tmax = std::max(tmax, o.time);
      smin = std::min(smin, o.sigma);
      smax = std::max(smax, o.sigma);
      if (!o.usable()) ++bad;
    }
    os << " t=[" << tmin << ", " << tmax << "] sigma=[" << smin << ", " << smax
       << "] rejected=" << bad;
  }
  return os.str();
}

// ===========================================================================
// CSV 读写
// ===========================================================================

namespace {

/// RFC4180 字段转义：仅在必要时加引号
std::string csv_escape(const std::string& s) {
  bool need = false;
  for (char c : s) {
    if (c == ',' || c == '"' || c == '\n' || c == '\r') { need = true; break; }
  }
  if (!need) return s;
  std::string out = "\"";
  for (char c : s) {
    if (c == '"') out += "\"\"";
    else out.push_back(c);
  }
  out.push_back('"');
  return out;
}

/// 解析一行 CSV（支持引号内的逗号与转义引号）
std::vector<std::string> csv_split(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  bool in_quotes = false;
  for (Size i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (in_quotes) {
      if (c == '"') {
        if (i + 1 < line.size() && line[i + 1] == '"') { cur.push_back('"'); ++i; }
        else in_quotes = false;
      } else {
        cur.push_back(c);
      }
    } else if (c == '"') {
      in_quotes = true;
    } else if (c == ',') {
      out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  out.push_back(cur);
  return out;
}

std::string trim(const std::string& s) {
  Size a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

Real parse_real(const std::string& s, const char* what) {
  const std::string t = trim(s);
  try {
    Size used = 0;
    const Real v = std::stod(t, &used);
    VIBE_CHECK_MSG(used == t.size(), std::string("非法实数(字段 ") + what + "): " + t);
    return v;
  } catch (const std::exception&) {
    throw ConfigError(std::string("CSV 实数解析失败(字段 ") + what + "): " + t);
  }
}

int parse_int(const std::string& s, const char* what) {
  const std::string t = trim(s);
  try {
    Size used = 0;
    const int v = std::stoi(t, &used);
    VIBE_CHECK_MSG(used == t.size(), std::string("非法整数(字段 ") + what + "): " + t);
    return v;
  } catch (const std::exception&) {
    throw ConfigError(std::string("CSV 整数解析失败(字段 ") + what + "): " + t);
  }
}

/// 网格单元中心 x（变分辨率时用逐列宽度前缀和）
Real cell_center_x(const grid::Grid& g, Int i) {
  const auto& geom = g.geom();
  if (!geom.variable_resolution || geom.dx_cell.empty()) {
    return geom.x0 + (static_cast<Real>(i) + Real(0.5)) * geom.dx;
  }
  Real x = geom.x0;
  for (Int m = 0; m < i; ++m) x += geom.dx_cell[static_cast<Size>(m)];
  return x + Real(0.5) * geom.dx_cell[static_cast<Size>(i)];
}

/// 网格单元中心 y
Real cell_center_y(const grid::Grid& g, Int j) {
  const auto& geom = g.geom();
  if (!geom.variable_resolution || geom.dy_cell.empty()) {
    return geom.y0 + (static_cast<Real>(j) + Real(0.5)) * geom.dy;
  }
  Real y = geom.y0;
  for (Int m = 0; m < j; ++m) y += geom.dy_cell[static_cast<Size>(m)];
  return y + Real(0.5) * geom.dy_cell[static_cast<Size>(j)];
}

constexpr char kBinaryMagic[8] = {'V', 'I', 'B', 'E', 'O', 'B', 'S', '1'};

/// 解析可选元数据注释行（以 '#' 开头）
void parse_metadata(const std::string& line, ObsSpace& os) {
  std::istringstream is(line.substr(1));
  std::string tok;
  while (is >> tok) {
    const auto eq = tok.find('=');
    if (eq == std::string::npos) continue;
    const std::string key = tok.substr(0, eq);
    const std::string val = tok.substr(eq + 1);
    if (key == "window_start") os.window_start = parse_real(val, "window_start");
    else if (key == "window_length") os.window_length = parse_real(val, "window_length");
    else if (key == "source") os.source = val;
    else if (key == "valid_time") os.valid_time = val;
  }
}

}  // namespace

void write_csv(const std::string& path, const ObsSpace& os) {
  std::ofstream f(path);
  if (!f) throw IoError("无法写入观测文件: " + path);
  f << "# window_start=" << os.window_start << " window_length=" << os.window_length
    << " source=" << csv_escape(os.source) << " valid_time=" << csv_escape(os.valid_time)
    << '\n';
  f << "type,time,x,y,z,var,channel,value,sigma,qc\n";
  f << std::setprecision(17);
  for (const auto& o : os.obs) {
    f << csv_escape(to_string(o.type)) << ',' << o.time << ',' << o.x << ',' << o.y
      << ',' << o.z << ',' << csv_escape(to_string(o.variable)) << ',' << o.channel
      << ',' << o.value << ',' << o.sigma << ',' << o.qc_flag << '\n';
  }
  if (!f) throw IoError("写观测文件失败: " + path);
}

ObsSpace ObsReader::read_csv(const std::string& path, Real window_length) {
  std::ifstream f(path);
  if (!f) throw IoError("无法读取观测文件: " + path);
  ObsSpace os;
  os.source = path;
  os.window_length = window_length;

  std::string line;
  bool header_seen = false;
  Size line_no = 0;
  while (std::getline(f, line)) {
    ++line_no;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    if (line[0] == '#') { parse_metadata(line, os); continue; }
    if (!header_seen) {
      const auto cols = csv_split(line);
      bool has_type = false, has_val = false;
      for (const auto& c : cols) {
        const std::string t = canonical(c);
        if (t == "type") has_type = true;
        if (t == "value") has_val = true;
      }
      VIBE_CHECK_MSG(has_type && has_val,
                     "观测 CSV 表头缺少 type/value 列: " + path + " 行 " +
                         std::to_string(line_no));
      header_seen = true;
      continue;
    }
    const auto cols = csv_split(line);
    if (cols.size() < 10) {
      throw ConfigError("观测 CSV 列数不足(应为 10): " + path + " 行 " +
                        std::to_string(line_no));
    }
    Observation o;
    o.type = obs_type_from_string(trim(cols[0]));
    o.time = parse_real(cols[1], "time");
    o.x = parse_real(cols[2], "x");
    o.y = parse_real(cols[3], "y");
    o.z = parse_real(cols[4], "z");
    o.variable = var_kind_from_string(trim(cols[5]));
    o.channel = parse_int(cols[6], "channel");
    o.value = parse_real(cols[7], "value");
    o.sigma = parse_real(cols[8], "sigma");
    o.qc_flag = parse_int(cols[9], "qc");
    os.obs.push_back(o);
  }
  VIBE_CHECK_MSG(header_seen, "观测 CSV 缺少表头: " + path);
  return os;
}

ObsSpace ObsReader::read_binary(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw IoError("无法读取二进制观测文件: " + path);
  char magic[8] = {};
  f.read(magic, 8);
  if (!f || std::memcmp(magic, kBinaryMagic, 8) != 0) {
    throw IoError("二进制观测文件魔数不匹配: " + path);
  }
  std::int64_t n = 0;
  Real window_start = 0, window_length = 0;
  f.read(reinterpret_cast<char*>(&n), sizeof(n));
  f.read(reinterpret_cast<char*>(&window_start), sizeof(Real));
  f.read(reinterpret_cast<char*>(&window_length), sizeof(Real));
  if (!f || n < 0) throw IoError("二进制观测文件头损坏: " + path);

  ObsSpace os;
  os.source = path;
  os.window_start = window_start;
  os.window_length = window_length;
  os.obs.resize(static_cast<Size>(n));
  for (auto& o : os.obs) {
    std::int32_t type = 0, variable = 0;
    f.read(reinterpret_cast<char*>(&type), sizeof(type));
    f.read(reinterpret_cast<char*>(&o.time), sizeof(Real));
    f.read(reinterpret_cast<char*>(&o.x), sizeof(Real));
    f.read(reinterpret_cast<char*>(&o.y), sizeof(Real));
    f.read(reinterpret_cast<char*>(&o.z), sizeof(Real));
    f.read(reinterpret_cast<char*>(&o.lon), sizeof(Real));
    f.read(reinterpret_cast<char*>(&o.lat), sizeof(Real));
    f.read(reinterpret_cast<char*>(&variable), sizeof(variable));
    f.read(reinterpret_cast<char*>(&o.channel), sizeof(int));
    f.read(reinterpret_cast<char*>(&o.value), sizeof(Real));
    f.read(reinterpret_cast<char*>(&o.sigma), sizeof(Real));
    f.read(reinterpret_cast<char*>(&o.bias), sizeof(Real));
    f.read(reinterpret_cast<char*>(&o.qc_flag), sizeof(int));
    f.read(reinterpret_cast<char*>(&o.station_id), sizeof(Index));
    f.read(reinterpret_cast<char*>(&o.record_id), sizeof(Index));
    f.read(reinterpret_cast<char*>(&o.level_type), sizeof(int));
    o.type = static_cast<ObsType>(type);
    o.variable = static_cast<VarKind>(variable);
  }
  if (!f) throw IoError("二进制观测文件读取不完整: " + path);
  return os;
}

ObsSpace ObsReader::read(const std::string& path, Real window_length) {
  const auto dot = path.find_last_of('.');
  const std::string ext =
      dot == std::string::npos ? std::string() : lower(path.substr(dot + 1));
  if (ext == "bin" || ext == "binary" || ext == "dat") return read_binary(path);
  return read_csv(path, window_length);
}

// ===========================================================================
// 合成观测
// ===========================================================================

namespace {

/// 该观测类型在采样时使用的变量循环表（[O8] 表 1 的常规观测量）
std::vector<VarKind> variables_for(ObsType t) {
  switch (t) {
    case ObsType::Radiosonde:
    case ObsType::Aircraft:
    case ObsType::Profiler:
      return {VarKind::U, VarKind::V, VarKind::T, VarKind::Q, VarKind::P};
    case ObsType::Surface:
      return {VarKind::PS, VarKind::T, VarKind::Q, VarKind::U, VarKind::V};
    case ObsType::AMV:
      return {VarKind::U, VarKind::V};
    case ObsType::Scatterometer:
      return {VarKind::U, VarKind::V};
    case ObsType::GnssRo:
      return {VarKind::Refractivity};
    case ObsType::Radiance:
      return {VarKind::Radiance};
    case ObsType::RadarReflectivity:
      return {VarKind::Reflectivity};
    default:
      return {VarKind::T};
  }
}

/// 该类型/变量的代表观测高度（米）：地面与卫星类固定在近地层；<0 表示按格点层中心
Real height_for(ObsType t, VarKind v) {
  if (t == ObsType::Surface) return (v == VarKind::U || v == VarKind::V) ? Real(10) : Real(2);
  if (t == ObsType::Scatterometer) return Real(10);
  return Real(-1);
}

/// 各变量的默认观测误差标准差（与 [O8] 的量级一致，教学用）
Real default_sigma(VarKind v) {
  switch (v) {
    case VarKind::U: case VarKind::V: return Real(1.5);   // m/s
    case VarKind::W:                  return Real(1.0);
    case VarKind::T:                  return Real(1.0);   // K
    case VarKind::Q:                  return Real(1e-3);  // kg/kg
    case VarKind::PS:                 return Real(150);   // Pa
    case VarKind::P:                  return Real(150);
    case VarKind::Radiance:           return Real(1.0);   // K
    case VarKind::Refractivity:       return Real(1.0);   // N 单位
    case VarKind::Reflectivity:       return Real(2.0);   // dBZ
    default:                          return Real(1.0);
  }
}

}  // namespace

ObsSpace generate_synthetic_observations(const ObservationOperator& op,
                                         const ModelStateView& view,
                                         const SyntheticObsOptions& opt) {
  VIBE_CHECK_MSG(view.valid(), "generate_synthetic_observations 需要有效的 ModelStateView");
  const grid::Grid& g = *view.grid;
  const Int nx = g.nx(), ny = g.ny(), nz = g.nz();

  // 生成类型集合：算子只处理单一类型，复合算子则由 opt.types 指定
  std::vector<ObsType> types;
  if (op.type() != ObsType::Count) {
    types.push_back(op.type());
  } else {
    types = opt.types;
  }
  if (types.empty()) types.push_back(ObsType::Radiosonde);

  ObsSpace out;
  out.window_start = Real(0);
  out.window_length = Real(0);
  out.source = "synthetic";
  out.valid_time = "t=" + std::to_string(opt.time) + "s";

  const Size n_points =
      static_cast<Size>(nx) * static_cast<Size>(ny) * static_cast<Size>(nz);
  const Real frac = clamp(opt.density_fraction, Real(0), Real(1));
  const Size n_per_type =
      std::max<Size>(1, static_cast<Size>(std::llround(frac * static_cast<Real>(n_points))));

  std::mt19937 rng(opt.seed);
  std::uniform_int_distribution<Int> di(0, nx > 1 ? nx - 1 : 0);
  std::uniform_int_distribution<Int> dj(0, ny > 1 ? ny - 1 : 0);
  std::uniform_int_distribution<Int> dk(0, nz > 1 ? nz - 1 : 0);
  std::normal_distribution<Real> noise(Real(0), Real(1));

  for (ObsType t : types) {
    const auto vars = variables_for(t);
    for (Size m = 0; m < n_per_type; ++m) {
      Observation o;
      o.type = t;
      o.time = opt.time;
      o.variable = vars[m % vars.size()];
      const Int i = di(rng), j = dj(rng), k = dk(rng);
      o.x = cell_center_x(g, i);
      o.y = cell_center_y(g, j);
      const Real zh = height_for(t, o.variable);
      o.z = (zh >= Real(0)) ? zh : g.z_center(i, j, k);
      if (o.variable == VarKind::PS) o.z = g.z_center(i, j, 0);
      o.channel = (o.variable == VarKind::Radiance) ? (static_cast<int>(m) % 15) : -1;
      o.sigma = default_sigma(o.variable);
      o.qc_flag = 0;
      o.station_id = static_cast<Index>(m);
      o.record_id = static_cast<Index>(m);

      // 用观测算子采样模式值（逐条构造单点 ObsSpace，保证与算子语义一致）
      ObsSpace one;
      one.obs.push_back(o);
      std::vector<Real> y;
      op.apply(view, one, y);
      VIBE_CHECK_MSG(y.size() == 1, "观测算子返回长度与观测数不一致");
      o.value = y[0];
      if (opt.add_noise && o.sigma > Real(0)) o.value += o.sigma * noise(rng);
      out.obs.push_back(o);
    }
  }
  return out;
}

}  // namespace vibe::obs
