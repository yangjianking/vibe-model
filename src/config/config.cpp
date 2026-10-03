/// @file config.cpp
/// @brief 运行配置的装配、叠加与一致性校验（见 include/vibe/config/config.hpp）。
///
/// 设计要点
/// --------
///   1. 每个字段都从 YAML 读取并带结构体默认值；只有"C 语言层面的必填项"
///      （domain 段出现时的 nx/ny/nz/dx/dy，time 段出现时的 dt）缺失才抛
///      ConfigError，错误消息带字段路径（如 domain.nx）。
///   2. 多文件叠加：以空节点为底，用 config::merge 依次覆盖，再统一 validate。
///   3. 校验中的"表面警告"（进程网格不整除网格点数）只记录日志不抛错，
///      因为 grid::Decomposition::make 支持余数分配（各进程点数相差 1）。
///   4. 配置指纹用 FNV-1a 64 位哈希，供 RestartInfo::config_hash 使用。
///      注意：冻结的头文件未声明该函数，故放在 detail 命名空间内并由本 TU
///      声明；外部若要使用需自行前向声明（见实现总结）。
///
/// 网格分解的整除性说明见 [B12] Gropp et al. (1999) 第 4 章；
/// CFL 与声波子步的时间步限制见 [T5] Wicker & Skamarock (2002) 与
/// [D6] Klemp et al. (2008)。

#include "vibe/config/config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/common/logging.hpp"
#include "vibe/common/types.hpp"

namespace vibe::config {
namespace {

// ===========================================================================
// 小工具
// ===========================================================================

std::string trim_copy(const std::string& s) {
  Size b = 0;
  Size e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b])) != 0) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0) --e;
  return s.substr(b, e - b);
}

std::string lower_copy(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

std::string type_name(const YamlNode& n) {
  switch (n.type()) {
    case YamlNode::Type::Null: return "空值";
    case YamlNode::Type::Scalar: return "标量";
    case YamlNode::Type::Sequence: return "序列";
    case YamlNode::Type::Mapping: return "映射";
  }
  return "未知";
}

std::string join_allowed(std::initializer_list<std::string_view> allowed) {
  std::string out;
  for (const std::string_view v : allowed) {
    if (!out.empty()) out += ", ";
    out += std::string(v);
  }
  return out;
}

void require_in_list(const std::string& value, const std::string& path,
                     std::initializer_list<std::string_view> allowed) {
  for (const std::string_view v : allowed) {
    if (value == v) return;
  }
  throw ConfigError("字段 " + path + " = '" + value + "' 不在白名单内；允许值: " +
                    join_allowed(allowed));
}

/// 在映射中按别名列表查找子节点；不存在返回 nullptr。
const YamlNode* find_key(const YamlNode& section, std::initializer_list<const char*> keys) {
  if (!section.is_mapping()) return nullptr;
  for (const char* k : keys) {
    if (section.has(k)) return &section[k];
  }
  return nullptr;
}

/// 解析一个命名子段（必须为映射）；键存在但值为 null 时返回空节点。
const YamlNode& resolve_section(const YamlNode& root, std::initializer_list<const char*> keys,
                                const char* path) {
  static const YamlNode kEmpty;
  for (const char* k : keys) {
    if (!root.has(k)) continue;
    const YamlNode& v = root[k];
    if (v.is_null()) return kEmpty;
    if (!v.is_mapping()) {
      throw ConfigError(std::string("字段 ") + path + " 需要映射，得到 " + type_name(v));
    }
    return v;
  }
  return kEmpty;
}

bool has_any_section(const YamlNode& root, std::initializer_list<const char*> keys) {
  if (!root.is_mapping()) return false;
  for (const char* k : keys) {
    if (root.has(k) && !root[k].is_null()) return true;
  }
  return false;
}

// ===========================================================================
// 字段读取（默认值 + 类型/格式检查 + 错误路径）
// ===========================================================================

Real real_field(const YamlNode* v, const std::string& path, Real def) {
  if (v == nullptr || v->is_null()) return def;
  if (!v->is_scalar()) {
    throw ConfigError("字段 " + path + " 需要标量，得到 " + type_name(*v));
  }
  const Real probe = v->as_real(std::numeric_limits<Real>::quiet_NaN());
  if (std::isnan(static_cast<double>(probe))) {
    throw ConfigError("字段 " + path + " 不是合法数值: '" + v->raw() + "'");
  }
  return probe;
}

Int int_field(const YamlNode* v, const std::string& path, Int def) {
  if (v == nullptr || v->is_null()) return def;
  if (!v->is_scalar()) {
    throw ConfigError("字段 " + path + " 需要标量，得到 " + type_name(*v));
  }
  const Real probe = v->as_real(std::numeric_limits<Real>::quiet_NaN());
  if (std::isnan(static_cast<double>(probe))) {
    throw ConfigError("字段 " + path + " 不是合法整数: '" + v->raw() + "'");
  }
  const double r = std::llround(static_cast<double>(probe));
  if (r > static_cast<double>(std::numeric_limits<Int>::max()) ||
      r < static_cast<double>(std::numeric_limits<Int>::min())) {
    throw ConfigError("字段 " + path + " 超出 Int 范围: '" + v->raw() + "'");
  }
  return static_cast<Int>(r);
}

bool bool_field(const YamlNode* v, const std::string& path, bool def) {
  if (v == nullptr || v->is_null()) return def;
  if (!v->is_scalar()) {
    throw ConfigError("字段 " + path + " 需要标量，得到 " + type_name(*v));
  }
  const std::string lw = lower_copy(trim_copy(v->raw()));
  if (lw == "true" || lw == "yes" || lw == "on" || lw == "1") return true;
  if (lw == "false" || lw == "no" || lw == "off" || lw == "0") return false;
  const Real probe = v->as_real(std::numeric_limits<Real>::quiet_NaN());
  if (!std::isnan(static_cast<double>(probe))) return probe != Real(0);
  throw ConfigError("字段 " + path + " 不是合法布尔值: '" + v->raw() + "'");
}

std::string str_field(const YamlNode* v, const std::string& path, const std::string& def) {
  if (v == nullptr || v->is_null()) return def;
  if (!v->is_scalar()) {
    throw ConfigError("字段 " + path + " 需要字符串，得到 " + type_name(*v));
  }
  return v->raw();
}

std::vector<Real> real_list_field(const YamlNode* v, const std::string& path,
                                  const std::vector<Real>& def) {
  if (v == nullptr || v->is_null()) return def;
  std::vector<Real> out;
  if (v->is_scalar()) {
    const Real probe = v->as_real(std::numeric_limits<Real>::quiet_NaN());
    if (std::isnan(static_cast<double>(probe))) {
      throw ConfigError("字段 " + path + " 不是合法数值: '" + v->raw() + "'");
    }
    out.push_back(probe);
    return out;
  }
  if (v->is_mapping()) {
    for (const std::string& k : v->keys()) {
      const YamlNode& e = (*v)[k];
      const Real probe = e.as_real(std::numeric_limits<Real>::quiet_NaN());
      if (std::isnan(static_cast<double>(probe))) {
        throw ConfigError("字段 " + path + "." + k + " 不是合法数值");
      }
      out.push_back(probe);
    }
    return out;
  }
  if (!v->is_sequence()) {
    throw ConfigError("字段 " + path + " 需要序列或标量，得到 " + type_name(*v));
  }
  for (Size i = 0; i < v->sequence().size(); ++i) {
    const YamlNode& e = (*v)[i];
    if (!e.is_scalar()) {
      throw ConfigError("字段 " + path + "[" + std::to_string(i) + "] 不是标量");
    }
    const Real probe = e.as_real(std::numeric_limits<Real>::quiet_NaN());
    if (std::isnan(static_cast<double>(probe))) {
      throw ConfigError("字段 " + path + "[" + std::to_string(i) + "] 不是合法数值: '" +
                        e.raw() + "'");
    }
    out.push_back(probe);
  }
  return out;
}

std::vector<int> int_list_field(const YamlNode* v, const std::string& path,
                                const std::vector<int>& def) {
  const std::vector<Real> raw = real_list_field(v, path, {});
  if (raw.empty() && (v == nullptr || v->is_null())) return def;
  std::vector<int> out;
  out.reserve(raw.size());
  for (const Real r : raw) out.push_back(static_cast<int>(std::llround(static_cast<double>(r))));
  return out;
}

std::vector<std::string> str_list_field(const YamlNode* v, const std::string& path,
                                        const std::vector<std::string>& def) {
  if (v == nullptr || v->is_null()) return def;
  std::vector<std::string> out;
  if (v->is_scalar()) {
    out.push_back(v->raw());
    return out;
  }
  if (!v->is_sequence()) {
    throw ConfigError("字段 " + path + " 需要字符串序列，得到 " + type_name(*v));
  }
  for (Size i = 0; i < v->sequence().size(); ++i) {
    const YamlNode& e = (*v)[i];
    if (!e.is_scalar()) {
      throw ConfigError("字段 " + path + "[" + std::to_string(i) + "] 不是字符串");
    }
    out.push_back(e.raw());
  }
  return out;
}

// ===========================================================================
// 白名单（校验用常量；同时记录在 docs/design/11_config_and_io.md）
// ===========================================================================

constexpr std::string_view kIntegrators[] = {"rk3_acoustic", "semi_implicit", "rk3", "hevi"};
constexpr std::string_view kAdvectionSchemes[] = {"central2", "central4", "central6",
                                                  "weno5", "upwind3"};
constexpr std::string_view kHelmholtzSolvers[] = {"vertical_tridiagonal", "krylov_multigrid",
                                                  "krylov_jacobi"};
constexpr std::string_view kCoriolisSchemes[] = {"none", "fplane", "betaplane"};
constexpr std::string_view kRefinementSchemes[] = {"none", "circular", "channel"};
constexpr std::string_view kMicrophysicsSchemes[] = {"none", "kessler", "lin", "thompson",
                                                     "morrison"};
constexpr std::string_view kRadiationSchemes[] = {"none", "rrtm", "rrtmg", "rrtmg_simple"};
constexpr std::string_view kPblSchemes[] = {"none", "ysu", "myj", "mye", "smagorinsky"};
constexpr std::string_view kSurfaceSchemes[] = {"none", "monin_obukhov", "noah",
                                                "noilhan_planton"};
constexpr std::string_view kCumulusSchemes[] = {"none", "kain_fritsch", "grell", "tiedke",
                                                "tiedtke"};
constexpr std::string_view kIoBackends[] = {"netcdf", "binary", "both"};
constexpr std::string_view kBackgroundMethods[] = {"nmc", "ensemble", "hybrid"};
constexpr std::string_view kBalanceMethods[] = {"none", "linear_balance", "omega"};
constexpr std::string_view kMinimizers[] = {"lbfgs", "cg", "lanczos"};

std::string advection_name(Int order) {
  switch (order) {
    case 2: return "central2";
    case 4: return "central4";
    case 5: return "weno5";
    case 6: return "central6";
    default: return "central2";
  }
}

std::string fmt(Real v) {
  std::ostringstream os;
  os << std::setprecision(10) << static_cast<double>(v);
  return os.str();
}

}  // namespace

// ===========================================================================
// ModelConfig
// ===========================================================================

ModelConfig ModelConfig::from_yaml(const YamlNode& root_in) {
  ModelConfig c;
  if (root_in.is_null()) return c;
  if (!root_in.is_mapping()) {
    throw ConfigError("配置根节点必须是映射，得到 " + type_name(root_in));
  }
  const YamlNode* model = nullptr;
  for (const char* k : {"model", "vibe_model"}) {
    if (root_in.has(k) && root_in[k].is_mapping()) { model = &root_in[k]; break; }
  }
  const YamlNode& root = model != nullptr ? *model : root_in;

  c.name = str_field(find_key(root, {"name", "run_name"}), "name", c.name);
  c.description = str_field(find_key(root, {"description"}), "description", c.description);

  // ------------------------------ domain --------------------------------
  {
    const YamlNode& d = resolve_section(root, {"domain", "grid"}, "domain");
    c.domain.nx = int_field(find_key(d, {"nx"}), "domain.nx", c.domain.nx);
    c.domain.ny = int_field(find_key(d, {"ny"}), "domain.ny", c.domain.ny);
    c.domain.nz = int_field(find_key(d, {"nz"}), "domain.nz", c.domain.nz);
    c.domain.dx = real_field(find_key(d, {"dx"}), "domain.dx", c.domain.dx);
    c.domain.dy = real_field(find_key(d, {"dy"}), "domain.dy", c.domain.dx);
    c.domain.x0 = real_field(find_key(d, {"x0", "x_min"}), "domain.x0", c.domain.x0);
    c.domain.y0 = real_field(find_key(d, {"y0", "y_min"}), "domain.y0", c.domain.y0);
    c.domain.z_top = real_field(find_key(d, {"z_top", "ztop", "top"}), "domain.z_top",
                                c.domain.z_top);
    c.domain.terrain = str_field(find_key(d, {"terrain", "terrain_file"}), "domain.terrain",
                                 c.domain.terrain);
    const std::vector<Real> zeta =
        real_list_field(find_key(d, {"zeta"}), "domain.zeta", c.domain.zeta);
    c.domain.zeta = zeta;
    c.domain.zeta_first_thickness =
        real_field(find_key(d, {"zeta_first_thickness", "first_thickness"}),
                   "domain.zeta_first_thickness", c.domain.zeta_first_thickness);
    c.domain.zeta_stretch =
        real_field(find_key(d, {"zeta_stretch", "vertical_stretch", "stretch"}),
                   "domain.zeta_stretch", c.domain.zeta_stretch);
    c.domain.variable_resolution =
        bool_field(find_key(d, {"variable_resolution", "var_res"}),
                   "domain.variable_resolution", c.domain.variable_resolution);
    c.domain.h_min = real_field(find_key(d, {"h_min"}), "domain.h_min", c.domain.h_min);
    c.domain.h_max = real_field(find_key(d, {"h_max"}), "domain.h_max", c.domain.h_max);
    c.domain.refinement =
        str_field(find_key(d, {"refinement"}), "domain.refinement", c.domain.refinement);
    c.domain.refinement_cx =
        real_field(find_key(d, {"refinement_cx", "refinement_x"}), "domain.refinement_cx",
                   c.domain.refinement_cx);
    c.domain.refinement_cy =
        real_field(find_key(d, {"refinement_cy", "refinement_y"}), "domain.refinement_cy",
                   c.domain.refinement_cy);
    c.domain.refinement_radius =
        real_field(find_key(d, {"refinement_radius", "refinement_r"}),
                   "domain.refinement_radius", c.domain.refinement_radius);
    c.domain.refinement_transition =
        real_field(find_key(d, {"refinement_transition"}), "domain.refinement_transition",
                   c.domain.refinement_transition);
    c.domain.periodic_x = bool_field(find_key(d, {"periodic_x"}), "domain.periodic_x",
                                     c.domain.periodic_x);
    c.domain.periodic_y = bool_field(find_key(d, {"periodic_y"}), "domain.periodic_y",
                                     c.domain.periodic_y);

    if (has_any_section(root, {"domain", "grid"})) {
      for (const char* k : {"nx", "ny", "nz", "dx", "dy"}) {
        if (!d.has(k) || d[k].is_null()) {
          throw ConfigError(std::string("缺少必填字段: domain.") + k);
        }
      }
    }
  }

  // ------------------------------- time ---------------------------------
  {
    const YamlNode& t = resolve_section(root, {"time", "timestep"}, "time");
    c.time.dt = real_field(find_key(t, {"dt", "time_step"}), "time.dt", c.time.dt);
    c.time.run_length =
        real_field(find_key(t, {"run_length", "length"}), "time.run_length", c.time.run_length);
    c.time.acoustic_substeps =
        int_field(find_key(t, {"acoustic_substeps", "n_sub"}), "time.acoustic_substeps",
                  c.time.acoustic_substeps);
    c.time.integrator =
        str_field(find_key(t, {"integrator", "scheme"}), "time.integrator", c.time.integrator);
    c.time.cfl_target =
        real_field(find_key(t, {"cfl_target", "cfl"}), "time.cfl_target", c.time.cfl_target);
    c.time.adaptive_dt = bool_field(find_key(t, {"adaptive_dt"}), "time.adaptive_dt",
                                    c.time.adaptive_dt);
    c.time.output_interval =
        real_field(find_key(t, {"output_interval", "output_dt"}), "time.output_interval",
                   c.time.output_interval);
    c.time.restart_interval =
        real_field(find_key(t, {"restart_interval", "restart_dt"}), "time.restart_interval",
                   c.time.restart_interval);
    c.time.start_time = str_field(find_key(t, {"start_time", "start"}), "time.start_time",
                                  c.time.start_time);
    if (has_any_section(root, {"time", "timestep"})) {
      const YamlNode* dt = find_key(t, {"dt", "time_step"});
      if (dt == nullptr || dt->is_null()) throw ConfigError("缺少必填字段: time.dt");
    }
  }

  // ------------------------------ numerics ------------------------------
  {
    const YamlNode& n = resolve_section(root, {"numerics", "dynamics"}, "numerics");
    c.numerics.advection =
        str_field(find_key(n, {"advection"}), "numerics.advection", c.numerics.advection);
    if (find_key(n, {"advection"}) == nullptr) {
      const YamlNode* order = find_key(n, {"advection_order", "order"});
      if (order != nullptr && order->is_scalar()) {
        c.numerics.advection = advection_name(int_field(order, "numerics.advection_order", 2));
      }
    }
    c.numerics.flux_form =
        bool_field(find_key(n, {"flux_form"}), "numerics.flux_form", c.numerics.flux_form);
    c.numerics.divergence_damping =
        real_field(find_key(n, {"divergence_damping", "damping"}), "numerics.divergence_damping",
                   c.numerics.divergence_damping);
    c.numerics.divergence_order =
        static_cast<int>(int_field(find_key(n, {"divergence_order"}), "numerics.divergence_order",
                                   c.numerics.divergence_order));
    c.numerics.sponge_alpha =
        real_field(find_key(n, {"sponge_alpha"}), "numerics.sponge_alpha", c.numerics.sponge_alpha);
    c.numerics.sponge_start_fraction =
        real_field(find_key(n, {"sponge_start_fraction", "sponge_depth"}),
                   "numerics.sponge_start_fraction", c.numerics.sponge_start_fraction);
    c.numerics.helmholtz_solver =
        str_field(find_key(n, {"helmholtz_solver", "solver"}), "numerics.helmholtz_solver",
                  c.numerics.helmholtz_solver);
    c.numerics.helmholtz_max_iter =
        static_cast<int>(int_field(find_key(n, {"helmholtz_max_iter", "krylov_max_iter"}),
                                   "numerics.helmholtz_max_iter", c.numerics.helmholtz_max_iter));
    c.numerics.helmholtz_tol =
        real_field(find_key(n, {"helmholtz_tol", "krylov_tolerance"}), "numerics.helmholtz_tol",
                   c.numerics.helmholtz_tol);
    c.numerics.weno = bool_field(find_key(n, {"weno"}), "numerics.weno", c.numerics.weno);
    c.numerics.coriolis =
        str_field(find_key(n, {"coriolis"}), "numerics.coriolis", c.numerics.coriolis);
    c.numerics.fplane_latitude =
        real_field(find_key(n, {"fplane_latitude", "latitude"}), "numerics.fplane_latitude",
                   c.numerics.fplane_latitude);
  }

  // ------------------------------ nesting -------------------------------
  {
    const YamlNode& nest = resolve_section(root, {"nesting", "nest"}, "nesting");
    c.nesting.enabled =
        bool_field(find_key(nest, {"enabled"}), "nesting.enabled", c.nesting.enabled);
    c.nesting.levels =
        int_field(find_key(nest, {"levels", "n_levels"}), "nesting.levels", c.nesting.levels);
    c.nesting.ratio =
        int_field(find_key(nest, {"ratio"}), "nesting.ratio", c.nesting.ratio);
    c.nesting.two_way =
        bool_field(find_key(nest, {"two_way"}), "nesting.two_way", c.nesting.two_way);
    c.nesting.boundary_zone =
        int_field(find_key(nest, {"boundary_zone", "relaxation_zone", "n_zone"}),
                  "nesting.boundary_zone", c.nesting.boundary_zone);
    c.nesting.feedback_interval =
        real_field(find_key(nest, {"feedback_interval"}), "nesting.feedback_interval",
                   c.nesting.feedback_interval);
    c.nesting.child_domains =
        str_list_field(find_key(nest, {"child_domains", "children"}), "nesting.child_domains",
                       c.nesting.child_domains);
  }

  // ------------------------------ physics -------------------------------
  {
    const YamlNode& p = resolve_section(root, {"physics"}, "physics");
    c.physics.microphysics =
        str_field(find_key(p, {"microphysics", "mp_physics"}), "physics.microphysics",
                  c.physics.microphysics);
    c.physics.radiation =
        str_field(find_key(p, {"radiation"}), "physics.radiation", c.physics.radiation);
    c.physics.pbl = str_field(find_key(p, {"pbl"}), "physics.pbl", c.physics.pbl);
    c.physics.surface =
        str_field(find_key(p, {"surface", "surface_layer"}), "physics.surface", c.physics.surface);
    c.physics.cumulus =
        str_field(find_key(p, {"cumulus", "convection"}), "physics.cumulus", c.physics.cumulus);
    c.physics.radiation_cadence =
        real_field(find_key(p, {"radiation_cadence", "radiation_interval"}),
                   "physics.radiation_cadence", c.physics.radiation_cadence);
    c.physics.use_cloud_fraction =
        bool_field(find_key(p, {"use_cloud_fraction"}), "physics.use_cloud_fraction",
                   c.physics.use_cloud_fraction);
    c.physics.co2_ppm =
        real_field(find_key(p, {"co2_ppm"}), "physics.co2_ppm", c.physics.co2_ppm);
    c.physics.aerosol_optical_depth =
        real_field(find_key(p, {"aerosol_optical_depth", "aod"}), "physics.aerosol_optical_depth",
                   c.physics.aerosol_optical_depth);
    c.physics.soil_layers =
        int_field(find_key(p, {"soil_layers"}), "physics.soil_layers", c.physics.soil_layers);
    c.physics.enable_tendency_physics =
        bool_field(find_key(p, {"enable_tendency_physics"}), "physics.enable_tendency_physics",
                   c.physics.enable_tendency_physics);
  }

  // ------------------------------ parallel ------------------------------
  {
    const YamlNode& par =
        resolve_section(root, {"parallel", "mpi", "decomposition"}, "parallel");
    c.parallel.px = int_field(find_key(par, {"px", "npx"}), "parallel.px", c.parallel.px);
    c.parallel.py = int_field(find_key(par, {"py", "npy"}), "parallel.py", c.parallel.py);
    c.parallel.io_backend =
        str_field(find_key(par, {"io_backend", "io"}), "parallel.io_backend",
                  c.parallel.io_backend);
    c.parallel.output_dir =
        str_field(find_key(par, {"output_dir", "out_dir"}), "parallel.output_dir",
                  c.parallel.output_dir);
    c.parallel.write_double =
        bool_field(find_key(par, {"write_double"}), "parallel.write_double",
                   c.parallel.write_double);
  }

  return c;
}

ModelConfig ModelConfig::load(const std::string& path) {
  ModelConfig c = from_yaml(YamlNode::parse_file(path));
  c.validate();
  return c;
}

ModelConfig ModelConfig::load(const std::vector<std::string>& paths, bool allow_missing) {
  YamlNode merged;  // Null：空底
  for (const std::string& p : paths) {
    if (p.empty()) continue;
    std::error_code ec;
    const bool exists = std::filesystem::exists(p, ec);
    if (ec || !exists) {
      if (allow_missing) {
        VIBE_WARN("[config] 配置层不存在，已跳过: ", p);
        continue;
      }
      throw ConfigError("配置文件不存在: " + p);
    }
    merged = merge(merged, YamlNode::parse_file(p));
  }
  ModelConfig c = from_yaml(merged);
  c.validate();
  return c;
}

// ---------------------------------------------------------------------------
// 一致性校验
//
// 所有判据都来自数值稳定性的必要条件：
//   * 水平 CFL： Δt <= CFL * min(Δx,Δy) / (U_max + c_s)             [T5][D6]
//   * 垂直声波： (c_s Δt / n_sub) / Δz <= CFL                        [D6]
//   * 层坐标必须单调且落在 [0,1]（Gal-Chen & Somerville 地形追随坐标）[D2]
// ---------------------------------------------------------------------------
void ModelConfig::validate() const {
  // ------------------------------ 域几何 --------------------------------
  if (domain.nx <= 0) throw ConfigError("domain.nx 必须 > 0，实际 " + std::to_string(domain.nx));
  if (domain.ny <= 0) throw ConfigError("domain.ny 必须 > 0，实际 " + std::to_string(domain.ny));
  if (domain.nz <= 0) throw ConfigError("domain.nz 必须 > 0，实际 " + std::to_string(domain.nz));
  if (!(domain.dx > Real(0))) throw ConfigError("domain.dx 必须 > 0，实际 " + fmt(domain.dx));
  if (!(domain.dy > Real(0))) throw ConfigError("domain.dy 必须 > 0，实际 " + fmt(domain.dy));
  if (!(domain.z_top > Real(0))) {
    throw ConfigError("domain.z_top 必须 > 0，实际 " + fmt(domain.z_top));
  }
  if (!domain.zeta.empty()) {
    if (static_cast<Int>(domain.zeta.size()) != domain.nz + 1) {
      throw ConfigError("domain.zeta 长度必须为 nz+1=" + std::to_string(domain.nz + 1) +
                        "，实际 " + std::to_string(domain.zeta.size()));
    }
    for (Size i = 0; i < domain.zeta.size(); ++i) {
      const Real z = domain.zeta[i];
      if (z < Real(0) || z > Real(1)) {
        throw ConfigError("domain.zeta[" + std::to_string(i) + "]=" + fmt(z) +
                          " 超出 [0,1]");
      }
      if (i > 0 && !(z > domain.zeta[i - 1])) {
        throw ConfigError("domain.zeta 必须严格单调递增：zeta[" + std::to_string(i - 1) +
                          "]=" + fmt(domain.zeta[i - 1]) + " >= zeta[" + std::to_string(i) +
                          "]=" + fmt(z));
      }
    }
  }
  if (!(domain.zeta_first_thickness > Real(0))) {
    throw ConfigError("domain.zeta_first_thickness 必须 > 0，实际 " +
                      fmt(domain.zeta_first_thickness));
  }
  if (!(domain.zeta_stretch >= Real(1))) {
    throw ConfigError("domain.zeta_stretch 必须 >= 1，实际 " + fmt(domain.zeta_stretch));
  }
  if (domain.variable_resolution) {
    if (!(domain.h_min > Real(0))) {
      throw ConfigError("domain.h_min 必须 > 0（变分辨率网格），实际 " + fmt(domain.h_min));
    }
    if (!(domain.h_max > domain.h_min)) {
      throw ConfigError("domain.h_max 必须 > domain.h_min（变分辨率网格），实际 h_min=" +
                        fmt(domain.h_min) + " h_max=" + fmt(domain.h_max));
    }
    require_in_list(domain.refinement, "domain.refinement",
                    {kRefinementSchemes[0], kRefinementSchemes[1], kRefinementSchemes[2]});
    if (domain.refinement != "none" && !(domain.refinement_radius > Real(0))) {
      throw ConfigError("domain.refinement_radius 必须 > 0（refinement=" + domain.refinement +
                        "）");
    }
  }

  // ------------------------------- 时间 ---------------------------------
  if (!(time.dt > Real(0))) throw ConfigError("time.dt 必须 > 0，实际 " + fmt(time.dt));
  if (time.run_length < Real(0)) {
    throw ConfigError("time.run_length 不能为负，实际 " + fmt(time.run_length));
  }
  if (time.acoustic_substeps < 1) {
    throw ConfigError("time.acoustic_substeps 必须 >= 1，实际 " +
                      std::to_string(time.acoustic_substeps));
  }
  require_in_list(time.integrator, "time.integrator",
                  {kIntegrators[0], kIntegrators[1], kIntegrators[2], kIntegrators[3]});
  if (!(time.cfl_target > Real(0) && time.cfl_target <= Real(1.5))) {
    throw ConfigError("time.cfl_target 必须落在 (0, 1.5]，实际 " + fmt(time.cfl_target));
  }
  if (time.output_interval < Real(0)) {
    throw ConfigError("time.output_interval 不能为负，实际 " + fmt(time.output_interval));
  }
  if (time.restart_interval < Real(0)) {
    throw ConfigError("time.restart_interval 不能为负，实际 " + fmt(time.restart_interval));
  }

  // ------------------------------ 数值格式 ------------------------------
  require_in_list(numerics.advection, "numerics.advection",
                  {kAdvectionSchemes[0], kAdvectionSchemes[1], kAdvectionSchemes[2],
                   kAdvectionSchemes[3], kAdvectionSchemes[4]});
  if (numerics.divergence_order != 2 && numerics.divergence_order != 4 &&
      numerics.divergence_order != 6) {
    throw ConfigError("numerics.divergence_order 必须是 2/4/6，实际 " +
                      std::to_string(numerics.divergence_order));
  }
  require_in_list(numerics.helmholtz_solver, "numerics.helmholtz_solver",
                  {kHelmholtzSolvers[0], kHelmholtzSolvers[1], kHelmholtzSolvers[2]});
  if (numerics.helmholtz_max_iter < 1) {
    throw ConfigError("numerics.helmholtz_max_iter 必须 >= 1，实际 " +
                      std::to_string(numerics.helmholtz_max_iter));
  }
  if (!(numerics.helmholtz_tol > Real(0))) {
    throw ConfigError("numerics.helmholtz_tol 必须 > 0，实际 " + fmt(numerics.helmholtz_tol));
  }
  require_in_list(numerics.coriolis, "numerics.coriolis",
                  {kCoriolisSchemes[0], kCoriolisSchemes[1], kCoriolisSchemes[2]});
  if (numerics.sponge_start_fraction < Real(0) || numerics.sponge_start_fraction > Real(1)) {
    throw ConfigError("numerics.sponge_start_fraction 必须落在 [0,1]，实际 " +
                      fmt(numerics.sponge_start_fraction));
  }
  if (numerics.sponge_alpha < Real(0)) {
    throw ConfigError("numerics.sponge_alpha 不能为负，实际 " + fmt(numerics.sponge_alpha));
  }

  // ------------------------------- 嵌套 ---------------------------------
  if (nesting.ratio < 1) {
    throw ConfigError("nesting.ratio 必须 >= 1，实际 " + std::to_string(nesting.ratio));
  }
  if (nesting.levels < 0) {
    throw ConfigError("nesting.levels 必须 >= 0，实际 " + std::to_string(nesting.levels));
  }
  if (nesting.enabled) {
    if (nesting.levels < 1) {
      throw ConfigError("nesting.enabled=true 时 nesting.levels 必须 >= 1");
    }
    if (nesting.ratio < 2) {
      throw ConfigError("嵌套细化比 nesting.ratio 必须 >= 2，实际 " +
                        std::to_string(nesting.ratio));
    }
    if (nesting.boundary_zone < 1) {
      throw ConfigError("nesting.boundary_zone 必须 >= 1（Davies 松弛区），实际 " +
                        std::to_string(nesting.boundary_zone));
    }
    if (nesting.feedback_interval < Real(0)) {
      throw ConfigError("nesting.feedback_interval 不能为负，实际 " +
                        fmt(nesting.feedback_interval));
    }
  } else if (nesting.ratio >= 2 && nesting.levels >= 1) {
    throw ConfigError("nesting.levels=" + std::to_string(nesting.levels) +
                      " 且 nesting.ratio=" + std::to_string(nesting.ratio) +
                      " 时 nesting.enabled 必须为 true");
  }
  if (!nesting.enabled && !nesting.child_domains.empty()) {
    VIBE_WARN("[config] nesting.enabled=false 但给出了 nesting.child_domains（",
              nesting.child_domains.size(), " 项），这些子域将被忽略");
  }

  // ------------------------------- 物理 ---------------------------------
  require_in_list(physics.microphysics, "physics.microphysics",
                  {kMicrophysicsSchemes[0], kMicrophysicsSchemes[1], kMicrophysicsSchemes[2],
                   kMicrophysicsSchemes[3], kMicrophysicsSchemes[4]});
  require_in_list(physics.radiation, "physics.radiation",
                  {kRadiationSchemes[0], kRadiationSchemes[1], kRadiationSchemes[2],
                   kRadiationSchemes[3]});
  require_in_list(physics.pbl, "physics.pbl",
                  {kPblSchemes[0], kPblSchemes[1], kPblSchemes[2], kPblSchemes[3],
                   kPblSchemes[4]});
  require_in_list(physics.surface, "physics.surface",
                  {kSurfaceSchemes[0], kSurfaceSchemes[1], kSurfaceSchemes[2],
                   kSurfaceSchemes[3]});
  require_in_list(physics.cumulus, "physics.cumulus",
                  {kCumulusSchemes[0], kCumulusSchemes[1], kCumulusSchemes[2],
                   kCumulusSchemes[3], kCumulusSchemes[4]});
  if (!(physics.co2_ppm > Real(0))) {
    throw ConfigError("physics.co2_ppm 必须 > 0，实际 " + fmt(physics.co2_ppm));
  }
  if (physics.soil_layers < 1) {
    throw ConfigError("physics.soil_layers 必须 >= 1，实际 " + std::to_string(physics.soil_layers));
  }
  if (physics.radiation_cadence < Real(0)) {
    throw ConfigError("physics.radiation_cadence 不能为负，实际 " +
                      fmt(physics.radiation_cadence));
  }

  // ------------------------------- 并行 ---------------------------------
  if (parallel.px < 1 || parallel.py < 1) {
    throw ConfigError("parallel.px/py 必须 >= 1，实际 px=" + std::to_string(parallel.px) +
                      " py=" + std::to_string(parallel.py));
  }
  if (parallel.px > domain.nx || parallel.py > domain.ny) {
    throw ConfigError("并行进程网格 (" + std::to_string(parallel.px) + "x" +
                      std::to_string(parallel.py) + ") 超过网格点数 (" +
                      std::to_string(domain.nx) + "x" + std::to_string(domain.ny) + ")");
  }
  if (parallel.px * parallel.py > domain.nx * domain.ny) {
    throw ConfigError("parallel.px*parallel.py 不能超过 nx*ny");
  }
  // 不整除时只给出"警告式"说明：Decomposition 支持余数分配，各进程点数相差 1。
  if (domain.nx % parallel.px != 0) {
    VIBE_WARN("[config] parallel.px=", parallel.px, " 不能整除 domain.nx=", domain.nx,
              "；grid::Decomposition 将按余数分配（各进程点数最多相差 1），不影响正确性");
  }
  if (domain.ny % parallel.py != 0) {
    VIBE_WARN("[config] parallel.py=", parallel.py, " 不能整除 domain.ny=", domain.ny,
              "；grid::Decomposition 将按余数分配（各进程点数最多相差 1），不影响正确性");
  }
  require_in_list(parallel.io_backend, "parallel.io_backend",
                  {kIoBackends[0], kIoBackends[1], kIoBackends[2]});
}

// ---------------------------------------------------------------------------
// 建议时间步
//
// 水平（声波-重力波显式稳定，[T5]）：
//     Δt_h = CFL * min(Δx, Δy) / (U_max + c_s),  U_max = 100 m/s, c_s = 350 m/s
// 垂直（声波子步，[D6]）：
//     Δt_a = CFL * Δz_min / c_s,   Δt <= n_sub * Δt_a
// 取两者最小值。Δz_min 在有显式 zeta 时取 z_top*min(Δζ)，否则取
// domain.zeta_first_thickness（近地层最薄层，m）。
// 复杂度：O(nz)。
// ---------------------------------------------------------------------------
Real ModelConfig::suggested_dt() const {
  constexpr Real kMaxWind = Real(100);
  constexpr Real kSoundSpeed = Real(350);
  const Real h_min = std::min(domain.dx, domain.dy);
  const Real dt_horizontal = time.cfl_target * h_min / (kMaxWind + kSoundSpeed);

  Real dz_min = domain.zeta_first_thickness;
  if (domain.zeta.size() >= 2) {
    Real dmin = domain.zeta[1] - domain.zeta[0];
    for (Size i = 1; i < domain.zeta.size(); ++i) {
      dmin = std::min(dmin, domain.zeta[i] - domain.zeta[i - 1]);
    }
    if (dmin > Real(0)) dz_min = dmin * domain.z_top;
  }
  const Int n_sub = std::max<Int>(1, time.acoustic_substeps);
  const Real dt_acoustic = time.cfl_target * dz_min / kSoundSpeed;
  const Real dt_vertical = dt_acoustic * static_cast<Real>(n_sub);

  const Real dt = std::min(dt_horizontal, dt_vertical);
  return dt > Real(0) ? dt : dt_horizontal;
}

// ---------------------------------------------------------------------------
// 配置指纹：FNV-1a 64 位哈希（[B11] 第 1 章哈希与校验思想）。
// 复杂度 O(1)（字段数固定）。
// ---------------------------------------------------------------------------
namespace detail {
std::string config_fingerprint(const ModelConfig& cfg);
}  // namespace detail

namespace {

std::string canonical_config_string(const ModelConfig& c) {
  std::ostringstream os;
  os << std::setprecision(17);
  os << "name=" << c.name << ';';
  os << "dom=" << c.domain.nx << ',' << c.domain.ny << ',' << c.domain.nz << ',' << c.domain.dx
     << ',' << c.domain.dy << ',' << c.domain.x0 << ',' << c.domain.y0 << ',' << c.domain.z_top
     << ',' << c.domain.terrain << ',' << c.domain.zeta_first_thickness << ','
     << c.domain.zeta_stretch << ',' << (c.domain.variable_resolution ? 1 : 0) << ','
     << c.domain.h_min << ',' << c.domain.h_max << ',' << c.domain.refinement << ','
     << c.domain.refinement_cx << ',' << c.domain.refinement_cy << ','
     << c.domain.refinement_radius << ',' << c.domain.refinement_transition << ','
     << (c.domain.periodic_x ? 1 : 0) << ',' << (c.domain.periodic_y ? 1 : 0) << ',';
  os << "zeta=";
  for (const Real z : c.domain.zeta) os << z << ',';
  os << ';';
  os << "time=" << c.time.dt << ',' << c.time.run_length << ',' << c.time.acoustic_substeps << ','
     << c.time.integrator << ',' << c.time.cfl_target << ',' << (c.time.adaptive_dt ? 1 : 0)
     << ',' << c.time.output_interval << ',' << c.time.restart_interval << ','
     << c.time.start_time << ';';
  os << "num=" << c.numerics.advection << ',' << (c.numerics.flux_form ? 1 : 0) << ','
     << c.numerics.divergence_damping << ',' << c.numerics.divergence_order << ','
     << c.numerics.sponge_alpha << ',' << c.numerics.sponge_start_fraction << ','
     << c.numerics.helmholtz_solver << ',' << c.numerics.helmholtz_max_iter << ','
     << c.numerics.helmholtz_tol << ',' << (c.numerics.weno ? 1 : 0) << ',' << c.numerics.coriolis
     << ',' << c.numerics.fplane_latitude << ';';
  os << "nest=" << (c.nesting.enabled ? 1 : 0) << ',' << c.nesting.levels << ','
     << c.nesting.ratio << ',' << (c.nesting.two_way ? 1 : 0) << ',' << c.nesting.boundary_zone
     << ',' << c.nesting.feedback_interval << ';';
  os << "phys=" << c.physics.microphysics << ',' << c.physics.radiation << ',' << c.physics.pbl
     << ',' << c.physics.surface << ',' << c.physics.cumulus << ','
     << c.physics.radiation_cadence << ',' << (c.physics.use_cloud_fraction ? 1 : 0) << ','
     << c.physics.co2_ppm << ',' << c.physics.aerosol_optical_depth << ','
     << c.physics.soil_layers << ',' << (c.physics.enable_tendency_physics ? 1 : 0) << ';';
  os << "par=" << c.parallel.px << ',' << c.parallel.py << ',' << c.parallel.io_backend << ','
     << (c.parallel.write_double ? 1 : 0);
  return os.str();
}

}  // namespace

namespace detail {

std::string config_fingerprint(const ModelConfig& cfg) {
  const std::string canon = canonical_config_string(cfg);
  std::uint64_t h = 1469598103934665603ull;  // FNV offset basis
  for (const unsigned char c : canon) {
    h ^= static_cast<std::uint64_t>(c);
    h *= 1099511628211ull;  // FNV prime
  }
  std::ostringstream os;
  os << std::hex << std::setw(16) << std::setfill('0') << h;
  return os.str();
}

}  // namespace detail

std::string ModelConfig::describe() const {
  std::ostringstream os;
  os << "ModelConfig '" << name << "'";
  if (!description.empty()) os << " — " << description;
  os << "\n  domain   : " << domain.nx << "x" << domain.ny << "x" << domain.nz << "  dx="
     << fmt(domain.dx) << " m dy=" << fmt(domain.dy) << " m z_top=" << fmt(domain.z_top)
     << " m" << (domain.zeta.empty() ? "（自动拉伸 zeta）" : "（显式 zeta）");
  if (domain.variable_resolution) {
    os << " 变分辨率[refinement=" << domain.refinement << " h_min=" << fmt(domain.h_min)
       << " h_max=" << fmt(domain.h_max) << "]";
  }
  if (!domain.terrain.empty()) os << " terrain=" << domain.terrain;
  os << "\n  time     : dt=" << fmt(time.dt) << " s run=" << fmt(time.run_length)
     << " s substeps=" << time.acoustic_substeps << " integrator=" << time.integrator
     << " cfl=" << fmt(time.cfl_target) << " adaptive=" << (time.adaptive_dt ? "on" : "off");
  os << "\n  numerics : advection=" << numerics.advection
     << " flux_form=" << (numerics.flux_form ? "on" : "off")
     << " div_damping=" << fmt(numerics.divergence_damping)
     << " helmholtz=" << numerics.helmholtz_solver << "(max_iter=" << numerics.helmholtz_max_iter
     << ", tol=" << fmt(numerics.helmholtz_tol) << ")"
     << " coriolis=" << numerics.coriolis;
  os << "\n  nesting  : " << (nesting.enabled ? "on" : "off")
     << " levels=" << nesting.levels << " ratio=" << nesting.ratio
     << " two_way=" << (nesting.two_way ? "on" : "off")
     << " boundary_zone=" << nesting.boundary_zone;
  os << "\n  physics  : mp=" << physics.microphysics << " rad=" << physics.radiation
     << " pbl=" << physics.pbl << " sfc=" << physics.surface << " cu=" << physics.cumulus
     << " co2=" << fmt(physics.co2_ppm) << " ppm";
  os << "\n  parallel : px=" << parallel.px << " py=" << parallel.py
     << " io=" << parallel.io_backend << " out=" << parallel.output_dir;
  os << "\n  suggested_dt = " << fmt(suggested_dt()) << " s";
  os << "\n  fingerprint  = " << detail::config_fingerprint(*this);
  return os.str();
}

// ===========================================================================
// DaConfig
// ===========================================================================

DaConfig DaConfig::from_yaml(const YamlNode& root_in) {
  DaConfig c;
  if (root_in.is_null()) return c;
  if (!root_in.is_mapping()) {
    throw ConfigError("同化配置根节点必须是映射，得到 " + type_name(root_in));
  }
  const YamlNode* wrapper = nullptr;
  for (const char* k : {"da_4dvar", "4dvar", "da", "assimilation"}) {
    if (root_in.has(k) && root_in[k].is_mapping()) { wrapper = &root_in[k]; break; }
  }
  const YamlNode& s = wrapper != nullptr ? *wrapper : root_in;

  c.enabled = bool_field(find_key(s, {"enabled"}), "da.enabled", c.enabled);

  const YamlNode& w = resolve_section(s, {"window", "assimilation_window"}, "da.window");
  c.window.start = str_field(find_key(w, {"start", "start_time"}), "da.window.start",
                             c.window.start);
  c.window.length = real_field(find_key(w, {"length", "window_length"}), "da.window.length",
                               c.window.length);
  c.window.slot_interval =
      int_field(find_key(w, {"slot_interval", "n_slots"}), "da.window.slot_interval",
                c.window.slot_interval);

  const YamlNode& b =
      resolve_section(s, {"background", "background_error", "b"}, "da.background");
  c.background.method =
      str_field(find_key(b, {"method", "type"}), "da.background.method", c.background.method);
  c.background.balance = str_field(find_key(b, {"balance"}), "da.background.balance",
                                   c.background.balance);
  const YamlNode& corr = resolve_section(b, {"correlation", "scale"}, "da.background.correlation");
  c.background.horizontal_length_scale = real_field(
      find_key(b, {"horizontal_length_scale", "horizontal_scale", "correlation_scale"}),
      "da.background.horizontal_length_scale", c.background.horizontal_length_scale);
  if (const YamlNode* nested = find_key(corr, {"horizontal", "horizontal_length"})) {
    c.background.horizontal_length_scale =
        real_field(nested, "da.background.correlation.horizontal",
                   c.background.horizontal_length_scale);
  }
  c.background.vertical_length_scale =
      real_field(find_key(b, {"vertical_length_scale", "vertical_scale"}),
                 "da.background.vertical_length_scale", c.background.vertical_length_scale);
  if (const YamlNode* nested = find_key(corr, {"vertical", "vertical_length"})) {
    c.background.vertical_length_scale =
        real_field(nested, "da.background.correlation.vertical",
                   c.background.vertical_length_scale);
  }
  c.background.diffusion_order =
      real_field(find_key(b, {"diffusion_order"}), "da.background.diffusion_order",
                 c.background.diffusion_order);
  c.background.variance_scale =
      real_field(find_key(b, {"variance_scale"}), "da.background.variance_scale",
                 c.background.variance_scale);
  c.background.vertical_eof_fractions =
      real_list_field(find_key(b, {"vertical_eof_fractions"}), "da.background.vertical_eof_fractions",
                      c.background.vertical_eof_fractions);
  c.background.n_eofs =
      static_cast<int>(int_field(find_key(b, {"n_eofs"}), "da.background.n_eofs",
                                 c.background.n_eofs));
  c.background.hybrid_weight =
      real_field(find_key(b, {"hybrid_weight"}), "da.background.hybrid_weight",
                 c.background.hybrid_weight);
  c.background.ensemble_dir = str_field(find_key(b, {"ensemble_dir"}), "da.background.ensemble_dir",
                                        c.background.ensemble_dir);

  const YamlNode& cost = resolve_section(s, {"cost", "cost_function"}, "da.cost");
  c.cost.use_penalty = bool_field(find_key(cost, {"use_penalty"}), "da.cost.use_penalty",
                                  c.cost.use_penalty);
  c.cost.penalty_weight =
      real_field(find_key(cost, {"penalty_weight"}), "da.cost.penalty_weight", c.cost.penalty_weight);
  c.cost.use_digital_filter =
      bool_field(find_key(cost, {"use_digital_filter"}), "da.cost.use_digital_filter",
                 c.cost.use_digital_filter);
  c.cost.digital_filter_weight =
      real_field(find_key(cost, {"digital_filter_weight"}), "da.cost.digital_filter_weight",
                 c.cost.digital_filter_weight);
  c.cost.use_model_error = bool_field(find_key(cost, {"use_model_error"}), "da.cost.use_model_error",
                                      c.cost.use_model_error);
  c.cost.model_error_scale =
      real_field(find_key(cost, {"model_error_scale"}), "da.cost.model_error_scale",
                 c.cost.model_error_scale);
  c.cost.use_weak_constraint =
      bool_field(find_key(cost, {"use_weak_constraint"}), "da.cost.use_weak_constraint",
                 c.cost.use_weak_constraint);

  const YamlNode& minim = resolve_section(s, {"minimizer", "minimisation"}, "da.minimizer");
  c.minimizer.method =
      str_field(find_key(minim, {"method", "solver"}), "da.minimizer.method", c.minimizer.method);
  c.minimizer.max_iterations =
      static_cast<int>(int_field(find_key(minim, {"max_iterations", "max_iter"}),
                                 "da.minimizer.max_iterations", c.minimizer.max_iterations));
  c.minimizer.gradient_tolerance =
      real_field(find_key(minim, {"gradient_tolerance", "grad_tol"}),
                 "da.minimizer.gradient_tolerance", c.minimizer.gradient_tolerance);
  c.minimizer.cost_tolerance =
      real_field(find_key(minim, {"cost_tolerance", "cost_tol"}), "da.minimizer.cost_tolerance",
                 c.minimizer.cost_tolerance);
  c.minimizer.memory =
      static_cast<int>(int_field(find_key(minim, {"memory", "n_memory"}), "da.minimizer.memory",
                                 c.minimizer.memory));
  c.minimizer.initial_step =
      real_field(find_key(minim, {"initial_step"}), "da.minimizer.initial_step",
                 c.minimizer.initial_step);
  c.minimizer.preconditioned =
      bool_field(find_key(minim, {"preconditioned"}), "da.minimizer.preconditioned",
                 c.minimizer.preconditioned);

  c.outer_loops =
      static_cast<int>(int_field(find_key(s, {"outer_loops", "n_outer"}), "da.outer_loops",
                                 c.outer_loops));
  c.inner_loops =
      static_cast<int>(int_field(find_key(s, {"inner_loops", "n_inner"}), "da.inner_loops",
                                 c.inner_loops));
  c.inner_resolution_factor =
      real_field(find_key(s, {"inner_resolution_factor", "resolution_factor"}),
                 "da.inner_resolution_factor", c.inner_resolution_factor);
  c.assimilate_moisture =
      bool_field(find_key(s, {"assimilate_moisture"}), "da.assimilate_moisture",
                 c.assimilate_moisture);
  c.use_tl_ad = bool_field(find_key(s, {"use_tl_ad"}), "da.use_tl_ad", c.use_tl_ad);
  c.obs_list = str_field(find_key(s, {"obs_list", "observations"}), "da.obs_list", c.obs_list);
  c.bias_correction = str_field(find_key(s, {"bias_correction"}), "da.bias_correction",
                                c.bias_correction);
  c.check_adjoint = bool_field(find_key(s, {"check_adjoint", "adjoint_check"}), "da.check_adjoint",
                               c.check_adjoint);
  return c;
}

DaConfig DaConfig::load(const std::string& path) {
  DaConfig c = from_yaml(YamlNode::parse_file(path));
  c.validate();
  return c;
}

void DaConfig::validate() const {
  if (!(window.length > Real(0))) {
    throw ConfigError("window.length 必须 > 0，实际 " + fmt(window.length));
  }
  if (window.slot_interval < 1) {
    throw ConfigError("window.slot_interval 必须 >= 1，实际 " +
                      std::to_string(window.slot_interval));
  }
  if (outer_loops < 1) {
    throw ConfigError("outer_loops 必须 >= 1，实际 " + std::to_string(outer_loops));
  }
  if (inner_loops < 1) {
    throw ConfigError("inner_loops 必须 >= 1，实际 " + std::to_string(inner_loops));
  }
  if (!(background.horizontal_length_scale > Real(0))) {
    throw ConfigError("background.horizontal_length_scale 必须 > 0，实际 " +
                      fmt(background.horizontal_length_scale));
  }
  if (!(background.vertical_length_scale > Real(0))) {
    throw ConfigError("background.vertical_length_scale 必须 > 0，实际 " +
                      fmt(background.vertical_length_scale));
  }
  if (!(background.diffusion_order >= Real(1))) {
    throw ConfigError("background.diffusion_order 必须 >= 1，实际 " +
                      fmt(background.diffusion_order));
  }
  if (!(background.variance_scale > Real(0))) {
    throw ConfigError("background.variance_scale 必须 > 0，实际 " +
                      fmt(background.variance_scale));
  }
  if (background.hybrid_weight < Real(0) || background.hybrid_weight > Real(1)) {
    throw ConfigError("background.hybrid_weight 必须落在 [0,1]，实际 " +
                      fmt(background.hybrid_weight));
  }
  if (background.n_eofs < 0) {
    throw ConfigError("background.n_eofs 不能为负，实际 " + std::to_string(background.n_eofs));
  }
  require_in_list(background.method, "background.method",
                  {kBackgroundMethods[0], kBackgroundMethods[1], kBackgroundMethods[2]});
  require_in_list(background.balance, "background.balance",
                  {kBalanceMethods[0], kBalanceMethods[1], kBalanceMethods[2]});
  require_in_list(minimizer.method, "minimizer.method",
                  {kMinimizers[0], kMinimizers[1], kMinimizers[2]});
  if (minimizer.max_iterations < 1) {
    throw ConfigError("minimizer.max_iterations 必须 >= 1，实际 " +
                      std::to_string(minimizer.max_iterations));
  }
  if (!(minimizer.gradient_tolerance > Real(0))) {
    throw ConfigError("minimizer.gradient_tolerance 必须 > 0，实际 " +
                      fmt(minimizer.gradient_tolerance));
  }
  if (!(minimizer.cost_tolerance > Real(0))) {
    throw ConfigError("minimizer.cost_tolerance 必须 > 0，实际 " + fmt(minimizer.cost_tolerance));
  }
  if (minimizer.memory < 1) {
    throw ConfigError("minimizer.memory 必须 >= 1，实际 " + std::to_string(minimizer.memory));
  }
  if (!(minimizer.initial_step > Real(0))) {
    throw ConfigError("minimizer.initial_step 必须 > 0，实际 " + fmt(minimizer.initial_step));
  }
  if (!(inner_resolution_factor >= Real(1))) {
    throw ConfigError("inner_resolution_factor 必须 >= 1，实际 " + fmt(inner_resolution_factor));
  }
  if (cost.penalty_weight < Real(0)) {
    throw ConfigError("cost.penalty_weight 不能为负，实际 " + fmt(cost.penalty_weight));
  }
  if (cost.digital_filter_weight < Real(0)) {
    throw ConfigError("cost.digital_filter_weight 不能为负，实际 " +
                      fmt(cost.digital_filter_weight));
  }
  if (cost.model_error_scale < Real(0)) {
    throw ConfigError("cost.model_error_scale 不能为负，实际 " + fmt(cost.model_error_scale));
  }
}

// ===========================================================================
// VerifyConfig
// ===========================================================================

VerifyConfig VerifyConfig::from_yaml(const YamlNode& root_in) {
  VerifyConfig c;
  if (root_in.is_null()) return c;
  if (!root_in.is_mapping()) {
    throw ConfigError("检验配置根节点必须是映射，得到 " + type_name(root_in));
  }
  const YamlNode* wrapper = nullptr;
  for (const char* k : {"verify", "verification"}) {
    if (root_in.has(k) && root_in[k].is_mapping()) { wrapper = &root_in[k]; break; }
  }
  const YamlNode& s = wrapper != nullptr ? *wrapper : root_in;

  c.variables =
      str_list_field(find_key(s, {"variables", "fields"}), "verify.variables", c.variables);
  c.thresholds =
      real_list_field(find_key(s, {"thresholds", "threshold"}), "verify.thresholds", c.thresholds);
  c.methods = str_list_field(find_key(s, {"methods", "metrics"}), "verify.methods", c.methods);
  c.neighborhood_radii =
      int_list_field(find_key(s, {"neighborhood_radii", "radii", "neighborhood"}),
                     "verify.neighborhood_radii", c.neighborhood_radii);
  c.matchup = str_field(find_key(s, {"matchup", "matchup_method"}), "verify.matchup", c.matchup);
  c.temporal_tolerance = real_field(find_key(s, {"temporal_tolerance"}), "verify.temporal_tolerance",
                                    c.temporal_tolerance);
  c.climatology = str_field(find_key(s, {"climatology", "climatology_file"}), "verify.climatology",
                            c.climatology);
  c.compute_ensemble_scores =
      bool_field(find_key(s, {"compute_ensemble_scores", "ensemble_scores"}),
                 "verify.compute_ensemble_scores", c.compute_ensemble_scores);
  c.output_format =
      str_field(find_key(s, {"output_format", "format"}), "verify.output_format", c.output_format);
  return c;
}

VerifyConfig VerifyConfig::load(const std::string& path) {
  return from_yaml(YamlNode::parse_file(path));
}

}  // namespace vibe::config
