/// @file verify_cli.cpp
/// @brief 独立检验工具：读取 .vibebin 二进制场文件，计算评分并打印 CSV/JSON。
///
/// 刻意**不包含任何 netCDF 头文件**（架构第 3 节：外部文件格式只在 io 层出现）。
/// 这里只实现 io 层的 .vibebin 约定，保证检验工具可独立编译、无第三方依赖。
///
/// .vibebin 布局（小端，原生字节序；与 io 层约定一致）
/// -----------------------------------------------
///   offset  type        field
///   0       char[8]     magic = "VIBEBIN1"
///   8       int32       nx
///   12      int32       ny
///   16      int32       nz
///   20      int32       real_kind   0 = float64, 1 = float32
///   24      float64     time        （Unix epoch 秒，UTC）
///   32      int32       nvars
///   36      ...         每个变量：
///                           char[32] name（NUL 填充）
///                           nx*ny*nz 个 real_kind 对应类型的数据
///
/// 用法
/// ----
///   verify_cli <file.vibebin> [options]
///     --list                    只列出变量与尺寸后退出
///     --pair FCST=OBS           显式指定预报/观测变量对（可重复）
///     --variable NAME           只检验该预报变量
///     --type TYPE               continuous|categorical|probabilistic|spatial
///     --thresholds t1,t2,...    事件阈值列表（默认 0）
///     --radius R                空间检验的单一半径（0 = 用默认多尺度）
///     --max-radius R            多尺度 FSS 最大半径（默认 10）
///     --json                    以 JSON 输出（默认 CSV）
///     --help
///
/// 复杂度：读取 O(nx*ny*nz*nvars)，评分 O(N) ~ O(N*R)。

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/common/types.hpp"
#include "vibe/verify/verification.hpp"

namespace {

using vibe::Int;
using vibe::Real;

struct BinaryField {
  std::string name;
  std::vector<Real> data;
};

struct BinaryFile {
  int nx = 0;
  int ny = 0;
  int nz = 0;
  int real_kind = 0;
  double time = 0.0;
  std::vector<BinaryField> fields;
};

/// 从内存读取一个小端 int32。
std::int32_t read_i32(std::istream& in, const char* what) {
  std::int32_t v = 0;
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  if (!in) throw vibe::IoError(std::string("读取 int32 失败: ") + what);
  return v;
}

double read_f64(std::istream& in, const char* what) {
  double v = 0.0;
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  if (!in) throw vibe::IoError(std::string("读取 float64 失败: ") + what);
  return v;
}

/// 读取 .vibebin（见文件头布局）。O(文件大小)。
BinaryFile read_vibebin(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vibe::IoError("无法打开文件: " + path);

  char magic[8] = {0};
  in.read(magic, 8);
  if (!in || std::memcmp(magic, "VIBEBIN1", 8) != 0) {
    throw vibe::IoError("magic 不匹配（期望 VIBEBIN1）: " + path);
  }

  BinaryFile f;
  f.nx = read_i32(in, "nx");
  f.ny = read_i32(in, "ny");
  f.nz = read_i32(in, "nz");
  f.real_kind = read_i32(in, "real_kind");
  f.time = read_f64(in, "time");
  const std::int32_t nvars = read_i32(in, "nvars");

  if (f.nx <= 0 || f.ny <= 0 || f.nz <= 0) {
    throw vibe::IoError(".vibebin 网格尺寸非正");
  }
  if (f.real_kind != 0 && f.real_kind != 1) {
    throw vibe::IoError(".vibebin real_kind 只支持 0(float64) / 1(float32)");
  }
  if (nvars < 0) throw vibe::IoError(".vibebin nvars 为负");

  const std::size_t n = static_cast<std::size_t>(f.nx) *
                        static_cast<std::size_t>(f.ny) *
                        static_cast<std::size_t>(f.nz);
  for (std::int32_t v = 0; v < nvars; ++v) {
    char name[32] = {0};
    in.read(name, 32);
    if (!in) throw vibe::IoError("读取变量名失败（变量序号 " + std::to_string(v) + "）");
    BinaryField bf;
    std::size_t name_len = 0;
    while (name_len < 32 && name[name_len] != '\0') ++name_len;
    bf.name.assign(name, name_len);
    bf.data.resize(n);
    if (f.real_kind == 0) {
      std::vector<double> tmp(n);
      in.read(reinterpret_cast<char*>(tmp.data()),
              static_cast<std::streamsize>(n * sizeof(double)));
      if (!in) throw vibe::IoError("读取 float64 数据失败: " + bf.name);
      for (std::size_t k = 0; k < n; ++k) bf.data[k] = static_cast<Real>(tmp[k]);
    } else {
      std::vector<float> tmp(n);
      in.read(reinterpret_cast<char*>(tmp.data()),
              static_cast<std::streamsize>(n * sizeof(float)));
      if (!in) throw vibe::IoError("读取 float32 数据失败: " + bf.name);
      for (std::size_t k = 0; k < n; ++k) bf.data[k] = static_cast<Real>(tmp[k]);
    }
    f.fields.push_back(std::move(bf));
  }
  return f;
}

/// 在变量表中查找。
const BinaryField* find_field(const BinaryFile& f, const std::string& name) {
  for (const BinaryField& b : f.fields) {
    if (b.name == name) return &b;
  }
  return nullptr;
}

void print_usage(std::ostream& os) {
  os << "用法: verify_cli <file.vibebin> [options]\n"
     << "  --list                 列出变量与尺寸\n"
     << "  --pair FCST=OBS        指定预报/观测变量对（可重复）\n"
     << "  --variable NAME        只检验该预报变量\n"
     << "  --type TYPE            continuous|categorical|probabilistic|spatial\n"
     << "  --thresholds t1,t2     事件阈值列表（默认 0）\n"
     << "  --radius R             空间检验单一半径（0 = 默认多尺度）\n"
     << "  --max-radius R         多尺度 FSS 最大半径（默认 10）\n"
     << "  --json                 以 JSON 输出（默认 CSV）\n"
     << "  --help                 显示本帮助\n";
}

}  // namespace

int main(int argc, char** argv) {
  using namespace vibe;
  using namespace vibe::verify;

  std::string path;
  bool list_only = false;
  bool as_json = false;
  std::string type_text = "continuous";
  std::string variable_filter;
  std::vector<Real> thresholds;
  bool radius_set = false;
  int radius = 0;
  int max_radius = 10;
  std::vector<std::pair<std::string, std::string>> pairs;

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&](const char* what) -> std::string {
        if (i + 1 >= argc) throw ConfigError(std::string("参数 ") + what + " 缺少取值");
        return argv[++i];
      };
      if (a == "--help" || a == "-h") {
        print_usage(std::cout);
        return 0;
      } else if (a == "--list") {
        list_only = true;
      } else if (a == "--json") {
        as_json = true;
      } else if (a == "--pair") {
        const std::string v = next("--pair");
        const std::size_t eq = v.find('=');
        if (eq == std::string::npos) throw ConfigError("--pair 需要 FCST=OBS 形式");
        pairs.emplace_back(v.substr(0, eq), v.substr(eq + 1));
      } else if (a == "--variable") {
        variable_filter = next("--variable");
      } else if (a == "--type") {
        type_text = next("--type");
      } else if (a == "--thresholds") {
        thresholds = VerifyConfig::parse_real_list(next("--thresholds"));
      } else if (a == "--radius") {
        const std::vector<int> rv = VerifyConfig::parse_int_list(next("--radius"));
        if (rv.size() != 1) throw ConfigError("--radius 需要一个整数");
        radius = rv[0];
        radius_set = true;
      } else if (a == "--max-radius") {
        const std::vector<int> mv = VerifyConfig::parse_int_list(next("--max-radius"));
        if (mv.size() != 1) throw ConfigError("--max-radius 需要一个整数");
        max_radius = mv[0];
      } else if (!a.empty() && a[0] == '-') {
        throw ConfigError("未知参数: " + a);
      } else if (path.empty()) {
        path = a;
      } else {
        throw ConfigError("多余的位置参数: " + a);
      }
    }

    if (path.empty()) {
      print_usage(std::cerr);
      return 2;
    }

    const BinaryFile file = read_vibebin(path);

    if (list_only) {
      std::printf("nx=%d ny=%d nz=%d real_kind=%d time=%.6f nvars=%zu\n",
                  file.nx, file.ny, file.nz, file.real_kind, file.time,
                  file.fields.size());
      for (const BinaryField& b : file.fields) {
        std::printf("  %-32s %zu\n", b.name.c_str(), b.data.size());
      }
      return 0;
    }

    // 变量配对：显式 --pair 优先；否则用 "X" <-> "X_obs" 约定。
    if (pairs.empty()) {
      for (const BinaryField& b : file.fields) {
        const std::string suffix = "_obs";
        if (b.name.size() > suffix.size() &&
            b.name.compare(b.name.size() - suffix.size(), suffix.size(), suffix) == 0) {
          continue;  // 观测字段本身不作为预报
        }
        if (find_field(file, b.name + "_obs") != nullptr) {
          pairs.emplace_back(b.name, b.name + "_obs");
        }
      }
    }
    if (pairs.empty() && file.fields.size() == 2) {
      std::fprintf(stderr,
                   "[verify_cli] 未发现 X/X_obs 命名约定，回退为把前两个变量作为一对\n");
      pairs.emplace_back(file.fields[0].name, file.fields[1].name);
    }
    if (pairs.empty()) {
      throw ConfigError("无法确定预报/观测变量对，请用 --pair FCST=OBS 显式指定");
    }

    const MatchType type = match_type_from_string(type_text);

    VerifyConfig cfg;
    cfg.fss_max_radius = max_radius;
    if (!thresholds.empty()) {
      cfg.default_variable.thresholds = thresholds;
    }
    cfg.default_variable.type = type;
    if (radius_set) {
      cfg.default_variable.radii = {radius};
    }

    std::vector<FieldSample> samples;
    for (const auto& p : pairs) {
      const BinaryField* fc = find_field(file, p.first);
      const BinaryField* ob = find_field(file, p.second);
      if (fc == nullptr) throw IoError("未找到预报变量: " + p.first);
      if (ob == nullptr) throw IoError("未找到观测变量: " + p.second);
      if (!variable_filter.empty() && variable_filter != p.first) continue;
      FieldSample s;
      s.variable = p.first;
      s.nx = static_cast<Int>(file.nx);
      s.ny = static_cast<Int>(file.ny);
      s.nz = static_cast<Int>(file.nz);
      s.time = static_cast<Real>(file.time);
      s.forecast = fc->data;
      s.observation = ob->data;
      s.validate();
      samples.push_back(std::move(s));
    }
    if (samples.empty()) {
      throw ConfigError("按 --variable 过滤后没有可检验的变量");
    }

    const Verifier verifier(cfg);
    const VerificationResult result = verifier.run(samples);
    if (as_json) {
      std::cout << result.to_json();
    } else {
      std::cout << result.to_csv(cfg.csv_delimiter, cfg.missing_token);
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[verify_cli] 错误: %s\n", e.what());
    return 1;
  }
}
