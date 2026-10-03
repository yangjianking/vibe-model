/// @file main.cpp
/// @brief VIBE-Model 命令行入口。
///
/// 用法
/// ----
///   vibe_model --config config/model.yaml [--nests config/nests.yaml]
///              [--da config/da_4dvar.yaml] [--verify config/verify.yaml]
///              [--mode forecast|assimilation|verification|both]
///              [--ic warm_bubble|density_current|mountain_wave|...]
///              [--restart path]
///              [--log-level info] [--dry-run]
///
/// 退出码
/// ------
///   0 正常；1 配置/运行错误；2 数值发散；3 命令行错误。

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/common/logging.hpp"
#include "vibe/common/mpi_wrapper.hpp"
#include "vibe/config/config.hpp"
#include "vibe/driver/driver.hpp"

namespace {

void print_usage() {
  std::cout
      << "VIBE-Model —— 可嵌套变分辨率半隐式数值天气预报模式\n\n"
      << "用法： vibe_model [选项]\n\n"
      << "选项：\n"
      << "  --config <path>      模式配置文件（必需；可重复，后者覆盖前者）\n"
      << "  --da <path>          同化配置文件\n"
      << "  --verify <path>      检验配置文件\n"
      << "  --mode <name>        forecast | assimilation | verification | both\n"
      << "  --ic <name>          理想试验：warm_bubble / cold_bubble / density_current /\n"
      << "                       mountain_wave / inertia_gravity_wave / rising_thermal /\n"
      << "                       baroclinic_wave / resting_isothermal / balanced_jet / from_file\n"
      << "  --restart <path>     从检查点重启\n"
      << "  --log-level <l>      trace | debug | info | warn | error\n"
      << "  --description <text> 覆盖运行描述\n"
      << "  --dry-run            只装配与校验，不积分\n"
      << "  --help               显示本帮助\n";
}

vibe::common::LogLevel parse_level(const std::string& s) {
  if (s == "trace") return vibe::common::LogLevel::Trace;
  if (s == "debug") return vibe::common::LogLevel::Debug;
  if (s == "warn") return vibe::common::LogLevel::Warn;
  if (s == "error") return vibe::common::LogLevel::Error;
  if (s == "off") return vibe::common::LogLevel::Off;
  return vibe::common::LogLevel::Info;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace vibe;

  std::vector<std::string> config_paths;   // 允许重复 --config 做分层叠加
  std::string da_path;
  std::string verify_path;
  std::string mode_name = "forecast";
  std::string ic_name;
  std::string restart_path;
  std::string description;
  std::string level = "info";
  bool dry_run = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "选项 " << name << " 缺少参数\n";
        std::exit(3);
      }
      return argv[++i];
    };
    if (a == "--help" || a == "-h") { print_usage(); return 0; }
    else if (a == "--config") config_paths.push_back(next("--config"));
    else if (a == "--da") da_path = next("--da");
    else if (a == "--verify") verify_path = next("--verify");
    else if (a == "--mode") mode_name = next("--mode");
    else if (a == "--ic") ic_name = next("--ic");
    else if (a == "--restart") restart_path = next("--restart");
    else if (a == "--description") description = next("--description");
    else if (a == "--log-level") level = next("--log-level");
    else if (a == "--dry-run") dry_run = true;
    else {
      std::cerr << "未知选项：" << a << "\n";
      print_usage();
      return 3;
    }
  }

  if (config_paths.empty()) {
    std::cerr << "必须指定 --config\n";
    print_usage();
    return 3;
  }

  common::Logger::instance().set_level(parse_level(level));
  common::Comm::initialize(&argc, &argv);
  common::Logger::instance().set_rank(common::Comm::world().rank());

  int rc = 0;
  try {
    // 分层叠加：后给的配置文件覆盖先给的（映射递归合并，序列整体替换）
    config::ModelConfig cfg = config::ModelConfig::load(config_paths, /*allow_missing=*/false);
    config::DaConfig da_cfg;
    if (!da_path.empty()) da_cfg = config::DaConfig::load(da_path);
    config::VerifyConfig v_cfg;
    if (!verify_path.empty()) v_cfg = config::VerifyConfig::load(verify_path);
    if (!description.empty()) cfg.description = description;

    driver::Driver drv(cfg, da_cfg, v_cfg);

    driver::IcOptions ic;
    if (!ic_name.empty()) ic.case_type = driver::case_from_string(ic_name);
    else if (!cfg.domain.terrain.empty()) ic.case_type = driver::IdealizedCase::MountainWave;
    ic.input_file = cfg.domain.terrain;
    drv.set_initial_conditions(ic);

    drv.initialize();
    common::Logger::instance().write(common::LogLevel::Info, drv.describe());

    if (!restart_path.empty()) drv.read_checkpoint(restart_path);

    if (!dry_run) {
      driver::RunMode mode = driver::RunMode::Forecast;
      if (mode_name == "assimilation") mode = driver::RunMode::Assimilation;
      else if (mode_name == "verification") mode = driver::RunMode::Verification;
      else if (mode_name == "both") mode = driver::RunMode::Both;
      const driver::RunSummary s = drv.run(mode);
      if (s.diverged) rc = 2;
    }
    drv.finalize();
  } catch (const Error& e) {
    std::cerr << "VIBE-Model 错误：" << e.what() << "\n";
    rc = 1;
  } catch (const std::exception& e) {
    std::cerr << "未处理异常：" << e.what() << "\n";
    rc = 1;
  }

  common::Comm::finalize();
  return rc;
}
