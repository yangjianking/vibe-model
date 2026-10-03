/// @file adjoint_check.cpp
/// @brief 独立的伴随点积检验工具（[A3] Sirkes & Tziperman 1997）。
///
/// 用途
/// ----
/// 在任何 TL/AD 代码改动之后，快速验证"转置正确"。检验式为
///
///     < M dx , dy >  ==  < dx , M^T dy >
///
/// 其中 M 为时间推进算子（本工具用切线性模式 M = TL，M^T = AD），
/// 内积按格点体积加权（与代价函数使用的度量一致）。
/// 数值上应满足相对误差 < 1e-8（TL/AD 是精确转置，误差只来自浮点求和顺序）。
///
/// 用法
/// ----
///     vibe_adjoint_check --config config/model.yaml --steps 20 --dt 6 --trials 5
///
/// 退出码：0 通过；1 失败；3 命令行错误。

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "vibe/common/error.hpp"
#include "vibe/common/mpi_wrapper.hpp"
#include "vibe/config/config.hpp"
#include "vibe/da/tangent_linear.hpp"
#include "vibe/driver/driver.hpp"
#include "vibe/dyn/equations.hpp"

namespace {

struct Options {
  std::string config;
  int steps = 20;
  Real dt = Real(6);
  int trials = 5;
  unsigned seed = 20240101;
  Real tolerance = Real(1e-8);
};

void usage() {
  std::cout << "vibe_adjoint_check —— 切线性/伴随点积检验\n\n"
            << "用法： vibe_adjoint_check --config <model.yaml> [选项]\n\n"
            << "选项：\n"
            << "  --steps <n>        传播步数（默认 20）\n"
            << "  --dt <seconds>     每步时长（默认 6）\n"
            << "  --trials <n>       随机试验次数（默认 5）\n"
            << "  --seed <n>         随机种子（默认 20240101）\n"
            << "  --tolerance <eps>  判定阈值（默认 1e-8）\n";
}

/// 生成随机扰动 dx（均匀分布），并保证与基础态同布局
void random_state(const vibe::grid::Grid& g, vibe::dyn::State& out, Real scale,
                  std::mt19937& gen) {
  std::uniform_real_distribution<Real> dist(-scale, scale);
  for (int sp = 0; sp < vibe::dyn::kNumSpecies; ++sp) {
    auto& f = out.field(static_cast<vibe::dyn::Species>(sp));
    for (vibe::Int k = 0; k < f.nz(); ++k)
      for (vibe::Int j = 0; j < f.ny(); ++j)
        for (vibe::Int i = 0; i < f.nx(); ++i) f(i, j, k) = dist(gen);
  }
  (void)g;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace vibe;

  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* n) -> std::string {
      if (i + 1 >= argc) { std::cerr << n << " 缺少参数\n"; std::exit(3); }
      return argv[++i];
    };
    if (a == "--help" || a == "-h") { usage(); return 0; }
    else if (a == "--config") opt.config = next("--config");
    else if (a == "--steps") opt.steps = std::stoi(next("--steps"));
    else if (a == "--dt") opt.dt = static_cast<Real>(std::stod(next("--dt")));
    else if (a == "--trials") opt.trials = std::stoi(next("--trials"));
    else if (a == "--seed") opt.seed = static_cast<unsigned>(std::stoul(next("--seed")));
    else if (a == "--tolerance") opt.tolerance = static_cast<Real>(std::stod(next("--tolerance")));
    else { std::cerr << "未知选项：" << a << "\n"; usage(); return 3; }
  }
  if (opt.config.empty()) { usage(); return 3; }

  common::Comm::initialize(&argc, &argv);
  int rc = 0;
  try {
    const config::ModelConfig cfg = config::ModelConfig::load(opt.config);
    config::DaConfig da_cfg;
    da_cfg.enabled = true;

    driver::Driver drv(cfg, da_cfg);
    driver::IcOptions ic;
    ic.case_type = driver::IdealizedCase::WarmBubble;
    drv.set_initial_conditions(ic);
    drv.initialize();

    da::TangentLinearModel tl(drv.grid(), drv.reference(), cfg, da_cfg);
    da::AdjointModel ad(drv.grid(), drv.reference(), cfg, da_cfg);

    std::mt19937 gen(opt.seed);
    Real worst = Real(0);
    int failures = 0;

    for (int t = 0; t < opt.trials; ++t) {
      const auto res = da::check_adjoint(tl, ad, drv.state(), opt.dt, opt.steps,
                                         opt.seed + static_cast<unsigned>(t),
                                         opt.tolerance);
      std::cout << "试验 " << (t + 1) << "/" << opt.trials
                << "  <M dx, dy> = " << res.lhs
                << "  <dx, M^T dy> = " << res.rhs
                << "  相对误差 = " << res.relative_error
                << "  " << (res.passed ? "通过" : "失败") << "\n";
      worst = std::max(worst, res.relative_error);
      if (!res.passed) ++failures;
    }

    std::cout << "\n最差相对误差 = " << worst
              << "（阈值 " << opt.tolerance << "），失败次数 = " << failures << "\n";
    rc = (failures == 0) ? 0 : 1;
  } catch (const Error& e) {
    std::cerr << "错误：" << e.what() << "\n";
    rc = 1;
  }

  common::Comm::finalize();
  return rc;
}
