/// @file test_config.cpp
/// @brief vibe::config 的单元测试：YAML 子集解析、合并、ModelConfig/DaConfig/
///        VerifyConfig 装配与校验。
///
/// 注意：vibe/common/error.hpp 与 vibe/common/test.hpp 都定义了 VIBE_CHECK 宏，
/// 因此先在库头文件之后 undef，再引入测试框架（不改动任何头文件）。

#include "vibe/config/config.hpp"
#include "vibe/config/yaml_node.hpp"

#include "vibe/common/error.hpp"
#undef VIBE_CHECK
#include "vibe/common/test.hpp"

#include <cmath>
#include <string>

namespace {

using vibe::ConfigError;
using vibe::config::DaConfig;
using vibe::config::ModelConfig;
using vibe::config::VerifyConfig;
using vibe::config::YamlNode;
using vibe::config::merge;

}  // namespace

// ---------------------------------------------------------------------------
// 1. 嵌套映射
// ---------------------------------------------------------------------------
VIBE_TEST(parse_nested_mapping) {
  const YamlNode root = YamlNode::parse(
      "domain:\n"
      "  nx: 64\n"
      "  ny: 32\n"
      "  sub:\n"
      "    a: 1\n"
      "    b: two\n");
  VIBE_CHECK(root.is_mapping());
  VIBE_CHECK(root.has("domain"));
  const YamlNode& d = root["domain"];
  VIBE_CHECK(d.is_mapping());
  VIBE_CHECK(d["nx"].as_int() == 64);
  VIBE_CHECK(d["ny"].as_int() == 32);
  VIBE_CHECK(d["sub"]["a"].as_int() == 1);
  VIBE_CHECK(d["sub"]["b"].as_string() == "two");
  VIBE_CHECK(root["missing"].is_null());
}

// ---------------------------------------------------------------------------
// 2. 缩进序列
// ---------------------------------------------------------------------------
VIBE_TEST(parse_sequence) {
  const YamlNode root = YamlNode::parse(
      "levels:\n"
      "  - 1\n"
      "  - 2\n"
      "  - 3\n"
      "empty:\n");
  const YamlNode& lv = root["levels"];
  VIBE_CHECK(lv.is_sequence());
  VIBE_CHECK(lv.size() == 3);
  VIBE_CHECK(lv[0].as_int() == 1);
  VIBE_CHECK(lv[2].as_int() == 3);
  VIBE_CHECK(root["empty"].is_null());
}

// ---------------------------------------------------------------------------
// 3. 序列中的映射项
// ---------------------------------------------------------------------------
VIBE_TEST(parse_sequence_of_mappings) {
  const YamlNode root = YamlNode::parse(
      "nests:\n"
      "  - name: d02\n"
      "    ratio: 3\n"
      "  - name: d03\n"
      "    ratio: 3\n"
      "tail: 9\n");
  const YamlNode& nests = root["nests"];
  VIBE_CHECK(nests.is_sequence());
  VIBE_CHECK(nests.size() == 2);
  VIBE_CHECK(nests[0]["name"].as_string() == "d02");
  VIBE_CHECK(nests[0]["ratio"].as_int() == 3);
  VIBE_CHECK(nests[1]["name"].as_string() == "d03");
  VIBE_CHECK(root["tail"].as_int() == 9);
}

// ---------------------------------------------------------------------------
// 4. 行内列表与行内映射
// ---------------------------------------------------------------------------
VIBE_TEST(parse_inline_flow) {
  const YamlNode root = YamlNode::parse(
      "zeta: [0.0, 0.25, 0.6, 1.0]\n"
      "opt: {a: 1, b: two}\n"
      "nested: [[1, 2], [3, 4]]\n");
  const std::vector<vibe::Real> z = root["zeta"].as_real_vector();
  VIBE_CHECK(z.size() == 4);
  VIBE_CHECK_NEAR(z[1], 0.25, 0.0);
  VIBE_CHECK(root["opt"]["a"].as_int() == 1);
  VIBE_CHECK(root["opt"]["b"].as_string() == "two");
  VIBE_CHECK(root["nested"].size() == 2);
  VIBE_CHECK(root["nested"][1][0].as_int() == 3);
}

// ---------------------------------------------------------------------------
// 5. 引号字符串
// ---------------------------------------------------------------------------
VIBE_TEST(parse_quoted_strings) {
  const YamlNode root = YamlNode::parse(
      "s1: \"hello world\"\n"
      "s2: 'it''s ok'\n"
      "s3: \"tab\\there\"\n"
      "quoted_key: 1\n");
  VIBE_CHECK(root["s1"].as_string() == "hello world");
  VIBE_CHECK(root["s2"].as_string() == "it's ok");
  VIBE_CHECK(root["s3"].as_string() == "tab\there");
  VIBE_CHECK(root["quoted_key"].as_int() == 1);
}

// ---------------------------------------------------------------------------
// 6. 注释
// ---------------------------------------------------------------------------
VIBE_TEST(parse_comments) {
  const YamlNode root = YamlNode::parse(
      "# 顶部注释\n"
      "a: 1  # 行尾注释\n"
      "b: \"has # inside\"\n"
      "c: 3#not-a-comment\n"
      "\n"
      "   # 纯注释缩进行\n");
  VIBE_CHECK(root["a"].as_int() == 1);
  VIBE_CHECK(root["b"].as_string() == "has # inside");
  VIBE_CHECK(root["c"].as_string() == "3#not-a-comment");
  VIBE_CHECK(root.size() == 3);
}

// ---------------------------------------------------------------------------
// 7. 科学计数法
// ---------------------------------------------------------------------------
VIBE_TEST(parse_scientific_notation) {
  const YamlNode root = YamlNode::parse(
      "a: 1e3\n"
      "b: 2.5E-3\n"
      "c: -1.5e+2\n"
      "d: +.5\n");
  VIBE_CHECK_NEAR(root["a"].as_real(), 1000.0, 0.0);
  VIBE_CHECK_NEAR(root["b"].as_real(), 0.0025, 1e-15);
  VIBE_CHECK_NEAR(root["c"].as_real(), -150.0, 0.0);
  VIBE_CHECK_NEAR(root["d"].as_real(), 0.5, 0.0);
}

// ---------------------------------------------------------------------------
// 8. 布尔变体
// ---------------------------------------------------------------------------
VIBE_TEST(parse_bool_variants) {
  const YamlNode root = YamlNode::parse(
      "t1: true\nt2: yes\nt3: on\nt4: 1\n"
      "f1: false\nf2: no\nf3: off\nf4: 0\n");
  VIBE_CHECK(root["t1"].as_bool() == true);
  VIBE_CHECK(root["t2"].as_bool() == true);
  VIBE_CHECK(root["t3"].as_bool() == true);
  VIBE_CHECK(root["t4"].as_bool() == true);
  VIBE_CHECK(root["f1"].as_bool(true) == false);
  VIBE_CHECK(root["f2"].as_bool(true) == false);
  VIBE_CHECK(root["f3"].as_bool(true) == false);
  VIBE_CHECK(root["f4"].as_bool(true) == false);
  VIBE_CHECK(root["t1"].as_bool(false) == true);
}

// ---------------------------------------------------------------------------
// 9. 空值
// ---------------------------------------------------------------------------
VIBE_TEST(parse_null_values) {
  const YamlNode root = YamlNode::parse(
      "a: ~\n"
      "b: null\n"
      "c:\n"
      "d: 5\n");
  VIBE_CHECK(root["a"].is_null());
  VIBE_CHECK(root["b"].is_null());
  VIBE_CHECK(root["c"].is_null());
  VIBE_CHECK(root["d"].as_int() == 5);
  VIBE_CHECK(root["d"].as_real(7.0) == 5.0);
  VIBE_CHECK(root["a"].as_real(7.0) == 7.0);
}

// ---------------------------------------------------------------------------
// 10. 错误消息带行号与上下文
// ---------------------------------------------------------------------------
VIBE_TEST(parse_error_reports_line) {
  std::string message;
  bool threw = false;
  try {
    (void)YamlNode::parse("a: 1\n  b: 2\n", "t.yaml");
  } catch (const ConfigError& e) {
    threw = true;
    message = e.what();
  }
  VIBE_CHECK(threw);
  VIBE_CHECK(message.find("t.yaml:2") != std::string::npos);

  std::string tab_message;
  bool tab_threw = false;
  try {
    (void)YamlNode::parse("a: 1\n\tb: 2\n", "tab.yaml");
  } catch (const ConfigError& e) {
    tab_threw = true;
    tab_message = e.what();
  }
  VIBE_CHECK(tab_threw);
  VIBE_CHECK(tab_message.find("制表符") != std::string::npos);
  VIBE_CHECK(tab_message.find("tab.yaml:2") != std::string::npos);
}

// ---------------------------------------------------------------------------
// 11. merge 深合并
// ---------------------------------------------------------------------------
VIBE_TEST(merge_deep_override) {
  const YamlNode base = YamlNode::parse(
      "a: 1\n"
      "b: {x: 1, y: 2}\n"
      "c: [1, 2]\n");
  const YamlNode over = YamlNode::parse(
      "b: {y: 9, z: 3}\n"
      "c: [7]\n"
      "d: 4\n");
  const YamlNode m = merge(base, over);
  VIBE_CHECK(m["a"].as_int() == 1);
  VIBE_CHECK(m["b"]["x"].as_int() == 1);
  VIBE_CHECK(m["b"]["y"].as_int() == 9);
  VIBE_CHECK(m["b"]["z"].as_int() == 3);
  VIBE_CHECK(m["c"].is_sequence());
  VIBE_CHECK(m["c"].size() == 1);
  VIBE_CHECK(m["c"][0].as_int() == 7);
  VIBE_CHECK(m["d"].as_int() == 4);

  // Null 视为"本层未指定"：不覆盖下层
  const YamlNode m2 = merge(base, YamlNode::parse("a:\n"));
  VIBE_CHECK(m2["a"].as_int() == 1);
}

// ---------------------------------------------------------------------------
// 12. as_real 的下划线分隔与单位后缀
// ---------------------------------------------------------------------------
VIBE_TEST(as_real_underscore_and_units) {
  const YamlNode root = YamlNode::parse(
      "a: 1_000\n"
      "b: 1e3\n"
      "c: 3km\n"
      "d: 2.5 h\n"
      "e: 1000\n"
      "bad: abc\n");
  VIBE_CHECK_NEAR(root["a"].as_real(), 1000.0, 0.0);
  VIBE_CHECK_NEAR(root["b"].as_real(), 1000.0, 0.0);
  VIBE_CHECK_NEAR(root["c"].as_real(), 3000.0, 0.0);
  VIBE_CHECK_NEAR(root["d"].as_real(), 9000.0, 0.0);
  VIBE_CHECK_NEAR(root["e"].as_real(), 1000.0, 0.0);
  VIBE_CHECK(root["a"].as_int() == 1000);
  VIBE_CHECK_NEAR(root["bad"].as_real(-1.0), -1.0, 0.0);
}

// ---------------------------------------------------------------------------
// 13. ModelConfig 完整示例
// ---------------------------------------------------------------------------
VIBE_TEST(model_config_full_example) {
  const char* yaml = R"YAML(name: demo
description: unit test
domain:
  nx: 64
  ny: 48
  nz: 4
  dx: 3000
  dy: 3000
  z_top: 20000
  zeta: [0.0, 0.1, 0.35, 0.7, 1.0]
  zeta_first_thickness: 50
time:
  dt: 5
  run_length: 3600
  acoustic_substeps: 6
  integrator: rk3_acoustic
  cfl_target: 0.8
numerics:
  advection: weno5
  weno: true
  divergence_damping: 0.05
  helmholtz_solver: krylov_multigrid
nesting:
  enabled: true
  levels: 1
  ratio: 3
  two_way: true
physics:
  microphysics: kessler
  radiation: rrtmg_simple
  pbl: ysu
  surface: monin_obukhov
  cumulus: none
parallel:
  px: 2
  py: 1
)YAML";
  const ModelConfig cfg = ModelConfig::from_yaml(YamlNode::parse(yaml, "demo.yaml"));
  VIBE_CHECK(cfg.name == "demo");
  VIBE_CHECK(cfg.domain.nx == 64);
  VIBE_CHECK(cfg.domain.ny == 48);
  VIBE_CHECK(cfg.domain.nz == 4);
  VIBE_CHECK_NEAR(cfg.domain.dx, 3000.0, 0.0);
  VIBE_CHECK(cfg.domain.zeta.size() == 5);
  VIBE_CHECK_NEAR(cfg.time.dt, 5.0, 0.0);
  VIBE_CHECK(cfg.time.acoustic_substeps == 6);
  VIBE_CHECK(cfg.numerics.advection == "weno5");
  VIBE_CHECK(cfg.numerics.weno);
  VIBE_CHECK(cfg.nesting.enabled);
  VIBE_CHECK(cfg.nesting.ratio == 3);
  VIBE_CHECK(cfg.physics.microphysics == "kessler");
  VIBE_CHECK(cfg.parallel.px == 2);

  cfg.validate();  // 不应抛出
  VIBE_CHECK(cfg.suggested_dt() > 0.0);
  const std::string summary = cfg.describe();
  VIBE_CHECK(summary.find("demo") != std::string::npos);
  VIBE_CHECK(summary.find("fingerprint") != std::string::npos);

  // 默认配置（空节点）也应通过校验
  const ModelConfig def = ModelConfig::from_yaml(YamlNode{});
  def.validate();
  VIBE_CHECK(def.domain.nx == 128);
}

// ---------------------------------------------------------------------------
// 14. validate 捕获非法 zeta
// ---------------------------------------------------------------------------
VIBE_TEST(validate_rejects_bad_zeta) {
  const char* non_monotonic =
      "domain:\n  nx: 8\n  ny: 8\n  nz: 3\n  dx: 100\n  dy: 100\n"
      "  zeta: [0.0, 0.5, 0.4, 1.0]\n";
  VIBE_CHECK_THROWS(
      ModelConfig::from_yaml(YamlNode::parse(non_monotonic)).validate());

  const char* out_of_range =
      "domain:\n  nx: 8\n  ny: 8\n  nz: 3\n  dx: 100\n  dy: 100\n"
      "  zeta: [0.0, 0.5, 1.2, 1.5]\n";
  VIBE_CHECK_THROWS(
      ModelConfig::from_yaml(YamlNode::parse(out_of_range)).validate());

  const char* wrong_length =
      "domain:\n  nx: 8\n  ny: 8\n  nz: 4\n  dx: 100\n  dy: 100\n"
      "  zeta: [0.0, 0.5, 1.0]\n";
  VIBE_CHECK_THROWS(
      ModelConfig::from_yaml(YamlNode::parse(wrong_length)).validate());

  const char* bad_dx =
      "domain:\n  nx: 8\n  ny: 8\n  nz: 3\n  dx: 0\n  dy: 100\n";
  VIBE_CHECK_THROWS(ModelConfig::from_yaml(YamlNode::parse(bad_dx)).validate());

  const char* bad_scheme =
      "domain:\n  nx: 8\n  ny: 8\n  nz: 3\n  dx: 100\n  dy: 100\n"
      "physics:\n  microphysics: not_a_scheme\n";
  VIBE_CHECK_THROWS(ModelConfig::from_yaml(YamlNode::parse(bad_scheme)).validate());

  const char* nest_on_without_enabled =
      "domain:\n  nx: 8\n  ny: 8\n  nz: 3\n  dx: 100\n  dy: 100\n"
      "nesting:\n  levels: 2\n  ratio: 3\n";
  VIBE_CHECK_THROWS(
      ModelConfig::from_yaml(YamlNode::parse(nest_on_without_enabled)).validate());
}

// ---------------------------------------------------------------------------
// 15. 必填字段缺失
// ---------------------------------------------------------------------------
VIBE_TEST(validate_requires_mandatory_fields) {
  std::string message;
  bool threw = false;
  try {
    (void)ModelConfig::from_yaml(
        YamlNode::parse("domain:\n  nx: 8\n  ny: 8\n  nz: 4\n  dx: 100\n"));
  } catch (const ConfigError& e) {
    threw = true;
    message = e.what();
  }
  VIBE_CHECK(threw);
  VIBE_CHECK(message.find("domain.dy") != std::string::npos);

  bool time_threw = false;
  try {
    (void)ModelConfig::from_yaml(YamlNode::parse("time:\n  run_length: 600\n"));
  } catch (const ConfigError& e) {
    time_threw = true;
    message = e.what();
  }
  VIBE_CHECK(time_threw);
  VIBE_CHECK(message.find("time.dt") != std::string::npos);

  // 段落缺失时全部取默认值，不报错
  const ModelConfig c = ModelConfig::from_yaml(YamlNode::parse("name: only_name\n"));
  VIBE_CHECK(c.name == "only_name");
  VIBE_CHECK(c.domain.nx == 128);
}

// ---------------------------------------------------------------------------
// 16. suggested_dt 数值合理性
// ---------------------------------------------------------------------------
VIBE_TEST(suggested_dt_is_reasonable) {
  ModelConfig cfg;
  cfg.domain.dx = 3000;
  cfg.domain.dy = 3000;
  cfg.domain.zeta.clear();                 // 自动 zeta
  cfg.domain.zeta_first_thickness = 100;   // 最薄层 100 m
  cfg.time.cfl_target = 0.8;
  cfg.time.acoustic_substeps = 6;

  const vibe::Real dt = cfg.suggested_dt();
  const vibe::Real dt_horizontal = 0.8 * 3000.0 / (100.0 + 350.0);
  const vibe::Real dt_acoustic_total = 6.0 * 0.8 * 100.0 / 350.0;
  VIBE_CHECK(std::isfinite(static_cast<double>(dt)));
  VIBE_CHECK(dt > 0.0);
  VIBE_CHECK(dt <= dt_horizontal + 1e-12);
  VIBE_CHECK_NEAR(dt, dt_acoustic_total, 1e-9);

  // 垂直层足够厚时，由水平 CFL 限制
  cfg.domain.zeta_first_thickness = 5000;
  const vibe::Real dt_h = cfg.suggested_dt();
  VIBE_CHECK_NEAR(dt_h, dt_horizontal, 1e-9);
}

// ---------------------------------------------------------------------------
// 17. dump 往返
// ---------------------------------------------------------------------------
VIBE_TEST(dump_roundtrip) {
  const YamlNode n = YamlNode::parse(
      "a: 1\n"
      "b: [1, 2, 3]\n"
      "c:\n"
      "  d: hello\n"
      "  e: 2.5\n");
  const std::string text = n.dump();
  VIBE_CHECK(text.find("  d: hello") != std::string::npos);
  const YamlNode m = YamlNode::parse(text);
  VIBE_CHECK(m["a"].as_int() == 1);
  VIBE_CHECK(m["b"].size() == 3);
  VIBE_CHECK(m["c"]["d"].as_string() == "hello");
  VIBE_CHECK_NEAR(m["c"]["e"].as_real(), 2.5, 0.0);

  // 路径查询
  VIBE_CHECK(n.get("c.d").as_string() == "hello");
  VIBE_CHECK(n.get("c.missing").is_null());
  VIBE_CHECK(n.get("b.1").as_int() == 2);
}

// ---------------------------------------------------------------------------
// 18. DaConfig 与 VerifyConfig
// ---------------------------------------------------------------------------
VIBE_TEST(da_and_verify_config) {
  const char* yaml = R"YAML(enabled: true
window:
  length: 3600
  slot_interval: 12
outer_loops: 2
inner_loops: 40
background:
  method: nmc
  balance: linear_balance
  horizontal_length_scale: 200000
  vertical_length_scale: 3000
minimizer:
  method: lbfgs
  max_iterations: 30
)YAML";
  const DaConfig da = DaConfig::from_yaml(YamlNode::parse(yaml));
  VIBE_CHECK(da.enabled);
  VIBE_CHECK(da.window.slot_interval == 12);
  VIBE_CHECK_NEAR(da.window.length, 3600.0, 0.0);
  VIBE_CHECK(da.outer_loops == 2);
  VIBE_CHECK(da.inner_loops == 40);
  VIBE_CHECK(da.background.method == "nmc");
  VIBE_CHECK(da.minimizer.method == "lbfgs");
  VIBE_CHECK_NEAR(da.background.horizontal_length_scale, 200000.0, 0.0);
  VIBE_CHECK_NEAR(da.background.vertical_length_scale, 3000.0, 0.0);
  da.validate();

  DaConfig bad = da;
  bad.inner_loops = 0;
  VIBE_CHECK_THROWS(bad.validate());
  bad = da;
  bad.background.horizontal_length_scale = 0;
  VIBE_CHECK_THROWS(bad.validate());

  const VerifyConfig vc = VerifyConfig::from_yaml(YamlNode::parse(
      "variables: [t2, u10]\n"
      "thresholds: [0.1, 1.0]\n"
      "neighborhood_radii: [1, 3]\n"
      "methods: [rmse, fss]\n"
      "matchup: bilinear\n"));
  VIBE_CHECK(vc.variables.size() == 2);
  VIBE_CHECK(vc.variables[0] == "t2");
  VIBE_CHECK(vc.thresholds.size() == 2);
  VIBE_CHECK(vc.neighborhood_radii.size() == 2);
  VIBE_CHECK(vc.neighborhood_radii[1] == 3);
  VIBE_CHECK(vc.methods.size() == 2);
  VIBE_CHECK(vc.matchup == "bilinear");
}
