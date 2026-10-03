# tests/data/ —— 测试数据

本目录**默认不包含任何大数据文件**，原因有二：

1. 仓库需要保持轻量，且避免再分发许可受限的数据；
2. 所有单元测试都应当能用合成数据自洽地验证（见
   [docs/design/12_build_and_test.md](../../docs/design/12_build_and_test.md) 的测试方法论）。

## 需要外部数据时

若某个测试需要真实资料，请把数据放在本目录下并遵守以下约定：

| 文件名 | 内容 | 格式 | 生成方式 |
|---|---|---|---|
| terrain_*.txt | 地形矩阵 | 纯文本，行主序 | scripts/make_terrain.py |
| sounding_*.txt | 探空廓线（z theta qv u v） | 纯文本列 | 手工或再分析插值 |
| obs_*.csv | 合成观测 | CSV（见 obs::ObsReader） | obs::generate_synthetic_observations |
| nc_*.nc | NetCDF 场 | CF 约定 | 模式自身输出 |

## 生成示例地形

~~~bash
python scripts/make_terrain.py --nx 128 --ny 64 --dx 2000 \
    --type agnesi --height 1000 --radius 6000 --out tests/data/terrain_agnesi.txt
~~~

随后在配置中引用：

~~~yaml
domain:
  terrain: "tests/data/terrain_agnesi.txt"
~~~

## 缺少数据时的行为

所有读取外部数据的代码路径都必须**优雅失败**：给出带路径的明确错误，
或按文档回退到合成数据（例如初始化模块在找不到文件时回退到静止等温大气并给出警告）。
