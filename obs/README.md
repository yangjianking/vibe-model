# obs/ —— 示例观测

本目录只包含**合成示例观测**，用于演示 4D-Var 的输入格式与跑通流程，
不含任何真实观测资料（真实资料请遵守相应数据提供方的许可）。

## 文件

| 文件 | 内容 | 条数 |
|---|---|---|
| example_obs.csv | 6 个探空站 x 8 层 x 4 变量（u, v, t, q）+ 10 个地面站 | 见文件 |

## CSV 格式

表头固定为：

~~~text
type,time,x,y,z,var,channel,value,sigma,qc
~~~

| 列 | 含义 |
|---|---|
| type | radiosonde / surface / aircraft / amv / scatterometer / gnssro / radiance / radar |
| time | 相对同化窗口起点的秒数 |
| x, y, z | 模式物理坐标（米） |
| var | u / v / w / t / q / ps / p / radiance / refractivity / reflectivity |
| channel | 卫星通道号（非辐射率为 -1） |
| value | 观测值（国际单位：m/s, K, kg/kg, Pa） |
| sigma | 观测误差标准差（同单位） |
| qc | 质量控制标志，0 = 可用 |

读取接口见 [observations.hpp](../include/vibe/obs/observations.hpp) 的
«ObsReader::read_csv»，返回 «ObsSpace»。

## 生成合成观测（OSE 试验）

更严格的做法是用模式真值通过观测算子采样并加入指定统计特性的误差：

~~~cpp
obs::SyntheticObsOptions opt;
opt.types = {obs::ObsType::Radiosonde, obs::ObsType::Surface};
opt.density_fraction = 0.01;
opt.add_noise = true;
opt.seed = 42;
auto obs = obs::generate_synthetic_observations(*op, view, opt);
obs::write_csv("obs/synthetic_obs.csv", obs);
~~~

这样得到的观测对 4D-Var 是"自洽"的（观测算子无代表性误差），可用于验证
极小化与伴随代码的正确性；真实资料则需要额外的代表性误差与偏差订正处理。
