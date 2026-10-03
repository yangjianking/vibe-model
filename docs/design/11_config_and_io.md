# 11 配置解析与 IO 设计

> 本文对应实现：src/config/mini_yaml.cpp、src/config/config.cpp、src/io/field_io.cpp、
> src/io/binary_writer.cpp、src/io/netcdf_writer.cpp。
> 接口基线见 [00_architecture.md](00_architecture.md) 第 10 节；公式编号见
> [references.md](references.md)。

---

## 1. 配置分层与叠加语义

### 1.1 分层

| 层 | 文件 | 内容 |
|---|---|---|
| 1 | config/model.yaml | 网格、时间步、数值格式、物理、并行 |
| 2 | config/nests.yaml | 嵌套层级与变分辨率设置（可选） |
| 3 | config/da_4dvar.yaml | 同化窗口、B 矩阵、极小化、观测选择 |
| 4 | config/verify.yaml | 检验变量、阈值、方法、邻域半径 |

C++ 侧对应 vibe::config::ModelConfig / DaConfig / VerifyConfig。
每个字段都有结构体默认值；只有"必填项"缺失才报错（见 1.4）。

### 1.2 叠加流程

ModelConfig::load(paths, allow_missing)：

1. 以**空节点**（Null）为基；
2. 依次读入各层文件并 merge(acc, layer)；
3. 对合并结果调用一次 ModelConfig::from_yaml；
4. 最后调用 validate()（错误消息带字段路径）。

allow_missing = true 时缺失的文件只记 VIBE_WARN 日志并跳过；
false 时抛 ConfigError("配置文件不存在: ...")。

### 1.3 merge 的语义

YamlNode merge(const YamlNode& base, const YamlNode& over)：

* 两侧都是映射 → **深合并**，over 中同名子节点递归处理；
* over 为 Null（YAML 里写成 ~ / null / 空值）→ **视为本层未指定**，保留 base；
* 其它情形（序列、标量）→ over **整体替换** base。序列不做元素级合并，
  这样"后来者完整覆盖列表"的直觉成立（例如检验阈值列表）。

~~~text
base:                over:                merge(base, over):
domain:              domain:              domain:
  nx: 128              nx: 256              nx: 256
  ny: 128              zeta: [0, 1]         ny: 128
                                           zeta: [0, 1]
nesting:             nesting:             nesting:
  levels: 1            ratio: 3             levels: 1
                                           ratio: 3
~~~

注意：因为缺失的键不会出现在映射里，over 中出现的 Null 只可能来自显式的
空值写法；把它当作"未指定"而不是"删除"，可让 config/nests.yaml 这类
局部覆盖文件安全地覆盖基础配置。

### 1.4 必填项

* 根节点必须是映射，否则 ConfigError；
* 当 **domain（或 grid）段出现**时，nx, ny, nz, dx, dy 必须存在，
  否则抛 ConfigError("缺少必填字段: domain.<name>")；
* 当 **time（或 timestep）段出现**时，dt 必须存在；
* 段整体缺失时全部取默认值（因此只用 nesting 段覆盖的层是正确的）；
* 未知键被忽略（不报错），以便配置向前兼容。

---

## 2. YAML 子集语法（EBNF）

实现为手写词法 + LL(1) 递归下降，缩进即文法层级；整体 **O(N)**（N 为字符数），
空间 O(嵌套深度)。

~~~ebnf
document      = { line } ;
value         = scalar | flow-seq | flow-map | block-map | block-seq ;

block-map     = { MAP_ENTRY } ;
MAP_ENTRY     = KEY ":" [ inline-value ] NEWLINE
                [ block-value ] ;
block-seq     = { SEQ_ENTRY } ;
SEQ_ENTRY     = "-" [ inline-value ] NEWLINE
                [ block-value ] ;
block-value   = block-map | block-seq | scalar ;

inline-value  = scalar | flow-seq | flow-map ;
flow-seq      = "[" [ scalar { "," scalar } ] "]" ;
flow-map      = "{" [ scalar ":" scalar { "," scalar ":" scalar } ] "}" ;

scalar        = quoted | plain ;
quoted        = '"' { char | escape } '"'
              | "'" { char | "''" } "'" ;
plain         = null | bool | number | string ;
null          = "~" | "null" | "" ;
bool          = "true" | "false" | "yes" | "no" | "on" | "off"
              | "1" | "0" ;
number        = [ "+" | "-" ] ( digits [ "." digits ] | "." digits )
                [ ( "e" | "E" ) [ "+" | "-" ] digits ]
                { "_" digits } ;
string        = { any char except comment start, ":", "[", "{", "}", "]" } ;
comment       = "#" { char } ;      (* 仅在行首或前导空白之后生效 *)
~~~

约定：

* **缩进只能用空格**。缩进中出现制表符立即抛 ConfigError，消息含行号与出错行。
* 键值分隔的冒号必须在顶层（不在引号与 []{} 内），且**后面是空白或行尾**，
  因此 url: http://x 与 a:b 都不会被误判。
* 数字按下标量文本保存，由 as_real / as_int / as_bool 解释，因此
  1_000、1e3、3km、2.5 h 都能取到数值（单位表与 Python 侧一致）。
* dump() 以 2 空格缩进输出可读 YAML；为保证往返语义，数字/布尔/null 词形
  以及含特殊字符的字符串会被加双引号。

---

## 3. .vibebin 格式规范

零依赖后备格式，小端字节序，**写时按字节拼装、读时按字节还原**，不依赖主机端序。
串行运行下 nx/ny/nz 即全局维度；MPI 下每个 rank 写自己的子域文件。

### 3.1 文件头（40 字节）

| 偏移 | 长度 | 类型 | 字段 | 说明 |
|---:|---:|---|---|---|
| 0 | 8 | char[8] | magic | 固定 "VIBEBIN1"，无终止 NUL |
| 8 | 4 | int32 LE | nx | 体心网格本地/全局 x 点数 |
| 12 | 4 | int32 LE | ny | y 点数 |
| 16 | 4 | int32 LE | nz | z 点数 |
| 20 | 4 | int32 LE | real_kind | 0 = 写者 Real 为 float64；1 = float32 |
| 24 | 8 | float64 LE | time | 记录时刻（秒） |
| 32 | 4 | int32 LE | nvars | 变量个数 |
| 36 | 4 | int32 LE | nlevels | 垂直层数（= nz，供剖面工具使用） |

### 3.2 变量记录

从偏移 40 开始，重复 nvars 次：

| 偏移 | 长度 | 类型 | 字段 |
|---:|---:|---|---|
| 0 | 32 | char[32] | name，不足补 NUL；过长截断为 31 字符 |
| 32 | 8*n | float64 LE | data，行主序，i 最快、k 最慢 |

线性下标：lin = (k * nj + j) * ni + i。

### 3.3 名字决定形状

文件里只有 32 字节名字能承载形状信息，因此约定**名字唯一决定错位与点数**：

| name | 错位 | (ni, nj, nk) |
|---|---|---|
| u, u_facex | FaceX | (nx+1, ny, nz) |
| v, v_facey | FaceY | (nx, ny+1, nz) |
| w, w_facez, zeta | FaceZ | (nx, ny, nz+1) |
| __restart_info__ | — | (2, 1, 1)，存 [time, step] |
| 其它（rho/theta/pi/qv/qc/qr/qi/qs/qg/x/y/z/terrain/dx_cell/dy_cell/...） | Cell | (nx, ny, nz) |

写者会校验 FieldMeta.stagger 与名字隐含的错位一致；不一致时抛 IoError，
避免静默写坏数据。real_kind 只记录写者原生 Real 是 float64 还是 float32，
**数据区永远是 float64**，读时再转回 Real（Python 端因此只有一种解码路径）。

### 3.4 Python numpy 读取示例

~~~python
import numpy as np

MAGIC = b"VIBEBIN1"
FACE = {"u": "x", "u_facex": "x",
        "v": "y", "v_facey": "y",
        "w": "z", "w_facez": "z", "zeta": "z"}


def read_vibebin(path):
    """读取 .vibebin，返回 {变量名: ndarray(k, j, i)} 以及 header 字典。

    复杂度 O(文件大小)；逐变量 np.frombuffer 是零拷贝视图，底层 bytes 由
    返回的数组引用而保持存活；超大文件可改用 np.memmap（见文末说明）。
    """
    with open(path, "rb") as fh:
        magic = fh.read(8)
        if magic != MAGIC:
            raise ValueError("bad magic: " + repr(magic))
        nx, ny, nz, real_kind = np.frombuffer(fh.read(16), dtype="<i4")
        (time,) = np.frombuffer(fh.read(8), dtype="<f8")
        nvars, nlevels = np.frombuffer(fh.read(8), dtype="<i4")

        header = dict(nx=int(nx), ny=int(ny), nz=int(nz),
                      real_kind=int(real_kind), time=float(time),
                      nlevels=int(nlevels), nvars=int(nvars))
        out = {"__header__": header}
        for _ in range(int(nvars)):
            name = fh.read(32).split(b"\0", 1)[0].decode("ascii")
            if name == "__restart_info__":
                shape = (1, 1, 2)
            else:
                s = FACE.get(name, "c")
                shape = (int(nz) + (s == "z"),
                         int(ny) + (s == "y"),
                         int(nx) + (s == "x"))
            count = int(np.prod(shape))
            data = np.frombuffer(fh.read(8 * count), dtype="<f8")
            out[name] = data.reshape(shape).copy()   # (k, j, i)
        return out
~~~

大文件可用 np.memmap(path, dtype="<f8", mode="r") 先读头再按偏移切片，
偏移规则同上（头 40 字节 + 每个变量 32 字节名 + 8*n 数据）。

---

## 4. NetCDF 变量与 CF 属性

仅当 VIBE_HAVE_NETCDF 定义时编译（CMake 的 VIBE_ENABLE_NETCDF）。
文件用 netCDF-4 C API 创建，维度固定：

| 维度 | 长度 | 说明 |
|---|---|---|
| time | unlimited | 记录维 |
| zeta | nz+1 | 地形追随层界面坐标 |
| z | nz | 层中心高度 |
| y | ny | |
| x | nx | |

变量：各预报量共享 (time, zeta, y, x)；坐标变量 x/y/z/zeta/terrain 以及
可选 dx_cell/dy_cell 按 1-D/2-D 写出。

| 变量 | 维度 | 属性（除 _FillValue = -9999） |
|---|---|---|
| time | (time) | units="seconds since 1970-01-01 00:00:00 UTC"，calendar="standard"，axis="T" |
| x | (x) | units="m"，long_name="x coordinate of cell centres" |
| y | (y) | units="m"，long_name="y coordinate of cell centres" |
| zeta | (zeta) | units="1"，long_name="terrain-following vertical coordinate" |
| z | (z) 或 (z,y,x) | units="m"，long_name="height of cell centres"（有地形时三维） |
| terrain | (y,x) | units="m"，long_name="terrain height" |
| u,v,w | (time,zeta,y,x) | units="m s-1"，stagger="face_x/face_y/face_z"，coordinates="x y zeta" |
| rho | (time,zeta,y,x) | units="kg m-3" |
| theta | (time,zeta,y,x) | units="K" |
| pi | (time,zeta,y,x) | units="1" |
| qv,qc,qr,qi,qs,qg | (time,zeta,y,x) | units="kg kg-1" |

全局属性：title、source、history、Conventions="CF-1.10"、grid_type="arakawa_c"、
vertical_coordinate="terrain_following_height"、refinement_ratio、model_version。

取舍：为使所有量共享同一组 CF 维度，面量按体心点数写出（丢掉最东/最北/最顶的面点），
原始错位记录在 stagger 属性中。因此 **NetCDF 面向后处理，不作为精确重启格式**；
重启一律用 .vibebin，它保留面点的完整信息。

---

## 5. 重启与检查点策略

### 5.1 文件内容

write_restart(path, s, info) 写一个 .vibebin：

1. 变量 __restart_info__，形状 (2,1,1)，内容 [time, step]；
2. 按 dyn::Species 顺序写出全部 12 个预报量（变量名 = dyn::species_name，
   单位/错位取自 src/io/field_io.cpp 的静态表）。

read_restart(path, s) 先查 __restart_info__（缺失则 time/step 置 0），
再填充已分配的 State，最后回写 s.time / s.step 并返回 RestartInfo。

### 5.2 周期与一致性

* time.restart_interval 控制检查点间隔（0 = 不写），由 driver 负责调度；
* RestartInfo::config_hash 用于防止"用错配置重启"：应与 ModelConfig 的
  配置指纹（第 6 节）比较，不一致时拒绝重启；
* RestartInfo::valid_time / config_hash 不存于 .vibebin（二进制格式只放数值），
  需要时由驱动在旁路元数据文件或日志中记录；未来 NetCDF 重启可携带属性；
* 读入后 halo 区未定义，driver 必须在第一次求趋势前调用 HaloExchange。

### 5.3 文件命名建议

~~~text
<output_dir>/<name>/restart_<step:08d>.vibebin
~~~

---

## 6. 配置指纹

目的：给出一个短的、确定性的配置摘要，用于
① 重启兼容性检查（RestartInfo::config_hash）；② 实验可复现性记录。

算法（src/config/config.cpp，命名空间 vibe::config::detail 的 config_fingerprint）：

1. 把 ModelConfig 的全部数值/字符串字段按固定顺序格式化为规范字符串
   （浮点用 17 位有效数字，避免舍入歧义）；
2. 对该字符串做 **FNV-1a 64 位**哈希：h = 0xcbf29ce484222325，
   对每个字节 h ^= b; h *= 0x100000001b3；
3. 输出 16 位十六进制小写字符串。

ModelConfig::describe() 末行会打印 fingerprint = <16 hex>。

> 实现偏差：冻结的 include/vibe/config/config.hpp 未声明 config_fingerprint，
> 按照任务约定未修改头文件，而是在 .cpp 中以 vibe::config::detail 命名空间的
> 自由函数提供（有外部链接，可在其它 TU 中前向声明后使用）。建议后续在头文件中
> 正式补一条声明。

---

## 7. 错误处理约定

| 情形 | 异常 | 消息要点 |
|---|---|---|
| YAML 语法错误 / 制表符缩进 / 引号未闭合 | vibe::ConfigError | origin:行号 + 出错行上下文 |
| 必填字段缺失 | ConfigError | "缺少必填字段: <dotted.path>" |
| 字段类型/取值非法 | ConfigError | 字段路径 + 实际值 + 可选白名单 |
| 物理/数值方案名不在白名单 | ConfigError | 允许值列表 |
| 进程网格与网格点数冲突 | ConfigError | 两者数值 |
| 进程网格不整除网格点数 | **仅日志警告** | 说明 Decomposition 支持余数分配 |
| 无法打开/写入文件 | IoError | 路径 + errno + strerror |
| magic 不匹配 / 头截断 / 数据截断 | IoError | "不是 .vibebin 文件" / "截断" + 路径 |
| 变量不存在 / 目标场维度或错位不匹配 | IoError | 变量名 + 文件与目标的维度 |
| NetCDF 未编译启用 | vibe::NotImplemented | 提示改用二进制写者或打开编译选项 |
| NetCDF 读取请求 | NotImplemented | 提示使用 .vibebin 或 Python 包 vibe_post |

约定的白名单常量（src/config/config.cpp）：

| 字段 | 允许值 |
|---|---|
| time.integrator | rk3_acoustic, semi_implicit, rk3, hevi |
| numerics.advection | central2, central4, central6, weno5, upwind3 |
| numerics.helmholtz_solver | vertical_tridiagonal, krylov_multigrid, krylov_jacobi |
| numerics.coriolis | none, fplane, betaplane |
| domain.refinement | none, circular, channel |
| physics.microphysics | none, kessler, lin, thompson, morrison |
| physics.radiation | none, rrtm, rrtmg, rrtmg_simple |
| physics.pbl | none, ysu, myj, mye, smagorinsky |
| physics.surface | none, monin_obukhov, noah, noilhan_planton |
| physics.cumulus | none, kain_fritsch, grell, tiedke, tiedtke |
| parallel.io_backend | netcdf, binary, both |
| da.background.method | nmc, ensemble, hybrid |
| da.background.balance | none, linear_balance, omega |
| da.minimizer.method | lbfgs, cg, lanczos |

校验中的数值判据（详见 ModelConfig::validate 与 ModelConfig::suggested_dt）：

* nx, ny, nz > 0；dx, dy > 0；z_top > 0；
* zeta 长度 = nz+1、严格单调递增、落在 [0,1]（[D2] Gal-Chen & Somerville）；
* dt > 0；acoustic_substeps >= 1；cfl_target 在 (0, 1.5]；
* nesting.ratio >= 2、levels >= 1 时 enabled 必须为 true；enabled 时 levels >= 1、ratio >= 2；
* px >= 1, py >= 1 且 px*py <= nx*ny；
* suggested_dt = min( CFL*min(dx,dy)/(U_max+c_s), n_sub*CFL*dz_min/c_s )，
  U_max = 100 m/s，c_s = 350 m/s（[T5][D6]）。
