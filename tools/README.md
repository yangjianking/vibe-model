# tools/ —— 独立小工具

这些工具刻意保持"零外部依赖"，只链接 vibe::model，便于在任何构建配置下使用。

| 工具 | 源文件 | 用途 |
|---|---|---|
| vibe_adjoint_check | [adjoint_check.cpp](adjoint_check.cpp) | 切线性/伴随点积检验（[A3]） |
| vibe_verify_cli | [../src/verify/verify_cli.cpp](../src/verify/verify_cli.cpp) | 读取 .vibebin 并计算检验评分 |

## vibe_adjoint_check

在每次修改切线性或伴随代码之后运行：

~~~bash
./build/gcc-release/src/vibe_adjoint_check --config config/model.yaml \
    --steps 20 --dt 6 --trials 5
~~~

期望输出：

~~~text
试验 1/5  <M dx, dy> = 1.234e+05  <dx, M^T dy> = 1.234e+05  相对误差 = 3.1e-13  通过
...
最差相对误差 = 8.7e-13（阈值 1e-8），失败次数 = 0
~~~

若相对误差远大于 $10^{-10}$，说明伴随实现有误。常见原因见
[docs/design/07_tangent_linear_adjoint.md](../docs/design/07_tangent_linear_adjoint.md) 的错误清单。
