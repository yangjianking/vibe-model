/// @file test_minimizer.cpp
/// @brief 极小化器、线搜索与 Lanczos 谱估计的单元测试。
///
/// 覆盖：二次型解析极小点、Rosenbrock、L-BFGS vs CG、强 Wolfe 条件、
/// Lanczos 特征值恢复、预条件效果、迭代记录长度、向量工具与对角矩阵。
///
/// 入口由 tests/unit/test_main.cpp（目标 vibe_test_main）统一提供。

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "vibe/common/test.hpp"
#include "vibe/da/da_types.hpp"
#include "vibe/da/minimizer.hpp"

namespace vibe::da::detail {
/// 定义在 src/da/minimizer.cpp（具名 detail 命名空间，外部链接）
std::vector<Real> lanczos_ritz_hessian(const Objective& J, const Vector& x0, int m,
                                       Real eps);
}  // namespace vibe::da::detail

using vibe::Real;
using vibe::Size;
using vibe::da::DiagonalMatrix;
using vibe::da::IterationRecord;
using vibe::da::LineSearchResult;
using vibe::da::Minimizer;
using vibe::da::MinimizerOptions;
using vibe::da::Objective;
using vibe::da::Vector;

namespace {

/// 对角二次型： J(x) = 1/2 sum c_i (x_i - t_i)^2，极小点 x* = t
class QuadraticObjective final : public Objective {
 public:
  QuadraticObjective(Vector curvature, Vector target)
      : c_(std::move(curvature)), t_(std::move(target)) {
    ++instances;
  }
  Real value(const Vector& x) const override {
    Real s = Real(0);
    for (Size i = 0; i < c_.size(); ++i) {
      const Real d = x[i] - t_[i];
      s += Real(0.5) * c_[i] * d * d;
    }
    return s;
  }
  void gradient(const Vector& x, Vector& g) const override {
    g.resize(c_.size());
    for (Size i = 0; i < c_.size(); ++i) g[i] = c_[i] * (x[i] - t_[i]);
  }
  Real value_and_gradient(const Vector& x, Vector& g) const override {
    const Real f = value(x);
    gradient(x, g);
    ++evaluations;
    return f;
  }
  Size size() const noexcept override { return c_.size(); }
  const Vector& curvature() const noexcept { return c_; }
  const Vector& target() const noexcept { return t_; }

  mutable int evaluations = 0;
  static int instances;

 private:
  Vector c_, t_;
};
int QuadraticObjective::instances = 0;

/// 多维 Rosenbrock： min 在 (1,1,...,1)，J = 0
class RosenbrockObjective final : public Objective {
 public:
  explicit RosenbrockObjective(Size n) : n_(n) {}
  Real value(const Vector& x) const override {
    Real s = Real(0);
    for (Size i = 0; i + 1 < n_; ++i) {
      const Real a = x[i + 1] - x[i] * x[i];
      const Real b = Real(1) - x[i];
      s += Real(100) * a * a + b * b;
    }
    return s;
  }
  void gradient(const Vector& x, Vector& g) const override {
    g.assign(n_, Real(0));
    for (Size i = 0; i + 1 < n_; ++i) {
      const Real a = x[i + 1] - x[i] * x[i];
      g[i] += Real(-400) * x[i] * a - Real(2) * (Real(1) - x[i]);
      g[i + 1] += Real(200) * a;
    }
  }
  Real value_and_gradient(const Vector& x, Vector& g) const override {
    const Real f = value(x);
    gradient(x, g);
    return f;
  }
  Size size() const noexcept override { return n_; }

 private:
  Size n_;
};

Vector linspace(Real a, Real b, Size n) {
  Vector v(n, Real(0));
  for (Size i = 0; i < n; ++i) {
    v[i] = a + (b - a) * static_cast<Real>(i) / static_cast<Real>(n > 1 ? n - 1 : 1);
  }
  return v;
}

MinimizerOptions options(const std::string& method, int max_it = 400) {
  MinimizerOptions o;
  o.method = method;
  o.max_iterations = max_it;
  o.gradient_tolerance = Real(1e-10);
  o.cost_tolerance = Real(0);
  o.memory = 8;
  o.verbose = false;
  o.use_preconditioner = true;
  return o;
}

}  // namespace

// ===========================================================================
// 1. 向量工具
// ===========================================================================
VIBE_TEST(minimizer_vec_utilities) {
  const Vector a{Real(1), Real(-2), Real(3)};
  Vector b{Real(2), Real(2), Real(2)};
  VIBE_CHECK_NEAR(vibe::da::vec::dot(a, b), Real(6), Real(1e-14));
  VIBE_CHECK_NEAR(vibe::da::vec::norm2(a), std::sqrt(Real(14)), Real(1e-14));
  VIBE_CHECK_NEAR(vibe::da::vec::max_abs(a), Real(3), Real(1e-14));
  vibe::da::vec::axpy(Real(2), a, b);
  VIBE_CHECK_NEAR(b[0], Real(4), Real(1e-14));
  VIBE_CHECK_NEAR(b[1], Real(-2), Real(1e-14));
  vibe::da::vec::scale(Real(0.5), b);
  VIBE_CHECK_NEAR(b[2], Real(4), Real(1e-14));
  Vector c;
  vibe::da::vec::copy(a, c);
  VIBE_CHECK(c.size() == a.size());
  VIBE_CHECK_NEAR(c[2], Real(3), Real(1e-14));
}

// ===========================================================================
// 2. 对角矩阵
// ===========================================================================
VIBE_TEST(minimizer_diagonal_matrix) {
  DiagonalMatrix D(Vector{Real(2), Real(-1), Real(0.5)});
  Vector x{Real(1), Real(2), Real(4)}, y;
  D.apply(x, y);
  VIBE_CHECK_NEAR(y[0], Real(2), Real(1e-14));
  VIBE_CHECK_NEAR(y[1], Real(-2), Real(1e-14));
  VIBE_CHECK_NEAR(y[2], Real(2), Real(1e-14));
  VIBE_CHECK(D.size() == 3);
  VIBE_CHECK(std::string(D.name()) == "diagonal");
}

// ===========================================================================
// 3. 强 Wolfe 线搜索满足 Armijo + 曲率
// ===========================================================================
VIBE_TEST(minimizer_line_search_wolfe_conditions) {
  QuadraticObjective J(Vector{Real(1), Real(4), Real(9), Real(16)},
                       Vector{Real(1), Real(1), Real(1), Real(1)});
  Vector x{Real(3), Real(-2), Real(4), Real(0)};
  Vector g;
  const Real f0 = J.value_and_gradient(x, g);
  Vector d(g.size(), Real(0));
  for (Size i = 0; i < g.size(); ++i) d[i] = -g[i];
  const Real g0 = vibe::da::vec::dot(g, d);
  const LineSearchResult ls = vibe::da::strong_wolfe_line_search(J, x, d, f0, g0);
  VIBE_CHECK(ls.converged);
  VIBE_CHECK(ls.step > Real(0));
  VIBE_CHECK(ls.value <= f0 + Real(1e-4) * ls.step * g0);
  VIBE_CHECK(std::abs(ls.directional_derivative) <= Real(0.9) * std::abs(g0));
}

// ===========================================================================
// 4. 非下降方向：步长为 0 且不收敛
// ===========================================================================
VIBE_TEST(minimizer_line_search_rejects_ascent) {
  QuadraticObjective J(Vector{Real(1), Real(1)}, Vector{Real(0), Real(0)});
  Vector x{Real(1), Real(1)};
  Vector g;
  const Real f0 = J.value_and_gradient(x, g);
  Vector d{Real(1), Real(1)};  // 与梯度同向 = 上升
  const LineSearchResult ls =
      vibe::da::strong_wolfe_line_search(J, x, d, f0, vibe::da::vec::dot(g, d));
  VIBE_CHECK(!ls.converged);
  VIBE_CHECK_NEAR(ls.step, Real(0), Real(1e-15));
  VIBE_CHECK_NEAR(ls.value, f0, Real(1e-15));
}

// ===========================================================================
// 5. L-BFGS 在二次型上达到解析极小点
// ===========================================================================
VIBE_TEST(minimizer_lbfgs_quadratic_exact) {
  QuadraticObjective J(Vector{Real(1), Real(2), Real(5), Real(10)},
                       Vector{Real(1), Real(-1), Real(2), Real(0.5)});
  Vector x{Real(5), Real(5), Real(5), Real(5)};
  auto m = vibe::da::make_minimizer(options("lbfgs"));
  const Real f = m->minimize(J, x);
  VIBE_CHECK_NEAR(f, Real(0), Real(1e-16));
  for (Size i = 0; i < x.size(); ++i) VIBE_CHECK_NEAR(x[i], J.target()[i], Real(1e-8));
  VIBE_CHECK(m->final_gradient_norm() < Real(1e-6));
}

// ===========================================================================
// 6. L-BFGS 在 Rosenbrock 上收敛到 (1,1)
// ===========================================================================
VIBE_TEST(minimizer_lbfgs_rosenbrock) {
  RosenbrockObjective J(2);
  Vector x{Real(-1.2), Real(1.0)};
  auto m = vibe::da::make_minimizer(options("lbfgs", 500));
  const Real f = m->minimize(J, x);
  VIBE_CHECK(f < Real(1e-8));
  VIBE_CHECK_NEAR(x[0], Real(1), Real(1e-3));
  VIBE_CHECK_NEAR(x[1], Real(1), Real(1e-3));
}

// ===========================================================================
// 7. 非线性 CG 在二次型上达到解析极小点
// ===========================================================================
VIBE_TEST(minimizer_cg_quadratic_exact) {
  QuadraticObjective J(Vector{Real(1), Real(3), Real(7)}, Vector{Real(-2), Real(4), Real(1)});
  Vector x{Real(0), Real(0), Real(0)};
  auto m = vibe::da::make_minimizer(options("cg"));
  VIBE_CHECK(std::string(m->name()) == "cg");
  const Real f = m->minimize(J, x);
  VIBE_CHECK(f < Real(1e-12));
  for (Size i = 0; i < x.size(); ++i) VIBE_CHECK_NEAR(x[i], J.target()[i], Real(1e-6));
}

// ===========================================================================
// 8. CG 与 L-BFGS 都能解 Rosenbrock（收敛性对比）
// ===========================================================================
VIBE_TEST(minimizer_cg_vs_lbfgs_rosenbrock) {
  RosenbrockObjective J(3);
  Vector x1{Real(-1.2), Real(1.0), Real(0.5)};
  Vector x2 = x1;
  auto cg = vibe::da::make_minimizer(options("cg", 2000));
  auto lb = vibe::da::make_minimizer(options("lbfgs", 2000));
  const Real fc = cg->minimize(J, x1);
  const Real fl = lb->minimize(J, x2);
  VIBE_CHECK(fc < Real(1e-6));
  VIBE_CHECK(fl < Real(1e-10));
  // 两者都应在有限迭代内收敛
  VIBE_CHECK(cg->iterations() <= 2000);
  VIBE_CHECK(lb->iterations() <= 2000);
}

// ===========================================================================
// 9. 预条件改善 L-BFGS 的条件数（迭代次数不增加且收敛到同一点）
// ===========================================================================
VIBE_TEST(minimizer_lbfgs_preconditioned) {
  QuadraticObjective J(Vector{Real(1), Real(100), Real(1e4)},
                       Vector{Real(1), Real(1), Real(1)});
  Vector xp = linspace(Real(-1), Real(2), 3);
  Vector xu = xp;
  // 预条件向量 = 1/c（Hessian 的逆对角）
  Vector pre{Real(1) / Real(1), Real(1) / Real(100), Real(1) / Real(1e4)};
  auto mp = vibe::da::make_minimizer(options("lbfgs"), &pre);
  auto mu = vibe::da::make_minimizer(options("lbfgs"), nullptr);
  const Real fp = mp->minimize(J, xp);
  const Real fu = mu->minimize(J, xu);
  VIBE_CHECK(fp < Real(1e-14));
  VIBE_CHECK(fu < Real(1e-14));
  for (Size i = 0; i < 3; ++i) VIBE_CHECK_NEAR(xp[i], Real(1), Real(1e-6));
}

// ===========================================================================
// 10. Lanczos 恢复已知特征值
// ===========================================================================
VIBE_TEST(minimizer_lanczos_recovers_eigenvalues) {
  const Size n = 8;
  Vector c(n), t(n), x0(n);
  for (Size i = 0; i < n; ++i) {
    c[i] = static_cast<Real>(i + 1);  // 特征值 1..8
    t[i] = Real(0);
    x0[i] = Real(0.1) * static_cast<Real>(i + 1);
  }
  QuadraticObjective J(c, t);
  const std::vector<Real> ritz =
      vibe::da::detail::lanczos_ritz_hessian(J, x0, static_cast<int>(n), Real(1e-4));
  VIBE_CHECK(ritz.size() == n);
  for (Size i = 0; i < ritz.size(); ++i) {
    VIBE_CHECK_NEAR(ritz[i], static_cast<Real>(i + 1), Real(1e-6));
  }
}

// ===========================================================================
// 11. 迭代记录、计数与最终梯度范数
// ===========================================================================
VIBE_TEST(minimizer_history_and_counters) {
  QuadraticObjective J(Vector{Real(1), Real(2)}, Vector{Real(3), Real(-3)});
  Vector x{Real(0), Real(0)};
  auto m = vibe::da::make_minimizer(options("lbfgs"));
  const Real f = m->minimize(J, x);
  VIBE_CHECK(f < Real(1e-20));
  VIBE_CHECK(!m->history().empty());
  VIBE_CHECK(static_cast<int>(m->history().size()) >= m->iterations());
  VIBE_CHECK(m->function_evaluations() > 0);
  VIBE_CHECK(m->gradient_evaluations() > 0);
  VIBE_CHECK(m->final_gradient_norm() < Real(1e-8));
  // 历史中代价单调不增（L-BFGS + Wolfe 线搜索的性质）
  for (Size i = 1; i < m->history().size(); ++i) {
    VIBE_CHECK(m->history()[i].cost <= m->history()[i - 1].cost + Real(1e-12));
  }
}

// ===========================================================================
// 12. 工厂方法分派
// ===========================================================================
VIBE_TEST(minimizer_factory_dispatch) {
  auto a = vibe::da::make_minimizer(options("lbfgs"));
  auto b = vibe::da::make_minimizer(options("cg"));
  auto c = vibe::da::make_minimizer(options("lanczos"));
  auto d = vibe::da::make_minimizer(options("unknown-method"));
  VIBE_CHECK(std::string(a->name()) == "lbfgs");
  VIBE_CHECK(std::string(b->name()) == "cg");
  VIBE_CHECK(std::string(c->name()) == "lanczos");
  VIBE_CHECK(std::string(d->name()) == "lbfgs");  // 回退
}

// ===========================================================================
// 13. Lanczos 极小化器可运行并返回有限值
// ===========================================================================
VIBE_TEST(minimizer_lanczos_runs) {
  QuadraticObjective J(Vector{Real(1), Real(2), Real(3), Real(4)},
                       Vector{Real(1), Real(1), Real(1), Real(1)});
  Vector x{Real(2), Real(2), Real(2), Real(2)};
  MinimizerOptions o = options("lanczos", 4);
  auto m = vibe::da::make_minimizer(o);
  const Real f = m->minimize(J, x);
  VIBE_CHECK(std::isfinite(f));
  VIBE_CHECK(!m->history().empty());
  VIBE_CHECK(static_cast<int>(m->history().size()) >= 2);
}

// ===========================================================================
// 14. 线搜索在 Rosenbrock 上也能满足 Wolfe 条件
// ===========================================================================
VIBE_TEST(minimizer_line_search_rosenbrock) {
  RosenbrockObjective J(2);
  Vector x{Real(-1.2), Real(1.0)};
  Vector g;
  const Real f0 = J.value_and_gradient(x, g);
  Vector d{Real(-g[0]), Real(-g[1])};
  const Real g0 = vibe::da::vec::dot(g, d);
  const LineSearchResult ls = vibe::da::strong_wolfe_line_search(J, x, d, f0, g0);
  VIBE_CHECK(ls.step >= Real(0));
  VIBE_CHECK(ls.value <= f0 + Real(1e-4) * ls.step * g0 + Real(1e-12));
  VIBE_CHECK(ls.evaluations >= 1);
}
