/// @file mpi_wrapper.cpp
/// @brief MPI 薄封装的实现；未启用 MPI 时全部退化为单进程语义。

#include "vibe/common/mpi_wrapper.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "vibe/common/error.hpp"
#include "vibe/common/logging.hpp"

namespace vibe::common {

#ifdef VIBE_HAVE_MPI
namespace {
bool g_mpi_active = false;
int g_owns_mpi = 0;
}  // namespace
#endif

Comm Comm::world() {
  Comm c;
#ifdef VIBE_HAVE_MPI
  c.handle_ = g_mpi_active ? MPI_COMM_WORLD : MPI_COMM_NULL;
  if (g_mpi_active) {
    MPI_Comm_rank(MPI_COMM_WORLD, &c.rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &c.size_);
  }
#else
  c.handle_ = 0;
  c.rank_ = 0;
  c.size_ = 1;
#endif
  return c;
}

void Comm::barrier() const {
#ifdef VIBE_HAVE_MPI
  if (g_mpi_active) MPI_Barrier(handle_);
#endif
}

Real Comm::allreduce_sum(Real v) const {
#ifdef VIBE_HAVE_MPI
  if (g_mpi_active && size_ > 1) {
    Real out = 0;
    MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_SUM, handle_);
    return out;
  }
#endif
  return v;
}

Real Comm::allreduce_max(Real v) const {
#ifdef VIBE_HAVE_MPI
  if (g_mpi_active && size_ > 1) {
    Real out = 0;
    MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_MAX, handle_);
    return out;
  }
#endif
  return v;
}

Real Comm::allreduce_min(Real v) const {
#ifdef VIBE_HAVE_MPI
  if (g_mpi_active && size_ > 1) {
    Real out = 0;
    MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_MIN, handle_);
    return out;
  }
#endif
  return v;
}

void Comm::allreduce_sum(std::vector<Real>& v) const {
#ifdef VIBE_HAVE_MPI
  if (g_mpi_active && size_ > 1) {
    std::vector<Real> out(v.size(), Real(0));
    MPI_Allreduce(v.data(), out.data(), static_cast<int>(v.size()), MPI_DOUBLE, MPI_SUM,
                  handle_);
    v.swap(out);
  }
#endif
}

std::pair<Real, int> Comm::max_with_rank(Real v) const {
#ifdef VIBE_HAVE_MPI
  if (g_mpi_active && size_ > 1) {
    struct { Real val; int rank; } in{v, rank_}, out{};
    MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, handle_);
    return {out.val, out.rank};
  }
#endif
  return {v, rank_};
}

void Comm::bcast(std::string& s, int root) const {
#ifdef VIBE_HAVE_MPI
  if (g_mpi_active && size_ > 1) {
    int len = static_cast<int>(s.size());
    MPI_Bcast(&len, 1, MPI_INT, root, handle_);
    s.resize(static_cast<Size>(len));
    MPI_Bcast(s.data(), len, MPI_CHAR, root, handle_);
  }
#else
  VIBE_UNUSED(root);
#endif
}

void Comm::bcast(std::vector<Real>& v, int root) const {
#ifdef VIBE_HAVE_MPI
  if (g_mpi_active && size_ > 1) {
    MPI_Bcast(v.data(), static_cast<int>(v.size()), MPI_DOUBLE, root, handle_);
  }
#else
  VIBE_UNUSED(root);
#endif
}

std::vector<int> Comm::partition_1d(Int n, int nparts) {
  std::vector<int> counts(static_cast<Size>(std::max(nparts, 1)), 0);
  const Int base = n / nparts;
  const Int rem = n % nparts;
  for (int p = 0; p < nparts; ++p) {
    counts[static_cast<Size>(p)] = static_cast<int>(base + (p < rem ? 1 : 0));
  }
  return counts;
}

void Comm::initialize(int* argc, char*** argv) {
#ifdef VIBE_HAVE_MPI
  int provided = 0;
  int already = 0;
  MPI_Initialized(&already);
  if (!already) {
    MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
    g_owns_mpi = 1;
    g_mpi_active = true;
    VIBE_INFO("MPI 初始化完成，线程级别=", provided);
  } else {
    g_mpi_active = true;
  }
#else
  VIBE_UNUSED(argc);
  VIBE_UNUSED(argv);
#endif
}

void Comm::finalize() {
#ifdef VIBE_HAVE_MPI
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (g_mpi_active && g_owns_mpi && !finalized) {
    MPI_Finalize();
  }
  g_mpi_active = false;
#endif
}

bool Comm::initialized() noexcept {
#ifdef VIBE_HAVE_MPI
  return g_mpi_active;
#else
  return true;
#endif
}

void Comm::abort(int code) const {
#ifdef VIBE_HAVE_MPI
  if (g_mpi_active) MPI_Abort(handle_, code);
#endif
  std::exit(code);
}

}  // namespace vibe::common
