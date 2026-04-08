#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

#include "cudss.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace xla::ffi {
template <>
struct CtxBinding<cudaStream_t> {
  using Ctx = PlatformStream<cudaStream_t>;
};

template <>
struct CtxBinding<int32_t> {
  using Ctx = DeviceOrdinal;
};
}  // namespace xla::ffi

namespace {

using Clock = std::chrono::steady_clock;

struct CacheKey {
  int32_t device_ordinal;
  int32_t reordering_alg;
  int32_t nd_nlevels;
  int32_t host_nthreads;
  bool mt_enabled;
  std::string threading_lib;
  int64_t n;
  int64_t nnz;
  int64_t batch;
  const void* indptr;
  const void* indices;

  bool operator==(const CacheKey& other) const {
    return device_ordinal == other.device_ordinal &&
           reordering_alg == other.reordering_alg &&
           nd_nlevels == other.nd_nlevels &&
           host_nthreads == other.host_nthreads &&
           mt_enabled == other.mt_enabled &&
           threading_lib == other.threading_lib && n == other.n &&
           nnz == other.nnz && batch == other.batch && indptr == other.indptr &&
           indices == other.indices;
  }
};

struct CacheKeyHash {
  size_t operator()(const CacheKey& key) const {
    size_t seed = 0;
    auto combine = [&](size_t value) {
      seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    };
    combine(std::hash<int32_t>{}(key.device_ordinal));
    combine(std::hash<int32_t>{}(key.reordering_alg));
    combine(std::hash<int32_t>{}(key.nd_nlevels));
    combine(std::hash<int32_t>{}(key.host_nthreads));
    combine(std::hash<bool>{}(key.mt_enabled));
    combine(std::hash<std::string>{}(key.threading_lib));
    combine(std::hash<int64_t>{}(key.n));
    combine(std::hash<int64_t>{}(key.nnz));
    combine(std::hash<int64_t>{}(key.batch));
    combine(std::hash<const void*>{}(key.indptr));
    combine(std::hash<const void*>{}(key.indices));
    return seed;
  }
};

struct CachedResources {
  ~CachedResources() {
    if (a_matrix != nullptr) cudssMatrixDestroy(a_matrix);
    if (x_matrix != nullptr) cudssMatrixDestroy(x_matrix);
    if (b_matrix != nullptr) cudssMatrixDestroy(b_matrix);
    if (config != nullptr) cudssConfigDestroy(config);
    if (handle != nullptr) cudssDestroy(handle);
  }

  std::mutex mu;
  cudssHandle_t handle = nullptr;
  cudssConfig_t config = nullptr;
  cudssMatrix_t a_matrix = nullptr;
  cudssMatrix_t x_matrix = nullptr;
  cudssMatrix_t b_matrix = nullptr;
};

struct ProfileData {
  double setup_ms;
  double analysis_ms;
  double factorization_ms;
  double solve_ms;
  int32_t device_ordinal;
  int32_t reordering_alg;
  int32_t nd_nlevels;
  int32_t host_nthreads;
  bool mt_enabled;
  int64_t n;
  int64_t nnz;
  int64_t batch;
  bool cache_hit;
};

struct DataGuard {
  cudssHandle_t handle = nullptr;
  cudssData_t data = nullptr;

  ~DataGuard() {
    if (handle != nullptr && data != nullptr) {
      cudssDataDestroy(handle, data);
    }
  }
};

std::mutex& CacheMutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

auto& ResourceCache() {
  static auto* cache =
      new std::unordered_map<CacheKey, std::shared_ptr<CachedResources>,
                             CacheKeyHash>();
  return *cache;
}

std::mutex& ProfileMutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

std::optional<ProfileData>& LastProfileStorage() {
  static auto* profile = new std::optional<ProfileData>();
  return *profile;
}

bool ProfilingEnabled() {
  const char* value = std::getenv("JAX_CUDSS_PROFILE");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

double DurationMs(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void StoreLastProfile(ProfileData profile) {
  std::lock_guard<std::mutex> lock(ProfileMutex());
  LastProfileStorage() = std::move(profile);
}

void ClearLastProfileData() {
  std::lock_guard<std::mutex> lock(ProfileMutex());
  LastProfileStorage().reset();
}

std::string CudssStatusToString(cudssStatus_t status) {
  switch (status) {
    case CUDSS_STATUS_SUCCESS:
      return "CUDSS_STATUS_SUCCESS";
    case CUDSS_STATUS_NOT_INITIALIZED:
      return "CUDSS_STATUS_NOT_INITIALIZED";
    case CUDSS_STATUS_ALLOC_FAILED:
      return "CUDSS_STATUS_ALLOC_FAILED";
    case CUDSS_STATUS_INVALID_VALUE:
      return "CUDSS_STATUS_INVALID_VALUE";
    case CUDSS_STATUS_NOT_SUPPORTED:
      return "CUDSS_STATUS_NOT_SUPPORTED";
    case CUDSS_STATUS_EXECUTION_FAILED:
      return "CUDSS_STATUS_EXECUTION_FAILED";
    case CUDSS_STATUS_INTERNAL_ERROR:
      return "CUDSS_STATUS_INTERNAL_ERROR";
  }
  return "CUDSS_STATUS_UNKNOWN";
}

ffi::Error CudssError(const char* op, cudssStatus_t status) {
  std::ostringstream stream;
  stream << op << " failed with " << CudssStatusToString(status) << " ("
         << static_cast<int>(status) << ")";
  return {ffi::ErrorCode::kInternal, stream.str()};
}

int32_t ReorderingAlgFromEnv() {
  const char* value = std::getenv("JAX_CUDSS_REORDERING_ALG");
  if (value == nullptr || value[0] == '\0') {
    return static_cast<int32_t>(CUDSS_ALG_DEFAULT);
  }

  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' ||
      parsed < static_cast<long>(CUDSS_ALG_DEFAULT) ||
      parsed > static_cast<long>(CUDSS_ALG_5)) {
    return static_cast<int32_t>(CUDSS_ALG_DEFAULT);
  }
  return static_cast<int32_t>(parsed);
}

int32_t NdNLevelsFromEnv() {
  const char* value = std::getenv("JAX_CUDSS_ND_NLEVELS");
  if (value == nullptr || value[0] == '\0') {
    return -1;
  }

  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 0 ||
      parsed > static_cast<long>(std::numeric_limits<int32_t>::max())) {
    return -1;
  }
  return static_cast<int32_t>(parsed);
}

bool MtEnabledFromEnv() {
  const char* value = std::getenv("JAX_CUDSS_ENABLE_MT");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

int32_t HostNThreadsFromEnv() {
  const char* value = std::getenv("JAX_CUDSS_HOST_NTHREADS");
  if (value == nullptr || value[0] == '\0') {
    return -1;
  }

  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0 ||
      parsed > static_cast<long>(std::numeric_limits<int32_t>::max())) {
    return -1;
  }
  return static_cast<int32_t>(parsed);
}

std::string ThreadingLibFromEnv() {
  const char* value = std::getenv("JAX_CUDSS_THREADING_LIB");
  if (value != nullptr && value[0] != '\0') {
    return std::string(value);
  }
  value = std::getenv("CUDSS_THREADING_LIB");
  if (value != nullptr && value[0] != '\0') {
    return std::string(value);
  }
  return std::string();
}

ffi::ErrorOr<std::shared_ptr<CachedResources>> GetOrCreateResources(
    int32_t device_ordinal, int32_t reordering_alg, int32_t nd_nlevels,
    bool mt_enabled, int32_t host_nthreads, const std::string& threading_lib,
    cudaStream_t stream, int64_t n, int64_t nnz, int64_t batch,
    void* row_start, void* col_indices, void* a_values, void* b_values,
    void* x_values, bool* cache_hit) {
  CacheKey key{device_ordinal, reordering_alg, nd_nlevels, host_nthreads,
               mt_enabled, threading_lib, n, nnz, batch, row_start,
               col_indices};

  {
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto it = ResourceCache().find(key);
    if (it != ResourceCache().end()) {
      *cache_hit = true;
      return it->second;
    }
  }

  auto resources = std::make_shared<CachedResources>();
  cudssStatus_t status = cudssCreate(&resources->handle);
  if (status != CUDSS_STATUS_SUCCESS) {
    return ffi::Unexpected(CudssError("cudssCreate", status));
  }
  if (mt_enabled) {
    const char* threading_lib_path =
        threading_lib.empty() ? nullptr : threading_lib.c_str();
    status = cudssSetThreadingLayer(resources->handle, threading_lib_path);
    if (status != CUDSS_STATUS_SUCCESS) {
      return ffi::Unexpected(CudssError("cudssSetThreadingLayer", status));
    }
  }
  status = cudssSetStream(resources->handle, stream);
  if (status != CUDSS_STATUS_SUCCESS) {
    return ffi::Unexpected(CudssError("cudssSetStream", status));
  }
  status = cudssConfigCreate(&resources->config);
  if (status != CUDSS_STATUS_SUCCESS) {
    return ffi::Unexpected(CudssError("cudssConfigCreate", status));
  }

  cudssAlgType_t reordering_alg_value =
      static_cast<cudssAlgType_t>(reordering_alg);
  status = cudssConfigSet(resources->config, CUDSS_CONFIG_REORDERING_ALG,
                          &reordering_alg_value,
                          sizeof(reordering_alg_value));
  if (status != CUDSS_STATUS_SUCCESS) {
    return ffi::Unexpected(
        CudssError("cudssConfigSet(CUDSS_CONFIG_REORDERING_ALG)", status));
  }

  if (nd_nlevels >= 0) {
    int nd_nlevels_value = static_cast<int>(nd_nlevels);
    status = cudssConfigSet(resources->config, CUDSS_CONFIG_ND_NLEVELS,
                            &nd_nlevels_value, sizeof(nd_nlevels_value));
    if (status != CUDSS_STATUS_SUCCESS) {
      return ffi::Unexpected(
          CudssError("cudssConfigSet(CUDSS_CONFIG_ND_NLEVELS)", status));
    }
  }

  if (host_nthreads > 0) {
    int host_nthreads_value = static_cast<int>(host_nthreads);
    status = cudssConfigSet(resources->config, CUDSS_CONFIG_HOST_NTHREADS,
                            &host_nthreads_value,
                            sizeof(host_nthreads_value));
    if (status != CUDSS_STATUS_SUCCESS) {
      return ffi::Unexpected(
          CudssError("cudssConfigSet(CUDSS_CONFIG_HOST_NTHREADS)", status));
    }
  }

  int ubatch_size = static_cast<int>(batch);
  status = cudssConfigSet(resources->config, CUDSS_CONFIG_UBATCH_SIZE,
                          &ubatch_size, sizeof(ubatch_size));
  if (status != CUDSS_STATUS_SUCCESS) {
    return ffi::Unexpected(
        CudssError("cudssConfigSet(CUDSS_CONFIG_UBATCH_SIZE)", status));
  }

  status = cudssMatrixCreateCsr(
      &resources->a_matrix, n, n, nnz, row_start, nullptr, col_indices,
      a_values, CUDA_R_32I, CUDA_R_32F, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL,
      CUDSS_BASE_ZERO);
  if (status != CUDSS_STATUS_SUCCESS) {
    return ffi::Unexpected(CudssError("cudssMatrixCreateCsr", status));
  }

  status = cudssMatrixCreateDn(&resources->b_matrix, n, 1, n, b_values,
                               CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR);
  if (status != CUDSS_STATUS_SUCCESS) {
    return ffi::Unexpected(CudssError("cudssMatrixCreateDn(rhs)", status));
  }

  status = cudssMatrixCreateDn(&resources->x_matrix, n, 1, n, x_values,
                               CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR);
  if (status != CUDSS_STATUS_SUCCESS) {
    return ffi::Unexpected(CudssError("cudssMatrixCreateDn(solution)", status));
  }

  {
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto [it, inserted] = ResourceCache().emplace(key, resources);
    *cache_hit = !inserted;
    return it->second;
  }
}

ffi::Error JaxCudssUniformBatchSolveImpl(
    int32_t device_ordinal, cudaStream_t stream, ffi::Buffer<ffi::S32, 1> indptr,
    ffi::Buffer<ffi::S32, 1> indices, ffi::Buffer<ffi::F32, 2> values,
    ffi::Buffer<ffi::F32, 2> rhs, ffi::ResultBuffer<ffi::F32, 2> out) {
  const auto value_dims = values.dimensions();
  const auto rhs_dims = rhs.dimensions();
  const auto out_dims = out->dimensions();
  if (value_dims.size() != 2 || rhs_dims.size() != 2 || out_dims.size() != 2) {
    return {ffi::ErrorCode::kInvalidArgument,
            "expected values, rhs, and out to all be rank-2"};
  }

  const int64_t batch = value_dims[0];
  const int64_t nnz = value_dims[1];
  const int64_t n = indptr.dimensions()[0] - 1;
  if (n <= 0) {
    return {ffi::ErrorCode::kInvalidArgument,
            "indptr must describe a non-empty matrix"};
  }
  if (indices.dimensions()[0] != nnz) {
    return {ffi::ErrorCode::kInvalidArgument,
            "indices length must equal values.shape[1]"};
  }
  if (rhs_dims[0] != batch || rhs_dims[1] != n) {
    return {ffi::ErrorCode::kInvalidArgument,
            "rhs must have shape [batch, n]"};
  }
  if (out_dims[0] != batch || out_dims[1] != n) {
    return {ffi::ErrorCode::kInvalidArgument,
            "output must have shape [batch, n]"};
  }

  void* row_start = static_cast<void*>(indptr.typed_data());
  void* col_indices = static_cast<void*>(indices.typed_data());
  void* a_values = static_cast<void*>(values.typed_data());
  void* b_values = static_cast<void*>(rhs.typed_data());
  void* x_values = static_cast<void*>(out->typed_data());

  const bool profiling_enabled = ProfilingEnabled();
  const auto setup_start = Clock::now();
  bool cache_hit = false;
  const int32_t reordering_alg = ReorderingAlgFromEnv();
  const int32_t nd_nlevels = NdNLevelsFromEnv();
  const bool mt_enabled = MtEnabledFromEnv();
  const int32_t host_nthreads = HostNThreadsFromEnv();
  const std::string threading_lib = ThreadingLibFromEnv();
  auto maybe_resources = GetOrCreateResources(
      device_ordinal, reordering_alg, nd_nlevels, mt_enabled, host_nthreads,
      threading_lib, stream, n, nnz, batch, row_start, col_indices, a_values,
      b_values, x_values, &cache_hit);
  if (!maybe_resources.has_value()) {
    return maybe_resources.error();
  }
  std::shared_ptr<CachedResources> resources = *maybe_resources;

  std::lock_guard<std::mutex> lock(resources->mu);
  cudssStatus_t status = cudssSetStream(resources->handle, stream);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssSetStream", status);
  }
  status = cudssMatrixSetValues(resources->a_matrix, a_values);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssMatrixSetValues(A)", status);
  }
  status = cudssMatrixSetValues(resources->b_matrix, b_values);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssMatrixSetValues(rhs)", status);
  }
  status = cudssMatrixSetValues(resources->x_matrix, x_values);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssMatrixSetValues(solution)", status);
  }

  DataGuard data_guard;
  data_guard.handle = resources->handle;
  status = cudssDataCreate(resources->handle, &data_guard.data);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssDataCreate", status);
  }
  const auto setup_end = Clock::now();

  const auto analysis_start = Clock::now();
  status = cudssExecute(resources->handle, CUDSS_PHASE_ANALYSIS,
                        resources->config, data_guard.data, resources->a_matrix,
                        resources->x_matrix, resources->b_matrix);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssExecute(CUDSS_PHASE_ANALYSIS)", status);
  }
  const auto analysis_end = Clock::now();

  const auto factorization_start = Clock::now();
  status = cudssExecute(resources->handle, CUDSS_PHASE_FACTORIZATION,
                        resources->config, data_guard.data, resources->a_matrix,
                        resources->x_matrix, resources->b_matrix);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssExecute(CUDSS_PHASE_FACTORIZATION)", status);
  }
  const auto factorization_end = Clock::now();

  int info = 0;
  size_t size_written = 0;
  status = cudssDataGet(resources->handle, data_guard.data, CUDSS_DATA_INFO,
                        &info, sizeof(info), &size_written);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssDataGet(CUDSS_DATA_INFO)", status);
  }
  if (size_written >= sizeof(info) && info != 0) {
    std::ostringstream stream_info;
    stream_info << "cuDSS factorization reported non-zero info=" << info;
    return {ffi::ErrorCode::kInternal, stream_info.str()};
  }

  const auto solve_start = Clock::now();
  status = cudssExecute(resources->handle, CUDSS_PHASE_SOLVE, resources->config,
                        data_guard.data, resources->a_matrix,
                        resources->x_matrix, resources->b_matrix);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssExecute(CUDSS_PHASE_SOLVE)", status);
  }
  const auto solve_end = Clock::now();

  if (profiling_enabled) {
    StoreLastProfile(ProfileData{
        DurationMs(setup_start, setup_end),
        DurationMs(analysis_start, analysis_end),
        DurationMs(factorization_start, factorization_end),
        DurationMs(solve_start, solve_end),
        device_ordinal,
        reordering_alg,
        nd_nlevels,
        host_nthreads,
        mt_enabled,
        n,
        nnz,
        batch,
        cache_hit,
    });
  }

  return ffi::Error();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(JaxCudssUniformBatchSolve,
                              JaxCudssUniformBatchSolveImpl);

PyObject* Registrations(PyObject*, PyObject*) {
  PyObject* dict = PyDict_New();
  if (dict == nullptr) {
    return nullptr;
  }
  PyObject* capsule =
      PyCapsule_New(reinterpret_cast<void*>(JaxCudssUniformBatchSolve), nullptr,
                    nullptr);
  if (capsule == nullptr) {
    Py_DECREF(dict);
    return nullptr;
  }
  if (PyDict_SetItemString(dict, "jax_cudss_uniform_batch_solve", capsule) !=
      0) {
    Py_DECREF(capsule);
    Py_DECREF(dict);
    return nullptr;
  }
  Py_DECREF(capsule);
  return dict;
}

PyObject* LastProfile(PyObject*, PyObject*) {
  std::lock_guard<std::mutex> lock(ProfileMutex());
  if (!LastProfileStorage().has_value()) {
    Py_RETURN_NONE;
  }

  const ProfileData& profile = *LastProfileStorage();
  PyObject* dict = PyDict_New();
  if (dict == nullptr) {
    return nullptr;
  }

  auto add_float = [&](const char* name, double value) -> bool {
    PyObject* py_value = PyFloat_FromDouble(value);
    if (py_value == nullptr) {
      return false;
    }
    const int rc = PyDict_SetItemString(dict, name, py_value);
    Py_DECREF(py_value);
    return rc == 0;
  };
  auto add_int = [&](const char* name, long long value) -> bool {
    PyObject* py_value = PyLong_FromLongLong(value);
    if (py_value == nullptr) {
      return false;
    }
    const int rc = PyDict_SetItemString(dict, name, py_value);
    Py_DECREF(py_value);
    return rc == 0;
  };
  auto add_bool = [&](const char* name, bool value) -> bool {
    PyObject* py_value = PyBool_FromLong(value ? 1 : 0);
    if (py_value == nullptr) {
      return false;
    }
    const int rc = PyDict_SetItemString(dict, name, py_value);
    Py_DECREF(py_value);
    return rc == 0;
  };

  if (!add_float("setup_ms", profile.setup_ms) ||
      !add_float("analysis_ms", profile.analysis_ms) ||
      !add_float("factorization_ms", profile.factorization_ms) ||
      !add_float("solve_ms", profile.solve_ms) ||
      !add_int("device_ordinal", profile.device_ordinal) ||
      !add_int("reordering_alg", profile.reordering_alg) ||
      !add_int("nd_nlevels", profile.nd_nlevels) ||
      !add_int("host_nthreads", profile.host_nthreads) ||
      !add_bool("mt_enabled", profile.mt_enabled) ||
      !add_int("n", profile.n) ||
      !add_int("nnz", profile.nnz) ||
      !add_int("batch", profile.batch) ||
      !add_bool("cache_hit", profile.cache_hit)) {
    Py_DECREF(dict);
    return nullptr;
  }

  return dict;
}

PyObject* ClearLastProfile(PyObject*, PyObject*) {
  ClearLastProfileData();
  Py_RETURN_NONE;
}

PyMethodDef kMethods[] = {
    {"registrations", Registrations, METH_NOARGS,
     PyDoc_STR("Return JAX FFI custom call registrations.")},
    {"last_profile", LastProfile, METH_NOARGS,
     PyDoc_STR("Return the last captured profiling record, if any.")},
    {"clear_last_profile", ClearLastProfile, METH_NOARGS,
     PyDoc_STR("Clear the last captured profiling record.")},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "_cudss",
    "Native cuDSS FFI bindings for JAX.",
    -1,
    kMethods,
};

}  // namespace

PyMODINIT_FUNC PyInit__cudss(void) { return PyModule_Create(&kModule); }
