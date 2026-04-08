#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

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
using S32BufferR1 = ffi::Buffer<ffi::S32, 1>;
using F32BufferR2 = ffi::Buffer<ffi::F32, 2>;
using S64ResultR0 = ffi::ResultBuffer<ffi::S64, 0>;
using F32ResultR2 = ffi::ResultBuffer<ffi::F32, 2>;

struct PreparedCacheKey {
  int32_t device_ordinal;
  int32_t reordering_alg;
  int32_t nd_nlevels;
  int32_t host_nthreads;
  bool mt_enabled;
  std::string threading_lib;
  int64_t structure_hash;
  int64_t n;
  int64_t nnz;
  int64_t batch;

  bool operator==(const PreparedCacheKey& other) const {
    return device_ordinal == other.device_ordinal &&
           reordering_alg == other.reordering_alg &&
           nd_nlevels == other.nd_nlevels &&
           host_nthreads == other.host_nthreads &&
           mt_enabled == other.mt_enabled &&
           threading_lib == other.threading_lib &&
           structure_hash == other.structure_hash && n == other.n &&
           nnz == other.nnz && batch == other.batch;
  }
};

struct PreparedCacheKeyHash {
  size_t operator()(const PreparedCacheKey& key) const {
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
    combine(std::hash<int64_t>{}(key.structure_hash));
    combine(std::hash<int64_t>{}(key.n));
    combine(std::hash<int64_t>{}(key.nnz));
    combine(std::hash<int64_t>{}(key.batch));
    return seed;
  }
};

struct SetupProfileData {
  double setup_ms;
  double analysis_ms;
  double factorization_ms;
  double solve_ms;
  int32_t device_ordinal;
  int32_t reordering_alg;
  int32_t nd_nlevels;
  int32_t host_nthreads;
  bool mt_enabled;
  bool async_allocator_enabled;
  int64_t structure_hash;
  int64_t token;
  int64_t n;
  int64_t nnz;
  int64_t batch;
  bool cache_hit;
};

struct PreparedSolver {
  ~PreparedSolver() {
    if (last_use_event != nullptr) {
      cudaEventDestroy(last_use_event);
    }
    if (a_matrix != nullptr) {
      cudssMatrixDestroy(a_matrix);
    }
    if (x_matrix != nullptr) {
      cudssMatrixDestroy(x_matrix);
    }
    if (b_matrix != nullptr) {
      cudssMatrixDestroy(b_matrix);
    }
    if (data != nullptr && handle != nullptr) {
      cudssDataDestroy(handle, data);
    }
    if (config != nullptr) {
      cudssConfigDestroy(config);
    }
    if (handle != nullptr) {
      cudssDestroy(handle);
    }
    if (dummy_a_values != nullptr) {
      cudaFree(dummy_a_values);
    }
    if (dummy_b_values != nullptr) {
      cudaFree(dummy_b_values);
    }
    if (dummy_x_values != nullptr) {
      cudaFree(dummy_x_values);
    }
    if (device_indptr != nullptr) {
      cudaFree(device_indptr);
    }
    if (device_indices != nullptr) {
      cudaFree(device_indices);
    }
  }

  std::mutex mu;
  PreparedCacheKey key;
  int64_t token = 0;
  int64_t owners = 0;
  uint64_t last_use_seq = 0;
  bool async_allocator_enabled = false;
  cudssHandle_t handle = nullptr;
  cudssConfig_t config = nullptr;
  cudssData_t data = nullptr;
  cudssMatrix_t a_matrix = nullptr;
  cudssMatrix_t x_matrix = nullptr;
  cudssMatrix_t b_matrix = nullptr;
  void* dummy_a_values = nullptr;
  void* dummy_b_values = nullptr;
  void* dummy_x_values = nullptr;
  void* device_indptr = nullptr;
  void* device_indices = nullptr;
  cudaEvent_t last_use_event = nullptr;
};

std::mutex& RegistryMutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

std::unordered_map<int64_t, std::shared_ptr<PreparedSolver>>& PreparedByToken() {
  static auto* registry =
      new std::unordered_map<int64_t, std::shared_ptr<PreparedSolver>>();
  return *registry;
}

std::unordered_map<PreparedCacheKey, int64_t, PreparedCacheKeyHash>&
PreparedByKey() {
  static auto* registry =
      new std::unordered_map<PreparedCacheKey, int64_t, PreparedCacheKeyHash>();
  return *registry;
}

std::mutex& ProfileMutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

std::optional<SetupProfileData>& LastProfileStorage() {
  static auto* profile = new std::optional<SetupProfileData>();
  return *profile;
}

int64_t& NextPreparedToken() {
  static auto* next = new int64_t(1);
  return *next;
}

uint64_t& UseSequence() {
  static auto* next = new uint64_t(1);
  return *next;
}

bool ProfilingEnabled() {
  const char* value = std::getenv("JAX_CUDSS_PROFILE");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool DebugLoggingEnabled() {
  const char* value = std::getenv("JAX_CUDSS_DEBUG_LOG");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void DebugLog(const char* message) {
  if (!DebugLoggingEnabled()) {
    return;
  }
  std::fprintf(stderr, "jax_cudss debug: %s\n", message);
  std::fflush(stderr);
}

double DurationMs(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void StoreLastProfile(SetupProfileData profile) {
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

ffi::Error CudaError(const char* op, cudaError_t status) {
  std::ostringstream stream;
  stream << op << " failed with " << cudaGetErrorName(status) << ": "
         << cudaGetErrorString(status);
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

int64_t MaxPreparedSolversFromEnv() {
  const char* value = std::getenv("JAX_CUDSS_MAX_PREPARED_SOLVERS");
  if (value == nullptr || value[0] == '\0') {
    return 16;
  }

  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0) {
    return 16;
  }
  return parsed;
}

int AsyncDeviceAlloc(void*, void** ptr, size_t size, cudaStream_t stream) {
  return cudaMallocAsync(ptr, size, stream) == cudaSuccess ? 0 : 1;
}

int AsyncDeviceFree(void*, void* ptr, size_t, cudaStream_t stream) {
  return cudaFreeAsync(ptr, stream) == cudaSuccess ? 0 : 1;
}

cudssDeviceMemHandler_t AsyncAllocatorHandler() {
  cudssDeviceMemHandler_t handler{};
  handler.ctx = nullptr;
  handler.device_alloc = &AsyncDeviceAlloc;
  handler.device_free = &AsyncDeviceFree;
  std::strncpy(handler.name, "jax_cudss_async",
               CUDSS_ALLOCATOR_NAME_LEN - 1);
  handler.name[CUDSS_ALLOCATOR_NAME_LEN - 1] = '\0';
  return handler;
}

ffi::Error AllocateDummyBuffers(PreparedSolver& prepared, cudaStream_t stream) {
  const size_t a_bytes =
      static_cast<size_t>(prepared.key.batch) *
      static_cast<size_t>(prepared.key.nnz) * sizeof(float);
  const size_t rhs_bytes =
      static_cast<size_t>(prepared.key.batch) *
      static_cast<size_t>(prepared.key.n) * sizeof(float);
  const size_t indptr_bytes =
      static_cast<size_t>(prepared.key.n + 1) * sizeof(int32_t);
  const size_t indices_bytes =
      static_cast<size_t>(prepared.key.nnz) * sizeof(int32_t);

  cudaError_t cuda_status = cudaMallocAsync(&prepared.dummy_a_values, a_bytes, stream);
  if (cuda_status != cudaSuccess) {
    return CudaError("cudaMallocAsync(dummy_a_values)", cuda_status);
  }
  cuda_status = cudaMallocAsync(&prepared.dummy_b_values, rhs_bytes, stream);
  if (cuda_status != cudaSuccess) {
    return CudaError("cudaMallocAsync(dummy_b_values)", cuda_status);
  }
  cuda_status = cudaMallocAsync(&prepared.dummy_x_values, rhs_bytes, stream);
  if (cuda_status != cudaSuccess) {
    return CudaError("cudaMallocAsync(dummy_x_values)", cuda_status);
  }
  cuda_status = cudaMallocAsync(&prepared.device_indptr, indptr_bytes, stream);
  if (cuda_status != cudaSuccess) {
    return CudaError("cudaMallocAsync(device_indptr)", cuda_status);
  }
  cuda_status = cudaMallocAsync(&prepared.device_indices, indices_bytes, stream);
  if (cuda_status != cudaSuccess) {
    return CudaError("cudaMallocAsync(device_indices)", cuda_status);
  }
  return ffi::Error::Success();
}

ffi::Error ConfigurePreparedSolver(PreparedSolver& prepared, cudaStream_t stream) {
  cudssStatus_t status = cudssCreate(&prepared.handle);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssCreate", status);
  }
  if (prepared.key.mt_enabled) {
    const char* threading_lib =
        prepared.key.threading_lib.empty() ? nullptr
                                           : prepared.key.threading_lib.c_str();
    status = cudssSetThreadingLayer(prepared.handle, threading_lib);
    if (status != CUDSS_STATUS_SUCCESS) {
      return CudssError("cudssSetThreadingLayer", status);
    }
  }

  cudssDeviceMemHandler_t async_handler = AsyncAllocatorHandler();
  status = cudssSetDeviceMemHandler(prepared.handle, &async_handler);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssSetDeviceMemHandler", status);
  }
  prepared.async_allocator_enabled = true;

  status = cudssSetStream(prepared.handle, stream);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssSetStream", status);
  }

  status = cudssConfigCreate(&prepared.config);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssConfigCreate", status);
  }

  cudssAlgType_t reordering_alg =
      static_cast<cudssAlgType_t>(prepared.key.reordering_alg);
  status = cudssConfigSet(prepared.config, CUDSS_CONFIG_REORDERING_ALG,
                          &reordering_alg, sizeof(reordering_alg));
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssConfigSet(CUDSS_CONFIG_REORDERING_ALG)", status);
  }

  if (prepared.key.nd_nlevels >= 0) {
    int nd_nlevels = static_cast<int>(prepared.key.nd_nlevels);
    status = cudssConfigSet(prepared.config, CUDSS_CONFIG_ND_NLEVELS,
                            &nd_nlevels, sizeof(nd_nlevels));
    if (status != CUDSS_STATUS_SUCCESS) {
      return CudssError("cudssConfigSet(CUDSS_CONFIG_ND_NLEVELS)", status);
    }
  }

  if (prepared.key.host_nthreads > 0) {
    int host_nthreads = static_cast<int>(prepared.key.host_nthreads);
    status = cudssConfigSet(prepared.config, CUDSS_CONFIG_HOST_NTHREADS,
                            &host_nthreads, sizeof(host_nthreads));
    if (status != CUDSS_STATUS_SUCCESS) {
      return CudssError("cudssConfigSet(CUDSS_CONFIG_HOST_NTHREADS)", status);
    }
  }

  int ubatch_size = static_cast<int>(prepared.key.batch);
  status = cudssConfigSet(prepared.config, CUDSS_CONFIG_UBATCH_SIZE,
                          &ubatch_size, sizeof(ubatch_size));
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssConfigSet(CUDSS_CONFIG_UBATCH_SIZE)", status);
  }

  ffi::Error dummy_status = AllocateDummyBuffers(prepared, stream);
  if (!dummy_status.success()) {
    return dummy_status;
  }

  status = cudssMatrixCreateCsr(
      &prepared.a_matrix, prepared.key.n, prepared.key.n, prepared.key.nnz,
      prepared.device_indptr, nullptr, prepared.device_indices,
      prepared.dummy_a_values, CUDA_R_32I, CUDA_R_32F, CUDSS_MTYPE_GENERAL,
      CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssMatrixCreateCsr", status);
  }

  status = cudssMatrixCreateDn(&prepared.b_matrix, prepared.key.n, 1, prepared.key.n,
                               prepared.dummy_b_values, CUDA_R_32F,
                               CUDSS_LAYOUT_COL_MAJOR);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssMatrixCreateDn(rhs)", status);
  }

  status = cudssMatrixCreateDn(&prepared.x_matrix, prepared.key.n, 1, prepared.key.n,
                               prepared.dummy_x_values, CUDA_R_32F,
                               CUDSS_LAYOUT_COL_MAJOR);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssMatrixCreateDn(solution)", status);
  }

  status = cudssDataCreate(prepared.handle, &prepared.data);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssDataCreate", status);
  }

  return ffi::Error::Success();
}

ffi::Error SetPreparedStructure(PreparedSolver& prepared, void* indptr,
                                void* indices, cudaStream_t stream) {
  cudaError_t cuda_status = cudaMemcpyAsync(
      prepared.device_indptr, indptr,
      static_cast<size_t>(prepared.key.n + 1) * sizeof(int32_t),
      cudaMemcpyDeviceToDevice, stream);
  if (cuda_status != cudaSuccess) {
    return CudaError("cudaMemcpyAsync(device_indptr)", cuda_status);
  }
  cuda_status = cudaMemcpyAsync(
      prepared.device_indices, indices,
      static_cast<size_t>(prepared.key.nnz) * sizeof(int32_t),
      cudaMemcpyDeviceToDevice, stream);
  if (cuda_status != cudaSuccess) {
    return CudaError("cudaMemcpyAsync(device_indices)", cuda_status);
  }

  cudssStatus_t status =
      cudssMatrixSetCsrPointers(prepared.a_matrix, prepared.device_indptr, nullptr,
                                prepared.device_indices, prepared.dummy_a_values);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssMatrixSetCsrPointers(A)", status);
  }
  return ffi::Error::Success();
}

ffi::Error EnsureRecordedEvent(PreparedSolver& prepared, cudaStream_t stream) {
  if (prepared.last_use_event == nullptr) {
    cudaError_t cuda_status =
        cudaEventCreateWithFlags(&prepared.last_use_event, cudaEventDisableTiming);
    if (cuda_status != cudaSuccess) {
      return CudaError("cudaEventCreateWithFlags", cuda_status);
    }
  }
  cudaError_t cuda_status = cudaEventRecord(prepared.last_use_event, stream);
  if (cuda_status != cudaSuccess) {
    return CudaError("cudaEventRecord", cuda_status);
  }
  return ffi::Error::Success();
}

ffi::Error WriteTokenResult(S64ResultR0 out, int64_t token, cudaStream_t stream) {
  cudaError_t cuda_status =
      cudaMemcpyAsync(out->typed_data(), &token, sizeof(token),
                      cudaMemcpyHostToDevice, stream);
  if (cuda_status != cudaSuccess) {
    return CudaError("cudaMemcpyAsync(result_token)", cuda_status);
  }
  return ffi::Error::Success();
}

bool ReadyForEviction(const PreparedSolver& prepared) {
  if (prepared.last_use_event == nullptr) {
    return true;
  }
  cudaError_t status = cudaEventQuery(prepared.last_use_event);
  if (status == cudaSuccess) {
    return true;
  }
  if (status == cudaErrorNotReady) {
    return false;
  }
  cudaGetLastError();
  return false;
}

void MaybeEvictPreparedLocked() {
  const int64_t capacity = MaxPreparedSolversFromEnv();
  while (static_cast<int64_t>(PreparedByToken().size()) > capacity) {
    std::shared_ptr<PreparedSolver> candidate;
    for (const auto& [token, prepared] : PreparedByToken()) {
      if (prepared->owners != 0 || !ReadyForEviction(*prepared)) {
        continue;
      }
      if (candidate == nullptr ||
          prepared->last_use_seq < candidate->last_use_seq) {
        candidate = prepared;
      }
    }
    if (candidate == nullptr) {
      break;
    }
    PreparedByKey().erase(candidate->key);
    PreparedByToken().erase(candidate->token);
  }
}

ffi::ErrorOr<std::shared_ptr<PreparedSolver>> LookupPreparedSolver(
    int64_t token, bool update_lru) {
  std::lock_guard<std::mutex> registry_lock(RegistryMutex());
  auto it = PreparedByToken().find(token);
  if (it == PreparedByToken().end()) {
    std::ostringstream stream;
    stream << "prepared solver token " << token
           << " is not available; it may have been evicted or never created";
    return ffi::Unexpected(
        ffi::Error(ffi::ErrorCode::kInvalidArgument, stream.str()));
  }
  std::shared_ptr<PreparedSolver> prepared = it->second;
  if (update_lru) {
    prepared->last_use_seq = UseSequence()++;
  }
  return prepared;
}

ffi::Error JaxCudssSetupSolverImpl(int32_t device_ordinal, cudaStream_t stream,
                                   S32BufferR1 indptr, S32BufferR1 indices,
                                   int64_t batch_size, int64_t structure_hash,
                                   S64ResultR0 out) {
  DebugLog("setup enter");
  const int64_t n = indptr.dimensions()[0] - 1;
  const int64_t nnz = indices.dimensions()[0];
  if (n <= 0) {
    return {ffi::ErrorCode::kInvalidArgument,
            "indptr must describe a non-empty matrix"};
  }
  if (batch_size <= 0) {
    return {ffi::ErrorCode::kInvalidArgument,
            "batch_size must be a positive integer"};
  }

  const int32_t reordering_alg = ReorderingAlgFromEnv();
  const int32_t nd_nlevels = NdNLevelsFromEnv();
  const bool mt_enabled = MtEnabledFromEnv();
  const int32_t host_nthreads = HostNThreadsFromEnv();
  const std::string threading_lib = ThreadingLibFromEnv();
  const bool profiling_enabled = ProfilingEnabled();
  const auto setup_start = Clock::now();

  PreparedCacheKey key{device_ordinal, reordering_alg, nd_nlevels, host_nthreads,
                       mt_enabled, threading_lib, structure_hash, n, nnz,
                       batch_size};

  std::lock_guard<std::mutex> registry_lock(RegistryMutex());
  auto key_it = PreparedByKey().find(key);
  if (key_it != PreparedByKey().end()) {
    auto token_it = PreparedByToken().find(key_it->second);
    if (token_it != PreparedByToken().end()) {
      std::shared_ptr<PreparedSolver> prepared = token_it->second;
      prepared->owners += 1;
      prepared->last_use_seq = UseSequence()++;
      DebugLog("setup cache hit");
      ffi::Error write_status = WriteTokenResult(out, prepared->token, stream);
      if (!write_status.success()) {
        return write_status;
      }
      if (profiling_enabled) {
        const auto setup_end = Clock::now();
        StoreLastProfile(SetupProfileData{
            DurationMs(setup_start, setup_end),
            0.0,
            0.0,
            0.0,
            device_ordinal,
            reordering_alg,
            nd_nlevels,
            host_nthreads,
            mt_enabled,
            prepared->async_allocator_enabled,
            structure_hash,
            prepared->token,
            n,
            nnz,
            batch_size,
            true,
        });
      }
      return ffi::Error::Success();
    }
    PreparedByKey().erase(key_it);
  }

  auto prepared = std::make_shared<PreparedSolver>();
  prepared->key = key;
  prepared->token = NextPreparedToken()++;
  prepared->owners = 1;
  prepared->last_use_seq = UseSequence()++;

  ffi::Error config_status = ConfigurePreparedSolver(*prepared, stream);
  if (!config_status.success()) {
    return config_status;
  }
  DebugLog("setup configured");

  ffi::Error structure_status = SetPreparedStructure(
      *prepared, static_cast<void*>(indptr.typed_data()),
      static_cast<void*>(indices.typed_data()), stream);
  if (!structure_status.success()) {
    return structure_status;
  }
  DebugLog("setup structure copied");

  const auto pre_analysis_end = Clock::now();
  cudssStatus_t status = cudssExecute(prepared->handle, CUDSS_PHASE_ANALYSIS,
                                      prepared->config, prepared->data,
                                      prepared->a_matrix, prepared->x_matrix,
                                      prepared->b_matrix);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssExecute(CUDSS_PHASE_ANALYSIS)", status);
  }
  DebugLog("setup analysis done");
  const auto analysis_end = Clock::now();

  ffi::Error event_status = EnsureRecordedEvent(*prepared, stream);
  if (!event_status.success()) {
    return event_status;
  }
  DebugLog("setup event recorded");

  PreparedByToken().emplace(prepared->token, prepared);
  PreparedByKey().emplace(prepared->key, prepared->token);
  MaybeEvictPreparedLocked();

  ffi::Error write_status = WriteTokenResult(out, prepared->token, stream);
  if (!write_status.success()) {
    return write_status;
  }
  DebugLog("setup token copied");
  if (profiling_enabled) {
    StoreLastProfile(SetupProfileData{
        DurationMs(setup_start, pre_analysis_end),
        DurationMs(pre_analysis_end, analysis_end),
        0.0,
        0.0,
        device_ordinal,
        reordering_alg,
        nd_nlevels,
        host_nthreads,
        mt_enabled,
        prepared->async_allocator_enabled,
        structure_hash,
        prepared->token,
        n,
        nnz,
        batch_size,
        false,
    });
  }
  return ffi::Error::Success();
}

ffi::Error JaxCudssSolveGraphImpl(int32_t, cudaStream_t stream,
                                  F32BufferR2 values, F32BufferR2 rhs,
                                  int64_t token, F32ResultR2 out) {
  auto maybe_prepared = LookupPreparedSolver(token, /*update_lru=*/true);
  if (!maybe_prepared.has_value()) {
    return maybe_prepared.error();
  }
  std::shared_ptr<PreparedSolver> prepared = *maybe_prepared;

  const auto value_dims = values.dimensions();
  const auto rhs_dims = rhs.dimensions();
  const auto out_dims = out->dimensions();
  if (value_dims[0] != prepared->key.batch ||
      value_dims[1] != prepared->key.nnz) {
    std::ostringstream stream_dims;
    stream_dims << "values must have shape [" << prepared->key.batch << ", "
                << prepared->key.nnz << "]";
    return {ffi::ErrorCode::kInvalidArgument, stream_dims.str()};
  }
  if (rhs_dims[0] != prepared->key.batch || rhs_dims[1] != prepared->key.n) {
    std::ostringstream stream_dims;
    stream_dims << "rhs must have shape [" << prepared->key.batch << ", "
                << prepared->key.n << "]";
    return {ffi::ErrorCode::kInvalidArgument, stream_dims.str()};
  }
  if (out_dims[0] != prepared->key.batch || out_dims[1] != prepared->key.n) {
    std::ostringstream stream_dims;
    stream_dims << "output must have shape [" << prepared->key.batch << ", "
                << prepared->key.n << "]";
    return {ffi::ErrorCode::kInvalidArgument, stream_dims.str()};
  }

  std::lock_guard<std::mutex> handle_lock(prepared->mu);
  cudssStatus_t status = cudssSetStream(prepared->handle, stream);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssSetStream", status);
  }
  status = cudssMatrixSetValues(prepared->a_matrix,
                                static_cast<void*>(values.typed_data()));
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssMatrixSetValues(A)", status);
  }
  status = cudssMatrixSetValues(prepared->b_matrix,
                                static_cast<void*>(rhs.typed_data()));
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssMatrixSetValues(rhs)", status);
  }
  status =
      cudssMatrixSetValues(prepared->x_matrix, static_cast<void*>(out->typed_data()));
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssMatrixSetValues(solution)", status);
  }

  status = cudssExecute(prepared->handle, CUDSS_PHASE_FACTORIZATION,
                        prepared->config, prepared->data, prepared->a_matrix,
                        prepared->x_matrix, prepared->b_matrix);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssExecute(CUDSS_PHASE_FACTORIZATION)", status);
  }

  int info = 0;
  size_t size_written = 0;
  status = cudssDataGet(prepared->handle, prepared->data, CUDSS_DATA_INFO, &info,
                        sizeof(info), &size_written);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssDataGet(CUDSS_DATA_INFO)", status);
  }
  if (size_written >= sizeof(info) && info != 0) {
    std::ostringstream stream_info;
    stream_info << "cuDSS factorization reported non-zero info=" << info;
    return {ffi::ErrorCode::kInternal, stream_info.str()};
  }

  status = cudssExecute(prepared->handle, CUDSS_PHASE_SOLVE, prepared->config,
                        prepared->data, prepared->a_matrix,
                        prepared->x_matrix, prepared->b_matrix);
  if (status != CUDSS_STATUS_SUCCESS) {
    return CudssError("cudssExecute(CUDSS_PHASE_SOLVE)", status);
  }

  ffi::Error event_status = EnsureRecordedEvent(*prepared, stream);
  if (!event_status.success()) {
    return event_status;
  }

  return ffi::Error::Success();
}

auto SetupSolverBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::DeviceOrdinal>()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<S32BufferR1>()
      .Arg<S32BufferR1>()
      .Attr<int64_t>("batch_size")
      .Attr<int64_t>("structure_hash")
      .Ret<ffi::Buffer<ffi::S64, 0>>();
}

auto SolveGraphBinding() {
  return ffi::Ffi::Bind()
      .Ctx<ffi::DeviceOrdinal>()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<F32BufferR2>()
      .Arg<F32BufferR2>()
      .Attr<int64_t>("token")
      .Ret<ffi::Buffer<ffi::F32, 2>>();
}

XLA_FFI_DEFINE_HANDLER(JaxCudssSetupSolver, JaxCudssSetupSolverImpl,
                       SetupSolverBinding());

XLA_FFI_DEFINE_HANDLER(JaxCudssSolveGraph, JaxCudssSolveGraphImpl,
                       SolveGraphBinding());

PyObject* Registrations(PyObject*, PyObject*) {
  PyObject* dict = PyDict_New();
  if (dict == nullptr) {
    return nullptr;
  }

  auto add_capsule = [&](const char* name, void* fn) -> bool {
    PyObject* capsule = PyCapsule_New(fn, nullptr, nullptr);
    if (capsule == nullptr) {
      return false;
    }
    const int rc = PyDict_SetItemString(dict, name, capsule);
    Py_DECREF(capsule);
    return rc == 0;
  };

  if (!add_capsule("jax_cudss_setup_solver",
                   reinterpret_cast<void*>(JaxCudssSetupSolver)) ||
      !add_capsule("jax_cudss_solve_graph",
                   reinterpret_cast<void*>(JaxCudssSolveGraph))) {
    Py_DECREF(dict);
    return nullptr;
  }

  return dict;
}

PyObject* LastProfile(PyObject*, PyObject*) {
  std::lock_guard<std::mutex> lock(ProfileMutex());
  if (!LastProfileStorage().has_value()) {
    Py_RETURN_NONE;
  }

  const SetupProfileData& profile = *LastProfileStorage();
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
      !add_bool("async_allocator_enabled", profile.async_allocator_enabled) ||
      !add_int("structure_hash", profile.structure_hash) ||
      !add_int("token", profile.token) ||
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

PyObject* ReleasePreparedSolver(PyObject*, PyObject* args) {
  long long token = 0;
  if (!PyArg_ParseTuple(args, "L", &token)) {
    return nullptr;
  }

  {
    std::lock_guard<std::mutex> registry_lock(RegistryMutex());
    auto it = PreparedByToken().find(token);
    if (it != PreparedByToken().end()) {
      if (it->second->owners > 0) {
        it->second->owners -= 1;
      }
      MaybeEvictPreparedLocked();
    }
  }

  Py_RETURN_NONE;
}

PyMethodDef kMethods[] = {
    {"registrations", Registrations, METH_NOARGS,
     PyDoc_STR("Return JAX FFI custom call registrations.")},
    {"last_profile", LastProfile, METH_NOARGS,
     PyDoc_STR("Return the last captured setup profiling record, if any.")},
    {"clear_last_profile", ClearLastProfile, METH_NOARGS,
     PyDoc_STR("Clear the last captured setup profiling record.")},
    {"release_prepared_solver", ReleasePreparedSolver, METH_VARARGS,
     PyDoc_STR("Release ownership of a prepared solver token.")},
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
