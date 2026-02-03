#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include <cstdio>

extern "C" {

void* trt_create_logger();
void trt_destroy_runtime(void* runtime);
int trt_engine_get_nb_bindings(void* engine);
const char* trt_engine_get_binding_name(void* engine, int index);
int trt_engine_binding_is_input(void* engine, int index);
int trt_engine_get_binding_dimensions(void* engine, int index, long long* dims, int max_dims);
void trt_destroy_engine(void* engine);
void* trt_engine_create_execution_context(void* engine);
int trt_context_enqueue_v2(void* context, void* const* bindings, void* stream);
void trt_destroy_execution_context(void* context);
void* trt_runtime_deserialize_cuda_engine(void* runtime, const void* data, unsigned long long size);
void* trt_engine_serialize(void* engine, unsigned long long* size);
void trt_free_serialized(void* data);
void* trt_builder_create_network(void* builder, int flags);
void* trt_builder_create_builder_config(void* builder);
void* trt_builder_build_serialized_network(void* builder, void* network, void* config, unsigned long long* size);
void trt_destroy_builder(void* builder);
void trt_destroy_network(void* network);
void trt_destroy_builder_config(void* config);
void trt_config_set_max_workspace_size(void* config, unsigned long long size);
void trt_config_set_flag(void* config, int flag);
void trt_config_set_dla_core(void* config, int core);
int trt_onnx_parser_parse_from_file(void* parser, const char* filepath, int verbosity);
void trt_destroy_onnx_parser(void* parser);

}

namespace {
class SimpleLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING && msg) {
            fprintf(stderr, "[TRT] %s\n", msg);
        }
    }
};

SimpleLogger gLogger;
}

extern "C" {

void* trt_create_logger() {
    return reinterpret_cast<void*>(&gLogger);
}

void trt_destroy_runtime(void* runtime) {
    if (runtime) {
        auto* rt = reinterpret_cast<nvinfer1::IRuntime*>(runtime);
        delete rt;
    }
}

int trt_engine_get_nb_bindings(void* engine) {
    auto* eng = reinterpret_cast<nvinfer1::ICudaEngine*>(engine);
    return eng ? eng->getNbIOTensors() : 0;
}

const char* trt_engine_get_binding_name(void* engine, int index) {
    auto* eng = reinterpret_cast<nvinfer1::ICudaEngine*>(engine);
    return eng ? eng->getIOTensorName(index) : nullptr;
}

int trt_engine_binding_is_input(void* engine, int index) {
    auto* eng = reinterpret_cast<nvinfer1::ICudaEngine*>(engine);
    if (!eng) return 0;
    auto name = eng->getIOTensorName(index);
    if (!name) return 0;
    return eng->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT ? 1 : 0;
}

int trt_engine_get_binding_dimensions(void* engine, int index, long long* dims, int max_dims) {
    auto* eng = reinterpret_cast<nvinfer1::ICudaEngine*>(engine);
    if (!eng || !dims || max_dims <= 0) return 0;
    auto name = eng->getIOTensorName(index);
    if (!name) return 0;
    auto d = eng->getTensorShape(name);
    int n = d.nbDims;
    if (n > max_dims) n = max_dims;
    for (int i = 0; i < n; ++i) dims[i] = static_cast<long long>(d.d[i]);
    return n;
}

void trt_destroy_engine(void* engine) {
    if (engine) {
        auto* eng = reinterpret_cast<nvinfer1::ICudaEngine*>(engine);
        delete eng;
    }
}

void* trt_engine_create_execution_context(void* engine) {
    auto* eng = reinterpret_cast<nvinfer1::ICudaEngine*>(engine);
    if (!eng) return nullptr;
    return eng->createExecutionContext();
}

int trt_context_enqueue_v2(void* context, void* const* bindings, void* stream) {
    auto* ctx = reinterpret_cast<nvinfer1::IExecutionContext*>(context);
    if (!ctx) return 0;
    auto& eng = ctx->getEngine();
    int nb = eng.getNbIOTensors();
    if (!bindings || nb <= 0) return 0;

    int idx = 0;
    for (int i = 0; i < nb; ++i) {
        auto name = eng.getIOTensorName(i);
        if (!name) continue;
        if (eng.getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            ctx->setTensorAddress(name, bindings[idx++]);
        }
    }
    for (int i = 0; i < nb; ++i) {
        auto name = eng.getIOTensorName(i);
        if (!name) continue;
        if (eng.getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
            ctx->setTensorAddress(name, bindings[idx++]);
        }
    }

    return ctx->enqueueV3(reinterpret_cast<cudaStream_t>(stream)) ? 1 : 0;
}

void trt_destroy_execution_context(void* context) {
    if (context) {
        auto* ctx = reinterpret_cast<nvinfer1::IExecutionContext*>(context);
        delete ctx;
    }
}

void* trt_runtime_deserialize_cuda_engine(void* runtime, const void* data, unsigned long long size) {
    auto* rt = reinterpret_cast<nvinfer1::IRuntime*>(runtime);
    if (!rt || !data || size == 0) return nullptr;
    return rt->deserializeCudaEngine(data, static_cast<size_t>(size));
}

void* trt_engine_serialize(void* engine, unsigned long long* size) {
    auto* eng = reinterpret_cast<nvinfer1::ICudaEngine*>(engine);
    if (!eng || !size) return nullptr;
    auto* mem = eng->serialize();
    if (!mem) return nullptr;
    *size = static_cast<unsigned long long>(mem->size());
    return mem;
}

void trt_free_serialized(void* data) {
    auto* mem = reinterpret_cast<nvinfer1::IHostMemory*>(data);
    if (mem) delete mem;
}

void* trt_builder_create_network(void* builder, int flags) {
    auto* b = reinterpret_cast<nvinfer1::IBuilder*>(builder);
    if (!b) return nullptr;
    return b->createNetworkV2(static_cast<uint32_t>(flags));
}

void* trt_builder_create_builder_config(void* builder) {
    auto* b = reinterpret_cast<nvinfer1::IBuilder*>(builder);
    if (!b) return nullptr;
    return b->createBuilderConfig();
}

void* trt_builder_build_serialized_network(void* builder, void* network, void* config, unsigned long long* size) {
    auto* b = reinterpret_cast<nvinfer1::IBuilder*>(builder);
    auto* net = reinterpret_cast<nvinfer1::INetworkDefinition*>(network);
    auto* cfg = reinterpret_cast<nvinfer1::IBuilderConfig*>(config);
    if (!b || !net || !cfg || !size) return nullptr;
    auto* mem = b->buildSerializedNetwork(*net, *cfg);
    if (!mem) return nullptr;
    *size = static_cast<unsigned long long>(mem->size());
    return mem;
}

void trt_destroy_builder(void* builder) {
    if (builder) delete reinterpret_cast<nvinfer1::IBuilder*>(builder);
}

void trt_destroy_network(void* network) {
    if (network) delete reinterpret_cast<nvinfer1::INetworkDefinition*>(network);
}

void trt_destroy_builder_config(void* config) {
    if (config) delete reinterpret_cast<nvinfer1::IBuilderConfig*>(config);
}

void trt_config_set_max_workspace_size(void* config, unsigned long long size) {
    auto* cfg = reinterpret_cast<nvinfer1::IBuilderConfig*>(config);
    if (!cfg) return;
    cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, static_cast<size_t>(size));
}

void trt_config_set_flag(void* config, int flag) {
    auto* cfg = reinterpret_cast<nvinfer1::IBuilderConfig*>(config);
    if (!cfg) return;
    cfg->setFlag(static_cast<nvinfer1::BuilderFlag>(flag));
}

void trt_config_set_dla_core(void* config, int core) {
    auto* cfg = reinterpret_cast<nvinfer1::IBuilderConfig*>(config);
    if (!cfg) return;
    cfg->setDLACore(core);
}

int trt_onnx_parser_parse_from_file(void* parser, const char* filepath, int verbosity) {
    auto* p = reinterpret_cast<nvonnxparser::IParser*>(parser);
    if (!p || !filepath) return 0;
    return p->parseFromFile(filepath, verbosity) ? 1 : 0;
}

void trt_destroy_onnx_parser(void* parser) {
    if (parser) delete reinterpret_cast<nvonnxparser::IParser*>(parser);
}

}
