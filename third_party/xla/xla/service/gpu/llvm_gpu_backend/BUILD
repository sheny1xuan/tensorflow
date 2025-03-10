load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "@local_config_sycl//sycl:build_defs.bzl",
    "if_sycl_is_configured",
)
load(
    "@local_tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load("//xla:xla.bzl", "xla_cc_test")
load(
    "//xla/tsl:tsl.bzl",
    "if_google",
    "internal_visibility",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "llvm_gpu_backend",
    srcs = [
        "gpu_backend_lib.cc",
    ],
    hdrs = [
        "gpu_backend_lib.h",
    ],
    local_defines = if_cuda_is_configured([
        "GOOGLE_CUDA=1",
    ]) + if_rocm_is_configured(["TENSORFLOW_USE_ROCM=1"]),
    deps = [
        ":load_ir_module",
        ":nvptx_libdevice_path",
        ":utils",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/service/gpu:metrics",
        "//xla/service/llvm_ir:llvm_command_line_options",
        "//xla/service/llvm_ir:llvm_type_conversion_util",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:semantic_version",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:BitReader",
        "@llvm-project//llvm:BitWriter",
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:NVPTXCodeGen",  # buildcleaner: keep
        "@llvm-project//llvm:ObjCARC",  # buildcleaner: keep
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Scalar",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//mlir:NVVMDialect",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:cuda_root_path",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:random",
        "@local_tsl//tsl/platform:rocm_rocdl_path",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:scoped_annotation",
        "@local_tsl//tsl/profiler/lib:traceme",
    ] + if_cuda_is_configured([
        "//xla/stream_executor/cuda:cuda_asm_compiler",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
        "@llvm-project//llvm:AMDGPUCodeGen",
        "@llvm-project//llvm:AMDGPUAsmParser",
    ]) + if_sycl_is_configured([
        "@spirv_llvm_translator//:spirv_llvm_translator",
    ]),
)

cc_library(
    name = "load_ir_module",
    hdrs = ["load_ir_module.h"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
    ] + if_google(
        ["//xla/service/gpu/llvm_gpu_backend/google:load_ir_module"],
        ["//xla/service/gpu/llvm_gpu_backend/default:load_ir_module"],
    ),
)

cc_library(
    name = "nvptx_libdevice_path",
    hdrs = ["nvptx_libdevice_path.h"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
    ] + if_google(
        ["//xla/service/gpu/llvm_gpu_backend/google:nvptx_libdevice_path"],
        ["//xla/service/gpu/llvm_gpu_backend/default:nvptx_libdevice_path"],
    ),
)

cc_library(
    name = "nvptx_utils",
    srcs = ["nvptx_utils.cc"],
    hdrs = ["nvptx_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:cuda_root_path",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "gpu_backend_lib_test",
    size = "small",
    srcs = ["gpu_backend_lib_test.cc"],
    deps = [
        ":llvm_gpu_backend",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:semantic_version",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "load_ir_module_test",
    size = "small",
    srcs = ["load_ir_module_test.cc"],
    data = [
        "tests_data/saxpy.ll",
    ],
    deps = [
        ":load_ir_module",
        "//xla/tests:xla_internal_test_main",
        "@llvm-project//llvm:ir_headers",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "nvptx_utils_test",
    srcs = ["nvptx_utils_test.cc"],
    deps = [
        ":nvptx_utils",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "utils_test",
    size = "small",
    srcs = ["utils_test.cc"],
    deps = [
        ":utils",
        "//xla/tests:xla_internal_test_main",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:test",
    ],
)
