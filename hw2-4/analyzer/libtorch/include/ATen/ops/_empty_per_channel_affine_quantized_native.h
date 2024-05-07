#pragma once

// @generated by torchgen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>


namespace at {
namespace native {
TORCH_API at::Tensor & _empty_per_channel_affine_quantized_out_symint(c10::SymIntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, c10::optional<at::MemoryFormat> memory_format, at::Tensor & out);
TORCH_API at::Tensor empty_per_channel_affine_quantized_other_backends_stub(at::IntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, c10::optional<at::ScalarType> dtype={}, c10::optional<at::Layout> layout={}, c10::optional<at::Device> device={}, c10::optional<bool> pin_memory={}, c10::optional<at::MemoryFormat> memory_format=MemoryFormat::Contiguous);
TORCH_API at::Tensor empty_per_channel_affine_quantized(at::IntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, c10::optional<at::ScalarType> dtype={}, c10::optional<at::Layout> layout={}, c10::optional<at::Device> device={}, c10::optional<bool> pin_memory={}, c10::optional<at::MemoryFormat> memory_format=MemoryFormat::Contiguous);
} // namespace native
} // namespace at
