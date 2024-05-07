#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/_sparse_softmax_ops.h>

namespace at {


// aten::_sparse_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
inline at::Tensor _sparse_softmax(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype=c10::nullopt) {
    return at::_ops::_sparse_softmax_int::call(self, dim, dtype);
}

// aten::_sparse_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
inline at::Tensor _sparse_softmax(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype=c10::nullopt) {
    return at::_ops::_sparse_softmax_Dimname::call(self, dim, dtype);
}

// aten::_sparse_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
inline at::Tensor _sparse_softmax(const at::Tensor & self, int64_t dim, bool half_to_float) {
    return at::_ops::_sparse_softmax::call(self, dim, half_to_float);
}

// aten::_sparse_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _sparse_softmax_out(at::Tensor & out, const at::Tensor & self, int64_t dim, bool half_to_float) {
    return at::_ops::_sparse_softmax_out::call(self, dim, half_to_float, out);
}
// aten::_sparse_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _sparse_softmax_outf(const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out) {
    return at::_ops::_sparse_softmax_out::call(self, dim, half_to_float, out);
}

}
