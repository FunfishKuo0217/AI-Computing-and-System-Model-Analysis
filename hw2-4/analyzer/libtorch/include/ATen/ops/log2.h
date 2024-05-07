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



#include <ATen/ops/log2_ops.h>

namespace at {


// aten::log2(Tensor self) -> Tensor
inline at::Tensor log2(const at::Tensor & self) {
    return at::_ops::log2::call(self);
}

// aten::log2_(Tensor(a!) self) -> Tensor(a!)
inline at::Tensor & log2_(at::Tensor & self) {
    return at::_ops::log2_::call(self);
}

// aten::log2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & log2_out(at::Tensor & out, const at::Tensor & self) {
    return at::_ops::log2_out::call(self, out);
}
// aten::log2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & log2_outf(const at::Tensor & self, at::Tensor & out) {
    return at::_ops::log2_out::call(self, out);
}

}
