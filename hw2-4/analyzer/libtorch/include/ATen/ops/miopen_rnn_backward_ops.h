#pragma once

// @generated by torchgen/gen.py from Operator.h

#include <tuple>
#include <vector>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {
namespace _ops {


struct TORCH_API miopen_rnn_backward {
  using schema = ::std::tuple<at::Tensor,at::Tensor,at::Tensor,::std::vector<at::Tensor>> (const at::Tensor &, at::TensorList, int64_t, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, int64_t, int64_t, int64_t, bool, double, bool, bool, at::IntArrayRef, const c10::optional<at::Tensor> &, const at::Tensor &, ::std::array<bool,4>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::miopen_rnn_backward")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "miopen_rnn_backward(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask) -> (Tensor, Tensor, Tensor, Tensor[])")
  static ::std::tuple<at::Tensor,at::Tensor,at::Tensor,::std::vector<at::Tensor>> call(const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & weight_buf, const at::Tensor & hx, const c10::optional<at::Tensor> & cx, const at::Tensor & output, const c10::optional<at::Tensor> & grad_output, const c10::optional<at::Tensor> & grad_hy, const c10::optional<at::Tensor> & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const c10::optional<at::Tensor> & dropout_state, const at::Tensor & reserve, ::std::array<bool,4> output_mask);
  static ::std::tuple<at::Tensor,at::Tensor,at::Tensor,::std::vector<at::Tensor>> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & weight_buf, const at::Tensor & hx, const c10::optional<at::Tensor> & cx, const at::Tensor & output, const c10::optional<at::Tensor> & grad_output, const c10::optional<at::Tensor> & grad_hy, const c10::optional<at::Tensor> & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const c10::optional<at::Tensor> & dropout_state, const at::Tensor & reserve, ::std::array<bool,4> output_mask);
};

struct TORCH_API miopen_rnn_backward_out {
  using schema = void (const at::Tensor &, at::TensorList, int64_t, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, int64_t, int64_t, int64_t, bool, double, bool, bool, at::IntArrayRef, const c10::optional<at::Tensor> &, const at::Tensor &, ::std::array<bool,4>, at::Tensor &, at::Tensor &, at::Tensor &, at::TensorList);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::miopen_rnn_backward")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "out")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "miopen_rnn_backward.out(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!)[] out3) -> ()")
  static void call(const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & weight_buf, const at::Tensor & hx, const c10::optional<at::Tensor> & cx, const at::Tensor & output, const c10::optional<at::Tensor> & grad_output, const c10::optional<at::Tensor> & grad_hy, const c10::optional<at::Tensor> & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const c10::optional<at::Tensor> & dropout_state, const at::Tensor & reserve, ::std::array<bool,4> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::TensorList out3);
  static void redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & weight_buf, const at::Tensor & hx, const c10::optional<at::Tensor> & cx, const at::Tensor & output, const c10::optional<at::Tensor> & grad_output, const c10::optional<at::Tensor> & grad_hy, const c10::optional<at::Tensor> & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const c10::optional<at::Tensor> & dropout_state, const at::Tensor & reserve, ::std::array<bool,4> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::TensorList out3);
};

}} // namespace at::_ops
