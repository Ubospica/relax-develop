/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "gradient_ops.h"

namespace tvm {
namespace relax {

Expr nll_loss_backward(Expr output_grad, Expr predictions, Expr targets,
                            Optional<Expr> weights, String reduction, int ignore_index) {
  ObjectPtr<NLLLossAttrs> attrs = make_object<NLLLossAttrs>();

  attrs->reduction = reduction;
  attrs->ignore_index = ignore_index;

  static const Op& op = Op::Get("relax.nll_loss_backward");
  if (weights.defined()) {
    return Call(op,
                {std::move(output_grad), std::move(predictions), std::move(targets),
                 std::move(weights.value())},
                Attrs{attrs}, {});
  } else {
    return Call(op, {std::move(output_grad), std::move(predictions), std::move(targets)},
                Attrs{attrs}, {});
  }
}

TVM_REGISTER_GLOBAL("relax.op.nll_loss_backward").set_body_typed(nll_loss_backward);

StructInfo InferStructInfoNLLLossBackwardPred(const Call& call, const BlockBuilder& ctx) {
  return GetStructInfo(call->args[1]);
}

TVM_REGISTER_OP("relax.nll_loss_backward")
    .set_attrs_type<NLLLossAttrs>()
    .set_num_inputs(4)
    .add_argument("output_grad", "Tensor", "The output gradient.")
    .add_argument("predictions", "Tensor", "The prediction tensor.")
    .add_argument("targets", "Tensor", "The target tensor.")
    .add_argument("weights", "Optional<Tensor>", "The weight of each target values.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoNLLLossBackwardPred);


Expr conv2d_backward_data(Expr output_grad, Expr data, Expr weight, Array<IntImm> strides, Array<IntImm> padding,
                     Array<IntImm> dilation, int groups, String data_layout, String kernel_layout,
                     Optional<String> out_layout, DataType out_dtype) {
  auto attrs = make_object<Conv2DAttrs>();
  attrs->strides = ConvertIntImmToInt64(strides);
  attrs->padding = ConvertIntImmToInt64(padding);
  attrs->dilation = ConvertIntImmToInt64(dilation);
  attrs->groups = groups;
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = out_layout.value_or(data_layout);
  attrs->out_dtype = std::move(out_dtype);
  const Op& op = Op::Get("relax.conv2d_backward_data");
  return Call(op, {std::move(output_grad), std::move(data), std::move(weight)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.conv2d_backward_data").set_body_typed(conv2d_backward_data);

StructInfo InferStructInfoConv2dBackwardData(const Call& call, const BlockBuilder& ctx) {
  return GetStructInfo(call->args[1]);
}

TVM_REGISTER_OP("relax.conv2d_backward_data")
    .set_num_inputs(3)
    .add_argument("output_grad", "Tensor", "The output gradient.")
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_attrs_type<Conv2DAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoConv2dBackwardData);


Expr conv2d_backward_weight(Expr output_grad, Expr data, Expr weight, Array<IntImm> strides, Array<IntImm> padding,
                     Array<IntImm> dilation, int groups, String data_layout, String kernel_layout,
                     Optional<String> out_layout, DataType out_dtype) {
  auto attrs = make_object<Conv2DAttrs>();
  attrs->strides = ConvertIntImmToInt64(strides);
  attrs->padding = ConvertIntImmToInt64(padding);
  attrs->dilation = ConvertIntImmToInt64(dilation);
  attrs->groups = groups;
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = out_layout.value_or(data_layout);
  attrs->out_dtype = std::move(out_dtype);
  const Op& op = Op::Get("relax.conv2d_backward_weight");
  return Call(op, {std::move(output_grad), std::move(data), std::move(weight)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.conv2d_backward_weight").set_body_typed(conv2d_backward_weight);

StructInfo InferStructInfoConv2dBackwardWeight(const Call& call, const BlockBuilder& ctx) {
  return GetStructInfo(call->args[2]);
}

TVM_REGISTER_OP("relax.conv2d_backward_weight")
    .set_num_inputs(3)
    .add_argument("output_grad", "Tensor", "The output gradient.")
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_attrs_type<Conv2DAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoConv2dBackwardWeight);

Expr max_pool2d_backward(Expr output_grad, Expr data, Array<IntImm> pool_size, Array<IntImm> strides, Array<IntImm> padding,
                Array<IntImm> dilation, bool ceil_mode, String layout,
                Optional<String> out_layout) {
  auto attrs = make_object<MaxPool2DAttrs>();
  attrs->pool_size = std::move(pool_size);
  attrs->strides = ConvertIntImmToInt64(strides);
  attrs->padding = ConvertIntImmToInt64(padding);
  attrs->dilation = ConvertIntImmToInt64(dilation);
  attrs->ceil_mode = ceil_mode;
  attrs->layout = layout;
  attrs->out_layout = out_layout.value_or(layout);
  static const Op& op = Op::Get("relax.max_pool2d_backward");
  return Call(op, {std::move(output_grad), std::move(data)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.max_pool2d_backward").set_body_typed(max_pool2d_backward);

StructInfo InferStructInfoMaxPool2DBackward(const Call& call, const BlockBuilder& ctx) {
  return GetStructInfo(call->args[1]);
}

TVM_REGISTER_OP("relax.max_pool2d_backward")
    .set_num_inputs(2)
    .add_argument("output_grad", "Tensor", "The output gradient.")
    .add_argument("data", "Tensor", "The input tensor")
    .set_attrs_type<MaxPool2DAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoMaxPool2DBackward);

}  // namespace relax
}  // namespace tvm
