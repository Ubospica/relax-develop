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

#include "gradient.h"

namespace tvm {
namespace relax {

Expr nll_loss_backward_pred(Expr output_grad, Expr predictions, Expr targets, Optional<Expr> weights, String reduction,
              int ignore_index) {
  ObjectPtr<NLLLossAttrs> attrs = make_object<NLLLossAttrs>();

  attrs->reduction = reduction;
  attrs->ignore_index = ignore_index;

  static const Op& op = Op::Get("relax.nll_loss_backward_pred");
  if (weights.defined()) {
    return Call(op, {std::move(output_grad), std::move(predictions), std::move(targets), std::move(weights.value())},
                Attrs{attrs}, {});
  } else {
    return Call(op, {std::move(output_grad), std::move(predictions), std::move(targets)}, Attrs{attrs}, {});
  }
}

TVM_REGISTER_GLOBAL("relax.op.nll_loss_backward_pred").set_body_typed(nll_loss_backward_pred);

StructInfo InferStructInfoNLLLossBackwardPred(const Call& call, const BlockBuilder& ctx) {
  return GetStructInfo(call->args[1]);
}

TVM_REGISTER_OP("relax.nll_loss_backward_pred")
    .set_attrs_type<NLLLossAttrs>()
    .set_num_inputs(4)
    .add_argument("output_grad", "Tensor", "The output gradient.")
    .add_argument("predictions", "Tensor", "The prediction tensor.")
    .add_argument("targets", "Tensor", "The target tensor.")
    .add_argument("weights", "Optional<Tensor>", "The weight of each target values.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoNLLLossBackwardPred);

}  // namespace relax
}  // namespace tvm
