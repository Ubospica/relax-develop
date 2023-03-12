# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
"""Utility functions for relax training."""

from typing import Optional

import tvm
from ..expr import Function
from . import _ffi_api


def AppendLoss(
    func_name: str,
    loss_function: Function,
    num_backbone_outputs: int = 1,
    new_func_name: Optional[str] = None,
) -> tvm.ir.transform.Pass:
    """Append the loss function to the backbone function specified by `func_name`. Generally, the
    loss function is generated by instances of `relax.training.Loss`.

    The backbone function and the loss function should satisfy a few restrictions:
    - Both backbone and loss should contain exactly one DataflowBlock.
    - Backbone should return either one Var, or a tuple of Vars
    - Loss should return a scalar(0-dim Tensor) Var

    They should be like:

    .. code-block:: python
        @R.function
        def backbone(input_of_backbones, input_of_states):
            # input_of_backbones, input_of_states denote a number of parameters
            #
            # You can return the updated model states in `output_of_states`, such as `running_mean`
            # and `running_var` in `batch_norm`
            #
            # input_of_states, output_of_states can be empty
            return prediction_outputs, output_of_states

        @R.function
        def loss(backbone_prediction_outputs, targets):
            # backbone_prediction_outputs, targets denote a number of parameters
            # loss should be a scalar Var
            return loss

    The length of `prediction_outputs` and `backbone_prediction_outputs` is specified by
    `num_backbone_outputs`.

    The appended result contains only one DataflowBlock containing all bindings in backbone and
    loss. It will be like:

    .. code-block:: python
        @R.function
        def backbone_loss(input_of_backbones, input_of_states, targets):
            # backbone_loss contains all bindings in backbone and loss
            return loss, output_of_states

    Parameters
    ----------
    func_name : str
        The name of the backbone function in the IRModule.

    loss_func : Function
        The loss function.

    num_backbone_outputs : int
        Specify the number of `prediction_outputs` of the backbone function. Default: 1.

    new_func_name : Optional[str]
        Specify the name of the appended result. If is is not specified, the name will be
        `func_name + "_loss"`.

    Returns
    -------
    ret : Function
        The result function.

    Examples
    --------
    .. code-block:: python
        @I.ir_module
        class Module
            @R.function
            def predict(x: R.Tensor((2, 4), "float32"), y: R.Tensor((2, 4), "float32")):
                with R.dataflow():
                    out = R.add(x, y)
                    R.output(out)
                return out

        @R.function
        def loss(predictions: R.Tensor((2, 4), "float32"), labels: R.Tensor((2, 4), "float32")):
            with R.dataflow():
                lv = R.subtract(predictions, labels)
                lv1 = R.multiply(lv, lv)
                gv = R.sum(lv1)
                R.output(gv)
            return gv

        expected = AppendLoss("predict", loss)(Module)
        expected.show()

    Will get

    .. code-block:: python
        @I.ir_module
        class Module
            @R.function
            def predict(x: R.Tensor((2, 4), "float32"), y: R.Tensor((2, 4), "float32")):
                with R.dataflow():
                    out = R.add(x, y)
                    R.output(out)
                return out

            @R.function
            def predict_loss(x: R.Tensor((2, 4), "float32"), y: R.Tensor((2, 4), "float32"),
                             labels: R.Tensor((2, 4), "float32")) -> R.Tensor((), "float32"):
                with R.dataflow():
                    out: R.Tensor((2, 4), "float32") = R.add(x, y)
                    lv: R.Tensor((2, 4), "float32") = R.subtract(out, labels)
                    lv1: R.Tensor((2, 4), "float32") = R.multiply(lv, lv)
                    gv: R.Tensor((), "float32") = R.sum(lv1)
                    R.output(gv)
                return gv

    Notes
    -----
    This util can be replaced if we have inline pass. It is equivalent to inline a tail call in
    some sense.
    """

    return _ffi_api.AppendLoss(  # type: ignore
        func_name,
        loss_function,
        num_backbone_outputs,
        new_func_name,
    )
