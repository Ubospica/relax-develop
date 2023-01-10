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

#include "utils.h"

namespace tvm {
namespace relax {

bool IsNestedTensor(const StructInfo& sinfo) {
  if (sinfo->IsInstance<TensorStructInfoNode>()) {
    return true;
  } else if (const auto* tuple_sinfo = sinfo.as<TupleStructInfoNode>()) {
    return !std::any_of(tuple_sinfo->fields.begin(), tuple_sinfo->fields.end(),
                        [&](const StructInfo& field) { return !IsNestedTensor(field); });
  }
  return false;
}

bool IsNestedTensor(const Expr& expr) { return IsNestedTensor(GetStructInfo(expr)); }

}  // namespace relax
}  // namespace tvm