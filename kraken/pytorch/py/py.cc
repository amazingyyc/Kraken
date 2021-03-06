#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <string>
#include <unordered_map>

#include "ps/initializer/initializer.h"
#include "ps/optim/optim.h"
#include "pytorch/py/jagged_ops.h"
#include "pytorch/py/pytorch.h"
#include "rpc/protocol.h"
#include "worker/emitter.h"

namespace kraken {
namespace py {

PYBIND11_MODULE(kraken_native, m) {
  pybind11::enum_<OptimType>(m, "OptimType")
      .value("kAdagrad", OptimType::kAdagrad)
      .value("kAdam", OptimType::kAdam)
      .value("kRMSprop", OptimType::kRMSprop)
      .value("kSGD", OptimType::kSGD);

  pybind11::enum_<InitializerType>(m, "InitializerType")
      .value("kConstant", InitializerType::kConstant)
      .value("kUniform", InitializerType::kUniform)
      .value("kNormal", InitializerType::kNormal)
      .value("kXavierUniform", InitializerType::kXavierUniform)
      .value("kXavierNormal", InitializerType::kXavierNormal);

  pybind11::enum_<CompressType>(m, "CompressType")
      .value("kDefault", CompressType::kNo)
      .value("kDCT", CompressType::kSnappy);

  pybind11::enum_<EmitterType>(m, "EmitterType")
      .value("kDefault", EmitterType::kDefault)
      .value("kDCT", EmitterType::kDCT);

  m.def("initialize", &Initialize, pybind11::arg("s_addr"),
        pybind11::arg("emitter_type") = EmitterType::kDefault,
        pybind11::arg("life_span") = 1000, pybind11::arg("eta") = 0.75);

  m.def("stop", &Stop);

  m.def("init_model", &InitModel, pybind11::arg("model_name"),
        pybind11::arg("optim_type"),
        pybind11::arg("optim_conf") =
            std::unordered_map<std::string, std::string>());

  m.def("update_lr", &UpdateLR, pybind11::arg("lr"));

  m.def("register_dense_table", &RegisterDenseTable, pybind11::arg("name"),
        pybind11::arg("val"));

  m.def("register_sparse_table", &RegisterSparseTable, pybind11::arg("name"),
        pybind11::arg("dimension"), pybind11::arg("dtype"),
        pybind11::arg("init_type"), pybind11::arg("init_conf"));

  m.def("pull_dense_table", &PullDenseTable, pybind11::arg("table_id"));

  m.def("combine_pull_dense_table", &CombinePullDenseTable,
        pybind11::arg("table_ids"));

  m.def("push_dense_table", &PushDenseTable, pybind11::arg("table_id"),
        pybind11::arg("grad"));

  m.def("pull_sparse_table", &PullSparseTable, pybind11::arg("table_id"),
        pybind11::arg("indices"));

  m.def("combine_pull_sparse_table", &CombinePullSparseTable,
        pybind11::arg("table_ids"), pybind11::arg("indices"));

  m.def("push_sparse_table", &PushSparseTable, pybind11::arg("table_id"),
        pybind11::arg("indices"), pybind11::arg("grad"));

  m.def("combine_push_sparse_table", &CombinePushSparseTable,
        pybind11::arg("table_ids"), pybind11::arg("indices"),
        pybind11::arg("grads"));

  m.def("try_save_model", &TrySaveModel);

  m.def("try_load_model_blocked", &TryLoadModelBlocked);

  // Jagged tensor Sum op.
  m.def("jagged_sum_forward", &jagged::SumForward, pybind11::arg("values"),
        pybind11::arg("offsets"), pybind11::arg("patch_value"));
  m.def("jagged_sum_backward", &jagged::SumBackward, pybind11::arg("offsets"),
        pybind11::arg("grads"));

  // Jagged tensor Mean op.
  m.def("jagged_mean_forward", &jagged::MeanForward, pybind11::arg("values"),
        pybind11::arg("offsets"), pybind11::arg("patch_value"));
  m.def("jagged_mean_backward", &jagged::MeanBackward, pybind11::arg("offsets"),
        pybind11::arg("grads"));
}

}  // namespace py
}  // namespace kraken
