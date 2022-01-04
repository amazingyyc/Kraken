#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <string>
#include <unordered_map>

#include "ps/optim/optim.h"
#include "pybind11/pytorch.h"

namespace kraken {
namespace py {

PYBIND11_MODULE(kraken_native, m) {
  pybind11::enum_<OptimType>(m, "OptimType")
      .value("kAdagrad", kAdagrad)
      .value("kAdam", kAdam)
      .value("kRMSprop", kRMSprop)
      .value("kSGD", kSGD)
      .export_values();

  m.def("initialize", &Initialize, pybind11::arg("addrs"));

  m.def("stop", &Stop);

  m.def("register_model", &RegisterModel, pybind11::arg("model_name"),
        pybind11::arg("optim_type"),
        pybind11::arg("optim_conf") =
            std::unordered_map<std::string, std::string>());

  m.def("update_lr", &UpdateLR, pybind11::arg("lr"));

  m.def("register_dense_table", &RegisterDenseTable, pybind11::arg("name"),
        pybind11::arg("val"));

  m.def("register_sparse_table", &RegisterSparseTable, pybind11::arg("name"),
        pybind11::arg("dimension"), pybind11::arg("dtype"));

  m.def("push_dense_table", &PushDenseTable, pybind11::arg("table_id"),
        pybind11::arg("grad"));

  m.def("pull_dense_table", &PullDenseTable, pybind11::arg("table_id"));

  m.def("push_pull_dense_table", &PushPullDenseTable, pybind11::arg("table_id"),
        pybind11::arg("grad"));

  m.def("push_sparse_table", &PushSparseTable, pybind11::arg("table_id"),
        pybind11::arg("indices"), pybind11::arg("grads"));

  m.def("pull_sparse_table", &PullSparseTable, pybind11::arg("table_id"),
        pybind11::arg("indices"));
}

}  // namespace py
}  // namespace kraken
