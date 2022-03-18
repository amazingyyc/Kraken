// #include <pybind11/numpy.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>

// #include <cinttypes>
// #include <sstream>
// #include <string>
// #include <vector>

// #include "common/exception.h"
// #include "ps/info.h"
// #include "ps/initializer/initializer.h"
// #include "ps/table.h"
// #include "t/element_type.h"
// #include "t/shape.h"
// #include "watch/watcher.h"

// namespace kraken {
// namespace watch {
// namespace py {

// PYBIND11_MODULE(kraken_watcher, m) {
//   pybind11::enum_<InitializerType>(m, "InitializerType")
//       .value("kConstant", InitializerType::kConstant)
//       .value("kUniform", InitializerType::kUniform)
//       .value("kNormal", InitializerType::kNormal)
//       .value("kXavierUniform", InitializerType::kXavierUniform)
//       .value("kXavierNormal", InitializerType::kXavierNormal);

//   pybind11::enum_<OptimType>(m, "OptimType")
//       .value("kAdagrad", OptimType::kAdagrad)
//       .value("kAdam", OptimType::kAdam)
//       .value("kRMSprop", OptimType::kRMSprop)
//       .value("kSGD", OptimType::kSGD);

//   pybind11::enum_<TableType>(m, "TableType")
//       .value("kDense", TableType::kDense)
//       .value("kSparse", TableType::kSparse);

//   pybind11::enum_<DType>(m, "DType")
//       .value("kUnKnown", DType::kUnKnown)
//       .value("kBool", DType::kBool)
//       .value("kUint8", DType::kUint8)
//       .value("kInt8", DType::kInt8)
//       .value("kUint16", DType::kUint16)
//       .value("kInt16", DType::kInt16)
//       .value("kUint32", DType::kUint32)
//       .value("kInt32", DType::kInt32)
//       .value("kUint64", DType::kUint64)
//       .value("kInt64", DType::kInt64)
//       .value("kFloat16", DType::kFloat16)
//       .value("kFloat32", DType::kFloat32)
//       .value("kFloat64", DType::kFloat64);

//   pybind11::class_<ElementType>(m, "ElementType")
//       .def("name", &ElementType::Name)
//       .def("byte_width", &ElementType::ByteWidth);

//   pybind11::class_<Shape>(m, "Shape")
//       .def(pybind11::init())
//       .def(pybind11::init<const std::vector<int64_t>&>())
//       .def("ndims", &Shape::NDims)
//       .def("size", &Shape::Size)
//       .def("dim", &Shape::Dim)
//       .def("stride", &Shape::Stride)
//       .def("__repr__", [](const Shape& shape) { return shape.Str(); });

//   pybind11::class_<Tensor>(m, "Tensor", pybind11::buffer_protocol())
//       .def_buffer([](Tensor& t) -> pybind11::buffer_info {
//         ARGUMENT_CHECK(t.IsDense(), "def_buffer only support DenseTensor!");

//         void* ptr = t.Ptr();
//         pybind11::ssize_t itemsize = t.element_type().ByteWidth();
//         std::string format;

//         if (t.element_type().Is<float>()) {
//           format = pybind11::format_descriptor<float>::format();
//         } else if (t.element_type().Is<double>()) {
//           format = pybind11::format_descriptor<double>::format();
//         } else {
//           RUNTIME_ERROR("Unsupported ElementType:" << t.element_type().Name());
//         }

//         pybind11::ssize_t ndim = t.shape().NDims();
//         std::vector<pybind11::ssize_t> shape;
//         std::vector<pybind11::ssize_t> strides;

//         for (pybind11::ssize_t i = 0; i < ndim; ++i) {
//           shape.emplace_back(t.shape().Dim(i));
//           strides.emplace_back(t.shape().Stride(i) *
//                                t.element_type().ByteWidth());
//         }

//         return pybind11::buffer_info(ptr, itemsize, format, ndim, shape,
//                                      strides);
//       });

//   pybind11::class_<TableInfo>(m, "TableInfo")
//       .def_readonly("id", &TableInfo::id)
//       .def_readonly("name", &TableInfo::name)
//       .def_readonly("table_type", &TableInfo::table_type)
//       .def_readonly("element_type", &TableInfo::element_type)
//       .def_readonly("shape", &TableInfo::shape)
//       .def_readonly("dimension", &TableInfo::dimension)
//       .def_readonly("init_type", &TableInfo::init_type)
//       .def_readonly("init_conf", &TableInfo::init_conf)
//       .def("__repr__", [](const TableInfo& table_info) {
//         std::ostringstream oss;
//         oss << "<TableInfo type:";

//         if (table_info.table_type == TableType::kDense) {
//           oss << "kDense";
//         } else {
//           oss << "kSparse";
//         }

//         oss << ", id:" << table_info.id << ", name:" << table_info.name << ">";

//         return oss.str();
//       });

//   pybind11::class_<ModelInfo>(m, "ModelInfo")
//       .def_readonly("id", &ModelInfo::id)
//       .def_readonly("name", &ModelInfo::name)
//       .def_readonly("optim_type", &ModelInfo::optim_type)
//       .def_readonly("optim_conf", &ModelInfo::optim_conf)
//       .def_readonly("table_infos", &ModelInfo::table_infos)
//       .def("__repr__", [](const ModelInfo& model_info) {
//         std::ostringstream oss;
//         oss << "<ModelInfo id:" << model_info.id << ", name:" << model_info.name
//             << ", optim_type:";

//         if (model_info.optim_type == OptimType::kAdagrad) {
//           oss << "Adagrad";
//         } else if (model_info.optim_type == OptimType::kAdam) {
//           oss << "Adam";
//         } else if (model_info.optim_type == OptimType::kRMSprop) {
//           oss << "RMSprop";
//         } else if (model_info.optim_type == OptimType::kSGD) {
//           oss << "SGD";
//         } else {
//           oss << "UnKnow";
//         }

//         oss << ">";

//         return oss.str();
//       });

//   pybind11::class_<Watcher>(m, "Watcher")
//       .def(pybind11::init())
//       .def("load", &Watcher::Load)
//       .def("model_info", &Watcher::model_info)
//       .def("dense_table_infos", &Watcher::DenseTableInfos)
//       .def("exist_dense_table_infos", &Watcher::ExistDenseTableInfos)
//       .def("sparse_table_infos", &Watcher::SparseTableInfos)
//       .def("exist_sparse_table_infos", &Watcher::ExistSparseTableInfos)
//       .def("exist_sparse_table_ids", &Watcher::ExistSparseTableIds)
//       .def("is_table_exist",
//            (bool(Watcher::*)(uint64_t) const) & Watcher::IsTableExist)
//       .def("is_table_exist",
//            (bool(Watcher::*)(const std::string&) const) & Watcher::IsTableExist)
//       .def("is_dense_table_val_exist", &Watcher::IsDenseTableValExist)
//       .def("is_sparse_table_val_exist", &Watcher::IsSparseTableValExist)
//       .def("dense_table_val", &Watcher::DenseTableVal)
//       .def("sparse_table_val", &Watcher::SparseTableVal);
// }

// }  // namespace py
// }  // namespace watch
// }  // namespace kraken
