#include "worker/dct_emitter.h"

#include "common/error_code.h"
#include "common/exception.h"
#include "common/log.h"
#include "common/utils.h"
#include "protocol/push_dense_table_prot.h"
#include "protocol/rpc_func_type.h"

namespace kraken {

DCTEmitter::DenseBag::DenseBag(const Tensor& e_grad)
    : e_grad_(e_grad), tau_(0.0), step_(0) {
}

Tensor DCTEmitter::DenseBag::MaybeToCoo(uint64_t life_span, float eta,
                                        const Tensor& grad) {
  if (grad.Size() < 256) {
    return grad;
  }

  // Fix gradient.
  Tensor f_grad = grad + e_grad_;

  if (step_ % life_span == 0) {
    // Update tau.
    int64_t k = (int64_t)(f_grad.Size() * (1.0 - eta));

    Tensor topk = f_grad.Abs(false).TopK(k);
    tau_ = topk[-1];
  }

  // convert f_grad to SparseCoo by tau_.
  Tensor coo_grad = f_grad.ToCoo(tau_);

  // keep ErrorGrad.
  e_grad_ = f_grad.LtKeep(tau_);

  // update step.
  step_ += 1;

  return coo_grad;
}

DCTEmitter::DCTEmitter(uint64_t life_span, float eta)
    : Emitter(EmitterType::kDCT), life_span_(life_span), eta_(eta) {
}

uint64_t DCTEmitter::RegisterDenseTable(const std::string& name,
                                        const Tensor& val) {
  uint64_t table_id = Emitter::RegisterDenseTable(name, val);

  dense_bags_.emplace(table_id, DenseBag(val.Clone().Zero()));

  return table_id;
}

void DCTEmitter::PushDenseTable(uint64_t table_id, const Tensor& grad) {
  ARGUMENT_CHECK(initialized_, "Emitter not initialize.");

  auto it = dense_bags_.find(table_id);
  if (it == dense_bags_.end()) {
    RUNTIME_ERROR("UnExpected table id:" << table_id);
  }

  size_t node_id = router_.Hit(utils::Hash(table_id));

  Tensor coo_grad = it->second.MaybeToCoo(life_span_, eta_, grad);

  PushDenseTableRequest req;
  req.router_version = router_.version();
  req.table_id = table_id;
  req.grad = coo_grad;
  req.lr = lr_;

  auto callback = [](int32_t ecode, PushDenseTableResponse& /*not care*/) {
    if (ecode != ErrorCode::kSuccess) {
      LOG_WARNING("PushDenseTable got error code:"
                  << ecode << ", msg:" << ErrorCode::Msg(ecode)
                  << ", we not handle Push error!");
    }
  };

  clients_.CallAsync<PushDenseTableRequest, PushDenseTableResponse>(
      node_id, RPCFuncType::kPushDenseTableType, req, std::move(callback));
}

}  // namespace kraken
