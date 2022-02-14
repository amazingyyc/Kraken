#include "worker/dct_emitter.h"

#include "common/exception.h"
#include "protocol/push_dense_table_prot.h"
#include "protocol/push_pull_dense_table_prot.h"

namespace kraken {

DCTEmitter::DenseBag::DenseBag(const Tensor& e_grad)
    : e_grad_(e_grad), tau_(0.0), step_(0) {
}

Tensor DCTEmitter::DenseBag::MaybeToCoo(uint64_t life_span, float eta,
                                        const Tensor& grad) {
  if (grad.Size() < 128) {
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

  std::unique_ptr<DenseBag> bag(new DenseBag(val.Clone().Zero()));

  std::unique_lock<std::shared_mutex> lock(mu_);
  dense_bags_.emplace(table_id, std::move(bag));

  return table_id;
}

void DCTEmitter::PushDenseTable(uint64_t table_id, const Tensor& grad) {
  ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");

  std::shared_lock<std::shared_mutex> lock(mu_);

  auto it = dense_bags_.find(table_id);
  if (it == dense_bags_.end()) {
    RUNTIME_ERROR("UnExpected table id:" << table_id);
  }

  Tensor coo_grad = it->second->MaybeToCoo(life_span_, eta_, grad);

  size_t server_id = DenseTableRouter(model_id_, table_id);

  PushDenseTableRequest req;
  req.model_id = model_id_;
  req.table_id = table_id;
  req.grad = coo_grad;
  req.lr = lr_.load();

  auto callback = [](int32_t ecode, PushDenseTableResponse& /*not care*/) {
    RPC_CALL(ecode);
  };

  clients_[server_id]->CallAsync<PushDenseTableRequest, PushDenseTableResponse>(
      RPCFuncType::kPushDenseTableType, req, std::move(callback));
}

Tensor DCTEmitter::PushPullDenseTable(uint64_t table_id, const Tensor& grad) {
  ARGUMENT_CHECK(initialized_.load(), "Emitter not initialize.");

  std::shared_lock<std::shared_mutex> lock(mu_);

  auto it = dense_bags_.find(table_id);
  if (it == dense_bags_.end()) {
    RUNTIME_ERROR("UnExpected table id:" << table_id);
  }

  Tensor coo_grad = it->second->MaybeToCoo(life_span_, eta_, grad);

  size_t server_id = DenseTableRouter(model_id_, table_id);

  PushPullDenseTableRequest req;
  PushPullDenseTableResponse rsp;

  req.model_id = model_id_;
  req.table_id = table_id;
  req.grad = coo_grad;
  req.lr = lr_.load();

  RPC_CALL(clients_[server_id]->Call(RPCFuncType::kPushPullDenseTableType, req,
                                     &rsp));

  return rsp.val;
}

}  // namespace kraken
