#pragma once

#include <string>

#include "ps/optim.h"

namespace kraken {

/**
 * \brief This is a pecial struct represent a Row in matrix.
 */
struct IndepVector {
  int64_t indice;
  Tensor val;
};

class Table {
protected:
  /**
   * \brief The optimizer.
   */
  Optim* optim_;

  uint64_t id_;
  std::string name_;

  Table(Optim* optim, uint64_t id, const std::string& name);

public:
  virtual ~Table() = default;

  uint64_t Id() const;

  const std::string Name() const;

  /**
   * \brief Push dense parameter for PS server.
   *
   * \param grad the parameter gradient.
   * \param lr learning rate.
   * \return true Update success.
   * \return false Update fail: like has different shape.
   */
  virtual bool Push(const Tensor& grad, float lr);

  /**
   * \brief Pull variable from this table.
   *
   * \param var store the table's variable, deep copy.
   * \return true Copy success.
   * \return false Fail.
   */
  virtual bool Pull(Tensor* var);

  virtual bool Push(const std::vector<IndepVector>& grads, float lr);

  /**
   * \brief Use for pull Sparse embedding.
   *
   * \param indices Which row will be pulled.
   * \param vars Store the result.
   * \return true Pull success.
   * \return false Pull fail.
   */
  virtual bool Pull(const std::vector<int64_t>& indices,
                    std::vector<Tensor>* vars);
};

}  // namespace kraken
