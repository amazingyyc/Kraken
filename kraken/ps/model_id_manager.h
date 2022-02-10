#pragma once

#include <mutex>
#include <string>
#include <unordered_map>

namespace kraken {

namespace io {
class CheckPoint;
}

class ModelIdManager {
  friend class io::CheckPoint;

private:
  struct TableInfo {
    uint64_t id;
    std::string name;
  };

  struct ModelInfo {
    uint64_t id;
    std::string name;

    std::unordered_map<std::string, TableInfo> table_infos;
  };

  std::mutex mu_;
  std::unordered_map<std::string, ModelInfo> model_infos_;

public:
  int32_t ApplyModelId(const std::string& model_name, uint64_t* model_id);

  int32_t ApplyTableId(const std::string& model_name,
                       const std::string& table_name, uint64_t* table_id);
};

}  // namespace kraken
