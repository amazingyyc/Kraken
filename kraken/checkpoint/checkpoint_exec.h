#pragma once

#include "checkpoint/checkpoint.h"
#include "common/info.h"
#include "common/router.h"

namespace kraken {
class Ps;

namespace io {

class CheckpointExec {
private:
  std::string saved_dir_;

  size_t max_save_count_;

public:
  CheckpointExec();

  CheckpointExec(const std::string& saved_dir, size_t max_save_count = 3);

  ~CheckpointExec() = default;

private:
  bool GetSortedShardSavedDirs(const std::string& dir,
                               std::vector<std::string>* saved_dirs) const;

  bool GetModelShardDirs(
      const std::string& dir,
      std::unordered_map<uint64_t /*node id*/, std::string>* shard_dirs);

  bool GetLatestShardDir(const std::string& dir, std::string* latest_dir);

  bool LoadDenseTables(Ps* ps, const ModelMetaData& model_mdata,
                       const std::vector<std::string>& dirs) const;

  bool LoadSparseTables(Ps* ps, const ModelMetaData& model_mdata,
                        const std::vector<std::string>& dirs) const;

public:
  // Select a latest sharde
  bool GetLatestModelMetaDataBinaryPath(const std::string& load_dir,
                                        std::string* path);

  bool Save(Ps* ps, const ModelMetaData& model_mdata);

  // Load model from the load_dir.
  // The load_dir structure must be:
  // - save_dir
  //   - shard0
  //     - time stamp folder name
  //     - time stamp folder name
  //   - shard1
  //     - time stamp folder name
  //     - time stamp folder name
  bool Load(Ps* ps, const std::string& load_dir);
};

}  // namespace io
}  // namespace kraken
