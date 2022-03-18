// #pragma once

// #include <atomic>
// #include <functional>
// #include <list>
// #include <memory>
// #include <queue>
// #include <shared_mutex>
// #include <string>
// #include <thread>
// #include <unordered_map>

// #include "io/checkpoint.h"

// namespace kraken {

// class Ps;

// namespace io {

// class CheckpointExecutor : public Checkpoint {
// private:
//   struct CheckPointTask {
//     Ps* ps;
//     uint64_t model_id;

//     std::function<void(uint64_t, bool)> done;
//   };

// private:
//   struct SaveInfo {
//     uint64_t index;
//     std::list<std::string> save_paths;

//     SaveInfo() : index(0) {
//     }
//   };

//   // The directory to store the model file.
//   // The real model path is: models_dir_ + model_name + timestamp.
//   std::string save_dir_;

//   // How many checkpoint will be saved.
//   size_t max_save_count_;

//   // Store the saved model path will delete the oldest if the count >
//   // max_save_count_.
//   std::unordered_map<uint64_t, SaveInfo> model_save_infos_;

//   // Will use a separate thread to dump model.
//   std::thread worker_;

//   std::atomic_bool stop_;

//   std::mutex mu_;
//   std::condition_variable cond_var_;
//   std::queue<CheckPointTask> task_que_;

// public:
//   CheckpointExecutor(const std::string& save_dir, size_t max_save_count = 3);

// private:
//   bool GetSortedShardFolders(const std::string& dir,
//                              std::vector<std::string>* partition_folders) const;

//   bool GetLatestCheckPointFolderPath(const std::string& shard_dir,
//                                      std::string* path);

//   bool Save(Ps* ps, const std::string& save_dir, uint64_t model_id);

//   void Run();

// public:
//   void Stop();

//   void Save(Ps* ps, uint64_t model_id,
//             std::function<void(uint64_t, bool)>&& done);

//   bool Load(Ps* ps, const std::string& model_dir);
// };

// }  // namespace io
// }  // namespace kraken
