#pragma once

#include <fstream>
#include <string>

#include "common/ireader.h"

namespace kraken {
namespace io {

class FileReader : public IReader {
private:
  std::string file_path_;

  std::ifstream fs_;

public:
  FileReader(const std::string& file_path);

  ~FileReader();

  bool IsOpen() const;

  bool Read(void* target, size_t size) override;
};

}  // namespace io
}  // namespace kraken
