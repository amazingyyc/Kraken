#pragma once

#include <fstream>
#include <string>

#include "common/iwriter.h"

namespace kraken {
namespace io {

class FileWriter : public IWriter {
private:
  std::string file_path_;

  std::ofstream fs_;

public:
  FileWriter(const std::string& file_path);

  ~FileWriter();

  bool IsOpen() const;

  bool Write(const char* ptr, size_t size) override;
};

}  // namespace io
}  // namespace kraken
