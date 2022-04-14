#include "checkpoint/file_writer.h"

namespace kraken {
namespace io {

FileWriter::FileWriter(const std::string& file_path) : file_path_(file_path) {
  fs_.open(file_path_, std::ios::out | std::ios::trunc | std::ios::binary);
}

FileWriter::~FileWriter() {
  fs_.close();
}

bool FileWriter::IsOpen() const {
  return fs_.is_open();
}

bool FileWriter::Write(const char* ptr, size_t size) {
  if (fs_.write(ptr, size)) {
    return true;
  }

  return false;
}

}  // namespace io
}  // namespace kraken
