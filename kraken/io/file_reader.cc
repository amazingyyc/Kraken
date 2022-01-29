#include "io/file_reader.h"

namespace kraken {
namespace io {

FileReader::FileReader(const std::string& file_path) : file_path_(file_path) {
  fs_.open(file_path_, std::ios::in | std::ios::binary);
}

FileReader::~FileReader() {
  fs_.close();
}

bool FileReader::IsOpen() const {
  return fs_.is_open();
}

bool FileReader::Read(void* target, size_t size) {
  if (fs_.read((char*)target, size)) {
    return true;
  }

  return false;
}

}  // namespace io
}  // namespace kraken
