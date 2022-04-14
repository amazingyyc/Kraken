#include <gflags/gflags.h>

#include "ps/ps_server.h"

DEFINE_uint32(port, 50001, "The server port, default is:50000.");
DEFINE_uint32(thread_nums, 2, "The server thread_nums, default is:2.");
DEFINE_string(addr, "", "Local addr include port.");
DEFINE_string(s_addr, "", "Scheduler addr include port.");
DEFINE_string(saved_dir, "", "Model save dir.");
DEFINE_uint32(max_save_count, 3, "Max saved model count.");

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("Usage: [Options]");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  kraken::PsServer ps_server(FLAGS_port, FLAGS_thread_nums, FLAGS_addr,
                             FLAGS_s_addr, FLAGS_saved_dir,
                             FLAGS_max_save_count);
  ps_server.Start();

  return 0;
}
