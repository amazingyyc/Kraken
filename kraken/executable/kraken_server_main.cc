#include <gflags/gflags.h>

#include "ps/ps_server.h"

DEFINE_uint32(port, 50000, "The server port, default is:50000.");
DEFINE_uint32(thread_nums, 2, "The server thread_nums, default is:2.");
DEFINE_uint32(shard_num, 1, "How many Ps server in current cluster.");
DEFINE_uint32(shard_id, 0, "Current server's id.");

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("Usage: [Options]");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  kraken::PsServer ps_server(FLAGS_port, FLAGS_thread_nums, FLAGS_shard_num,
                             FLAGS_shard_id);
  ps_server.Start();

  return 0;
}
