#include <gflags/gflags.h>

#include "scheduler/scheduler_server.h"

DEFINE_uint32(port, 50000, "The server port, default is:50000.");

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("Usage: [Options]");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  kraken::SchedulerServer scheduler_server(FLAGS_port);
  scheduler_server.Start();

  return 0;
}
