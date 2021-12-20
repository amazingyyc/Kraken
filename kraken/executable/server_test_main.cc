#include <iostream>
#include <string>

#include "common/log.h"
#include "rpc/server.h"

int32_t test(const std::string& req, std::string* reply) {
  LOG_INFO("request:" << req);

  *reply = "success:" + req;

  return 0;
}

int main(int argc, char* argv[]) {
  using namespace std::placeholders;

  kraken::Server server(5000, 8);

  server.RegisterFunc<std::string, std::string>(0, std::bind(test, _1, _2));

  server.start();

  return 0;
}
