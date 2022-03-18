# start ps
./ps_server --port=50001 --addr=127.0.0.1:50001 --s_addr=127.0.0.1:50000
./ps_server --port=50002 --addr=127.0.0.1:50002 --s_addr=127.0.0.1:50000
./ps_server --port=50003 --addr=127.0.0.1:50003 --s_addr=127.0.0.1:50000


# start scheduler server
./scheduler_server
