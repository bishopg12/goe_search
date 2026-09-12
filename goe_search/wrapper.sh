#!/bin/bash
export LD_LIBRARY_PATH=/home/bishopg:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=16
if [[ "$(hostname)" == "soldier10" ]] || [[ "$(hostname)" == "soldier12" ]]; then
    exec /home/bishopg/antigravity/goe_search/goe_search_7x7_hip > /home/bishopg/local_gpu.log 2>&1
else
    exec /home/bishopg/antigravity/goe_search/goe_search_7x7
fi
