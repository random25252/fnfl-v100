#!/bin/bash
exec nvidia-smi --query-gpu=timestamp,index,temperature.gpu,temperature.memory,power.draw,clocks.sm,clocks.mem,utilization.gpu,clocks_event_reasons.active --format=csv,noheader,nounits -lms 500 > "$1"
