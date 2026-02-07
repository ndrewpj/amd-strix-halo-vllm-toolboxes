#!/usr/bin/env bash

while true; do
  A_IN=$(rdma statistic | awk '/ip4InOctets/ {print $2}')
  A_OUT=$(rdma statistic | awk '/ip4OutOctets/ {print $2}')
  sleep 1
  B_IN=$(rdma statistic | awk '/ip4InOctets/ {print $2}')
  B_OUT=$(rdma statistic | awk '/ip4OutOctets/ {print $2}')

  RX=$(( (B_IN - A_IN) * 8 ))
  TX=$(( (B_OUT - A_OUT) * 8 ))

  printf "%s RDMA RX: %7sbit/s  TX: %7sbit/s  SUM: %7sbit/s\n" \
    "$(date +%T)" \
    "$(numfmt --to=iec $RX)" \
    "$(numfmt --to=iec $TX)" \
    "$(numfmt --to=iec $((RX+TX)))"
done
