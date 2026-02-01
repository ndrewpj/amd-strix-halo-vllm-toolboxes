#!/usr/bin/env bash

# -------- dynamic config --------
HOST_ROCE="192.168.100.2"
HOST_ETH="192.168.1.127"

# Automatically detect local and remote RDMA device names
RDMA_DEV_LOCAL=$(ibv_devices | awk 'NR==3 {print $1}')
RDMA_DEV_REMOTE=$(ssh "$HOST_ROCE" "toolbox run -c vllm -- ibv_devices | awk 'NR==3 {print \$1}'")

WORKDIR="/tmp/rdma_bench"
mkdir -p "$WORKDIR"

# -------- helpers --------
parse_ping_avg() {
    grep rtt "$1" | awk -F'/' '{print $5}'
}

parse_iperf_gbps() {
    grep receiver "$1" | tail -n1 | awk '
    {
        val=$(NF-2);
        unit=$(NF-1);
        if (unit=="Mbits/sec") printf "%.2f", val/1000;
        else if (unit=="Gbits/sec") printf "%.2f", val;
        else print "N/A";
    }'
}

parse_rdma_lat_us() {
    val=$(grep -E '^[[:space:]]*[0-9]+' "$1" | tail -n1 | awk '{print $6}')
    echo "${val:-0}"
}

parse_rdma_bw_mib() {
    val=$(grep -E '^[[:space:]]*[0-9]+' "$1" | tail -n1 | awk '{print $4}')
    echo "${val:-0}"
}

# -------- normal ethernet --------
ping -c 10 "$HOST_ETH" > "$WORKDIR/ping_eth.txt"
ssh "$HOST_ROCE" "toolbox run -c vllm -- iperf3 -s -1" >/dev/null 2>&1 &
sleep 1
iperf3 -c "$HOST_ETH" -P 8 -t 10 > "$WORKDIR/iperf_eth.txt"

# -------- roce ethernet (tcp) --------
ping -c 10 "$HOST_ROCE" > "$WORKDIR/ping_roce.txt"
ssh "$HOST_ROCE" "toolbox run -c vllm -- iperf3 -s -1" >/dev/null 2>&1 &
sleep 1
iperf3 -c "$HOST_ROCE" -P 8 -t 10 > "$WORKDIR/iperf_roce.txt"

# -------- rdma latency --------
ssh "$HOST_ROCE" "toolbox run -c vllm -- ib_send_lat --rdma_cm -d $RDMA_DEV_REMOTE" > "$WORKDIR/rdma_lat_srv.txt" 2>&1 &
sleep 2
ib_send_lat --rdma_cm -d "$RDMA_DEV_LOCAL" "$HOST_ROCE" > "$WORKDIR/rdma_lat_cli.txt" 2>&1

# -------- rdma bandwidth (maximized) --------
# We use -x 1 because show_gids confirmed RoCE v2 is at Index 1
ssh "$HOST_ROCE" "toolbox run -c vllm -- ib_write_bw -a -x 1 -q 8 -m 4096" > "$WORKDIR/rdma_bw_srv.txt" 2>&1 &
sleep 2
ib_write_bw -a -x 1 -q 8 -m 4096 "$HOST_ROCE" > "$WORKDIR/rdma_bw_cli.txt" 2>&1

# -------- parse --------
ETH_LAT_MS=$(parse_ping_avg "$WORKDIR/ping_eth.txt")
ETH_BW=$(parse_iperf_gbps "$WORKDIR/iperf_eth.txt")

ROCE_LAT_MS=$(parse_ping_avg "$WORKDIR/ping_roce.txt")
ROCE_BW=$(parse_iperf_gbps "$WORKDIR/iperf_roce.txt")

RDMA_LAT_US=$(parse_rdma_lat_us "$WORKDIR/rdma_lat_cli.txt")
RDMA_BW_MIB=$(parse_rdma_bw_mib "$WORKDIR/rdma_bw_cli.txt")

# Convert units for dual display
ETH_LAT_US=$(python3 -c "print(f'{float(${ETH_LAT_MS:-0}) * 1000:.2f}')")
ROCE_LAT_US=$(python3 -c "print(f'{float(${ROCE_LAT_MS:-0}) * 1000:.2f}')")
RDMA_LAT_MS=$(python3 -c "print(f'{float(${RDMA_LAT_US:-0}) / 1000:.3f}')")

RDMA_BW_GBPS=$(python3 - <<EOF
import sys
try:
    print(round($RDMA_BW_MIB * 8 / 1024, 2))
except:
    print("0.00")
EOF
)

# -------- output --------
echo
echo "=== Network Comparison ==="
echo
printf "%-20s %-15s %-15s %-12s\n" "Path" "Latency (ms)" "Latency (us)" "Bandwidth"
echo "----------------------------------------------------------------"
printf "%-20s %-15s %-15s %-12s\n" "Ethernet (1G LAN)" "${ETH_LAT_MS} ms" "${ETH_LAT_US} us" "${ETH_BW} Gbps"
printf "%-20s %-15s %-15s %-12s\n" "Ethernet (RoCE NIC)" "${ROCE_LAT_MS} ms" "${ROCE_LAT_US} us" "${ROCE_BW} Gbps"
printf "%-20s %-15s %-15s %-12s\n" "RDMA (RoCE)" "${RDMA_LAT_MS} ms" "${RDMA_LAT_US} us" "${RDMA_BW_GBPS} Gbps"
echo
