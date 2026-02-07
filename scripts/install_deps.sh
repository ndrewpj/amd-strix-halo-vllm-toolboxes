#!/bin/bash
set -e

# 1. System Base & Build Tools
# Added 'gperftools-libs' for tcmalloc (fixes double-free)
dnf -y install --setopt=install_weak_deps=False --nodocs \
  python3.13 python3.13-devel git rsync libatomic bash ca-certificates curl \
  gcc gcc-c++ binutils make ffmpeg-free \
  cmake ninja-build aria2c tar xz vim nano dialog \
  libdrm-devel zlib-devel openssl-devel pgrep \
  numactl-devel gperftools-libs iproute libibverbs-utils patch perftest ping iperf3 perfquery \
  && dnf clean all && rm -rf /var/cache/dnf/*
