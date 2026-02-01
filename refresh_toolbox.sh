#!/usr/bin/env bash

set -e

TOOLBOX_NAME="vllm"
IMAGE="docker.io/kyuz0/vllm-therock-gfx1151:latest"

# Base options
OPTIONS="--device /dev/dri --device /dev/kfd --group-add video --group-add render --security-opt seccomp=unconfined"

# Check for InfiniBand devices
if [ -d "/dev/infiniband" ]; then
    echo "🔎 InfiniBand devices detected! Adding RDMA support..."
    OPTIONS="$OPTIONS --device /dev/infiniband --group-add rdma --ulimit memlock=-1"
else
    echo "ℹ️  No InfiniBand devices detected."
fi

# Check dependencies
for cmd in podman toolbox; do
  command -v "$cmd" > /dev/null || { echo "Error: '$cmd' is not installed." >&2; exit 1; }
done

echo "🔄 Refreshing $TOOLBOX_NAME (image: $IMAGE)"

# Remove the toolbox if it exists
if toolbox list 2>/dev/null | grep -q "$TOOLBOX_NAME"; then
  echo "🧹 Removing existing toolbox: $TOOLBOX_NAME"
  toolbox rm -f "$TOOLBOX_NAME"
fi

echo "⬇️ Pulling latest image: $IMAGE"
podman pull "$IMAGE"

# Identify current image ID/digest for this tag
new_id="$(podman image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || true)"
new_digest="$(podman image inspect --format '{{.Digest}}' "$IMAGE" 2>/dev/null || true)"

echo "📦 Recreating toolbox: $TOOLBOX_NAME"
echo "   Options: $OPTIONS"
# Note: toolbox create passes arguments after '--' to podman create
toolbox create "$TOOLBOX_NAME" --image "$IMAGE" -- $OPTIONS

# --- Cleanup: keep only the most recent image for this tag ---
repo="${IMAGE%:*}"

# Remove any other local images still carrying this exact tag but not the newest digest
while read -r id ref dig; do
  if [[ "$id" != "$new_id" ]]; then
      podman image rm -f "$id" >/dev/null 2>&1 || true
  fi
done < <(podman images --digests --format '{{.ID}} {{.Repository}}:{{.Tag}} {{.Digest}}' \
         | awk -v ref="$IMAGE" -v ndig="$new_digest" '$2==ref && $3!=ndig')

# Remove dangling images from this repository (typically prior pulls of this tag)
while read -r id; do
  podman image rm -f "$id" >/dev/null 2>&1 || true
done < <(podman images --format '{{.ID}} {{.Repository}}:{{.Tag}}' \
         | awk -v r="$repo" '$2==r":<none>" {print $1}')
# --- end cleanup ---

echo "✅ $TOOLBOX_NAME refreshed"
