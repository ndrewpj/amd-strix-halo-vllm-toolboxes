#!/bin/bash
set -e
# Configuration
# Paths identified from your environment
ROCM_LIB_PATH="/opt/rocm/lib/librccl.so.1.0"
VENV_LIB_PATH="/opt/venv/lib/python3.13/site-packages/_rocm_sdk_libraries_gfx1151/lib/librccl.so.1"
BACKUP_DIR="./rccl_backups_$(date +%Y%m%d_%H%M%S)"
# Files to replace
# We assume the new library is named 'librccl.so' or 'librccl.so.1' in the current directory or provided as arg
NEW_LIB="${1:-librccl.so.1}"
usage() {
    echo "Usage: $0 [install <path_to_new_lib> | restore]"
    echo "  install: Backs up existing libs and installs the new one."
    echo "  restore: Restores libraries from the most recent backup directory."
    exit 1
}
do_install() {
    if [ ! -f "$NEW_LIB" ]; then
        echo "Error: New library file '$NEW_LIB' not found."
        echo "Please provide the path to the newly built librccl.so.1"
        exit 1
    fi
    echo "=== Installing Custom RCCL (gfx1151) ==="
    echo "Creating backup directory: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    # 1. Backup /opt/rocm location
    if [ -f "$ROCM_LIB_PATH" ]; then
        echo "Backing up $ROCM_LIB_PATH..."
        cp -v "$ROCM_LIB_PATH" "$BACKUP_DIR/librccl.so.1.0.rocm.bak"
    else
        echo "Warning: $ROCM_LIB_PATH not found, skipping backup."
    fi
    # 2. Backup /opt/venv location
    if [ -f "$VENV_LIB_PATH" ]; then
        echo "Backing up $VENV_LIB_PATH..."
        cp -v "$VENV_LIB_PATH" "$BACKUP_DIR/librccl.so.1.venv.bak"
    else
        echo "Warning: $VENV_LIB_PATH not found, skipping backup."
    fi
    # Save backup dir name for restore
    echo "$BACKUP_DIR" > .last_rccl_backup
    # 3. Install to /opt/rocm
    echo "Installing to $ROCM_LIB_PATH..."
    # We use sudo assuming root ownership as shown in your ls output
    sudo cp -v "$NEW_LIB" "$ROCM_LIB_PATH"
    # 4. Install to /opt/venv
    if [ -d "$(dirname "$VENV_LIB_PATH")" ]; then
        echo "Installing to $VENV_LIB_PATH..."
        sudo cp -v "$NEW_LIB" "$VENV_LIB_PATH"
    else
        echo "Skipping venv install (directory not found)."
    fi
    echo "=== Installation Complete ==="
}
do_restore() {
    if [ ! -f .last_rccl_backup ]; then
        echo "Error: No previous backup record found (.last_rccl_backup)."
        echo "Please manually restore from your backup directories."
        exit 1
    fi
    
    LAST_BACKUP=$(cat .last_rccl_backup)
    echo "=== Restoring RCCL from $LAST_BACKUP ==="
    if [ ! -d "$LAST_BACKUP" ]; then
        echo "Error: Backup directory $LAST_BACKUP does not exist."
        exit 1
    fi
    # Restore ROCm lib
    if [ -f "$LAST_BACKUP/librccl.so.1.0.rocm.bak" ]; then
        echo "Restoring $ROCM_LIB_PATH..."
        sudo cp -v "$LAST_BACKUP/librccl.so.1.0.rocm.bak" "$ROCM_LIB_PATH"
    fi
    # Restore Venv lib
    if [ -f "$LAST_BACKUP/librccl.so.1.venv.bak" ]; then
        echo "Restoring $VENV_LIB_PATH..."
        sudo cp -v "$LAST_BACKUP/librccl.so.1.venv.bak" "$VENV_LIB_PATH"
    fi
    echo "=== Restore Complete ==="
}
COMMAND="$1"
shift
case "$COMMAND" in
    install)
        NEW_LIB="${1:-librccl.so.1}"
        do_install
        ;;
    restore)
        do_restore
        ;;
    *)
        usage
        ;;
esac
