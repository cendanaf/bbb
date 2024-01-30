#!/bin/bash

rootfs_path=~/BBB/rootfs
output_file=broken_symlinks_list.txt

# Find all files in rootfs_path with broken symlinks and list them in output_file
find "$rootfs_path" -type l ! -exec test -e {} \; -exec sh -c 'echo "{} -> $(readlink "{}")"' \; > "$output_file"

cat "$output_file"
