#!/usr/bin/env bash
# Provision the Nsight Systems CLI in a container that cannot install the full package.
#
# Why not just `apt-get install nsight-systems-...`: the packaged version depends on libopengl0,
# libegl1 and libxi6 -- GUI libraries that are not installable in this image -- so apt refuses.
# The CLI binary needs none of them, so the .deb is downloaded and unpacked and only
# target-linux-x64/nsys is used.
#
# Why nsys works here at all when ncu does not: nsys traces through CUPTI's Activity API, which
# is not gated by the GPU performance-counter permission. ncu needs the counters and this driver
# has RmProfilingAdminOnly=1, which cannot be changed from inside a container. So tracing is
# available and hardware counters are not.
#
# Prints the path to the nsys binary on stdout; everything else goes to stderr.
set -euo pipefail

DEST="${NSYS_DIR:-/opt/nsys}"
REPO_LIST=/etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list

existing="$(command -v nsys 2>/dev/null || true)"
if [ -n "$existing" ]; then echo "$existing"; exit 0; fi
# `find` on a missing dir exits non-zero, which under `set -e -o pipefail` would abort here
# before anything is printed -- hence the mkdir and the `|| true`.
mkdir -p "$DEST"
found="$(find "$DEST" -name nsys -type f -path '*target-linux-x64*' 2>/dev/null | head -1 || true)"
if [ -n "$found" ]; then echo "$found"; exit 0; fi

echo "provisioning Nsight Systems CLI into $DEST" >&2
mkdir -p "$DEST/pkg"
if [ -f "$REPO_LIST" ]; then
  apt-get update -o Dir::Etc::sourcelist="${REPO_LIST#/etc/apt/}" \
                 -o Dir::Etc::sourceparts="-" -o APT::Get::List-Cleanup="0" >&2
else
  apt-get update >&2
fi

PKG="$(apt-cache pkgnames 2>/dev/null | grep -E '^nsight-systems-[0-9]' | sort -V | tail -1)"
[ -n "$PKG" ] || { echo "no nsight-systems package in the configured repos" >&2; exit 1; }
echo "downloading $PKG" >&2
( cd "$DEST/pkg" && apt-get download "$PKG" >&2 )
dpkg-deb -x "$DEST"/pkg/*.deb "$DEST" >&2
rm -rf "$DEST/pkg"

BIN="$(find "$DEST" -name nsys -type f -path '*target-linux-x64*' | head -1)"
[ -n "$BIN" ] || { echo "nsys binary not found after unpacking" >&2; exit 1; }
chmod +x "$BIN"
"$BIN" --version >&2
echo "$BIN"
