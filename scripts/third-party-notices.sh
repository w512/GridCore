#!/bin/sh
# Prints the license and NOTICE files of every module compiled into the
# gridcore binary, for THIRD_PARTY_NOTICES. Apache-2.0 asks for the NOTICE
# texts of its modules to travel with the binaries built from them.
#
#   scripts/third-party-notices.sh > THIRD_PARTY_NOTICES
set -eu
cd "$(dirname "$0")/.."
go mod download
cat <<'HEAD'
GridCore is licensed under the MIT License (see LICENSE). The gridcore
binary also includes the following third-party modules, each under its own
license, reproduced below together with any NOTICE file it ships.
HEAD
# The union over every platform the release builds: some modules (procfs)
# are compiled in on Linux only.
for target in linux/amd64 linux/arm64 darwin/arm64 darwin/amd64; do
	GOOS=${target%/*} GOARCH=${target#*/} go list -deps \
		-f '{{with .Module}}{{if not .Main}}{{.Path}} {{.Version}} {{.Dir}}{{end}}{{end}}' ./cmd/gridcore
done |
	sort -u |
	while read -r path version dir; do
		printf '\n================================================================================\n'
		printf '%s %s\n' "$path" "$version"
		printf '================================================================================\n'
		for f in "$dir"/LICENSE* "$dir"/LICENCE* "$dir"/COPYING* "$dir"/NOTICE*; do
			[ -f "$f" ] || continue
			printf '\n--- %s ---\n\n' "$(basename "$f")"
			cat "$f"
		done
	done
