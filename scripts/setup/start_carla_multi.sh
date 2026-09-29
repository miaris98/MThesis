#!/usr/bin/env bash
# Per-port CARLA (re)start for concurrent closed-loop arms, logging to carla_server_<port>.log.
# start_carla.sh itself is now per-port (it used to kill every CarlaUE4 process); this wrapper only keeps the
# per-port log name its callers expect. Space ports by at least 10 (see start_carla.sh).
WS="${WORKSPACE:-/workspace}"
LOG="${LOG:-$WS/carla_server_${1:-2000}.log}" exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start_carla.sh" "$@"
