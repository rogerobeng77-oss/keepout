#!/bin/bash
# usage: run_all.sh <out-root> [--legacy-reference]
set -u
PY=${PYTHON:-python}
export PYTHONPATH=${KEEPOUT_SRC:-}
HERE=$(dirname "$0")
for r in ${RUNS:-courtyard m4-default m4-A m4-B prefab1-default prefab1-crane prefab-full}; do
  while [ $(awk '/MemAvailable/{print $2}' /proc/meminfo) -lt 4000000 ]; do sleep 20; done
  /usr/bin/time -f "$r maxrss_kb=%M" $PY $HERE/measure.py $r "$1" ${2:-} 2>&1 | grep -v '"level": "INFO"'
done
