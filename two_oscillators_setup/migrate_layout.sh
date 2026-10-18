#!/usr/bin/env bash
# migrate_layout.sh -- mueve las corridas con el nombre viejo a la estructura nueva
# (una sola vez; se puede volver a correr, solo mueve lo que falte):
#
#   <eje>_precession/output/setup2_<tag>/        -> <eje>_precession/output/DFF/F1-2/<tag>/
#   <eje>_precession/output/setup1_<tag>/        -> <eje>_precession/output/FDF/Fm1-1/<tag>/
#   <eje>_precession/output/logs/setup2_<tag>.log -> <eje>_precession/output/logs/DFF/F1-2/<tag>.log
#   <eje>_precession/output/logs/setup1_<tag>.log -> <eje>_precession/output/logs/FDF/Fm1-1/<tag>.log
#
# Uso (desde two_oscillators_setup, en el cluster y en el computador local):
#   bash migrate_layout.sh
set -u
cd "$(dirname "$0")"

dest() { case $1 in setup2) echo "DFF/F1-2" ;; setup1) echo "FDF/Fm1-1" ;; esac; }

for ax in y_precession z_precession; do
  out="$ax/output"
  [ -d "$out" ] || continue
  for run in "$out"/setup[12]_*/; do
    [ -d "$run" ] || continue
    b=$(basename "$run"); st=${b%%_*}; tag=${b#*_}
    to="$out/$(dest "$st")/$tag"
    if [ -e "$to" ]; then echo "  ya existe, no se mueve: $to"; continue; fi
    mkdir -p "$(dirname "$to")"
    mv "$run" "$to" && echo "  $run -> $to"
  done
  for lg in "$out"/logs/setup[12]_*.log; do
    [ -f "$lg" ] || continue
    b=$(basename "$lg" .log); st=${b%%_*}; tag=${b#*_}
    to="$out/logs/$(dest "$st")/$tag.log"
    mkdir -p "$(dirname "$to")"
    mv "$lg" "$to" && echo "  $lg -> $to"
  done
done
echo "listo"
