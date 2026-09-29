#!/usr/bin/env bash
# Compiles the example kernels against the TornadoVM SDK in $TORNADOVM_HOME.
set -euo pipefail
: "${TORNADOVM_HOME:?set TORNADOVM_HOME to a TornadoVM SDK with the CUDA backend}"
cd "$(dirname "$0")"
javac --release 21 --enable-preview -proc:none -Xlint:-preview \
  -cp "$TORNADOVM_HOME/share/java/tornado/*" -d classes src/*.java
echo "built examples/classes. Try:"
echo "  bin/tcd memcheck --tool racecheck -- -cp examples/classes SharedReduce"
