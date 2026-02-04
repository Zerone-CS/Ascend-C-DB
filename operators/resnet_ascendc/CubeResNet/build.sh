#!/bin/bash
set -e

cd "$(dirname "$0")"

source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

rm -rf build
mkdir -p build
cd build

cmake ..
make -j$(nproc)

echo "Build completed. Executables are in build/bin/"
