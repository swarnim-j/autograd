#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure and build
cmake ..
cmake --build . --target all

# Run the main executable
./autograd