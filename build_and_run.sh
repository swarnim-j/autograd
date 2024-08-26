#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure CMake
cmake ..

# Build the project
cmake --build .

# Run the executable
./autograd