#!/bin/bash

# Benchmark Runner Script
# Compares Python vs Walrus performance

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WALRUS_DIR="$(dirname "$SCRIPT_DIR")"
WALRUS="$WALRUS_DIR/target/release/walrus"

echo "=========================================="
echo "  Python vs Walrus Benchmark Suite"
echo "=========================================="
echo ""

# Check if walrus is built
if [ ! -f "$WALRUS" ]; then
    echo "Building Walrus in release mode..."
    cd "$WALRUS_DIR" && cargo build --release
fi

# Get Python version
PYTHON_VERSION=$(python3 --version 2>&1)
echo "Python version: $PYTHON_VERSION"
echo "Walrus binary: $WALRUS"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Run each benchmark
for walrus_file in "$SCRIPT_DIR"/*.walrus; do
    basename=$(basename "$walrus_file" .walrus)
    python_file="$SCRIPT_DIR/${basename}.py"
    
    if [ -f "$python_file" ]; then
        echo "=========================================="
        echo "Benchmark: $basename"
        echo "=========================================="
        
        # Run Python
        echo ""
        echo "--- Python ---"
        py_output=$( { /usr/bin/time -p python3 "$python_file"; } 2>&1 )
        py_time=$(echo "$py_output" | grep "^real" | awk '{print $2}')
        echo "$py_output" | grep -v "^real\|^user\|^sys"
        echo "Time: ${py_time}s"
        
        # Run Walrus
        echo ""
        echo "--- Walrus ---"
        wal_output=$( { /usr/bin/time -p "$WALRUS" "$walrus_file"; } 2>&1 )
        wal_time=$(echo "$wal_output" | grep "^real" | awk '{print $2}')
        echo "$wal_output" | grep -v "^real\|^user\|^sys"
        echo "Time: ${wal_time}s"
        
        # Compare
        if [ -n "$py_time" ] && [ -n "$wal_time" ]; then
            ratio=$(echo "scale=2; $wal_time / $py_time" | bc 2>/dev/null)
            if [ -n "$ratio" ]; then
                echo ""
                if (( $(echo "$ratio < 1" | bc -l) )); then
                    echo -e "${GREEN}Walrus is ${ratio}x faster${NC}"
                else
                    echo -e "${RED}Python is ${ratio}x faster${NC}"
                fi
            fi
        fi
        
        echo ""
    fi
done

echo "=========================================="
echo "  Benchmark Complete!"
echo "=========================================="
