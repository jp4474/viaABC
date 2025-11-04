#!/bin/bash

# Bash script to run Stan inference for Lotka-Volterra model
# Make sure cmdstan is installed and CMDSTAN environment variable is set

export CMDSTAN="/home/jp4474/cmdstan"

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stan model and data files
STAN_FILE="$SCRIPT_DIR/run_inference.stan"
DATA_FILE="$SCRIPT_DIR/lotka_volterra_data.json"
OUTPUT_DIR="$SCRIPT_DIR/output"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Model name (without extension)
MODEL_NAME="lotka_volterra_model"

# Check if cmdstan is available
if [ -z "$CMDSTAN" ]; then
    echo "Error: CMDSTAN environment variable not set."
    echo "Please install cmdstan and set CMDSTAN to the installation directory."
    echo "Example: export CMDSTAN=/path/to/cmdstan"
    exit 1
fi

if [ ! -d "$CMDSTAN" ]; then
    echo "Error: CMDSTAN directory does not exist: $CMDSTAN"
    exit 1
fi

# Check if required files exist
if [ ! -f "$STAN_FILE" ]; then
    echo "Error: Stan file not found: $STAN_FILE"
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file not found: $DATA_FILE"
    exit 1
fi

echo "Starting Stan inference for Lotka-Volterra model..."
echo "Stan file: $STAN_FILE"
echo "Data file: $DATA_FILE"
echo "Output directory: $OUTPUT_DIR"

# Copy the stan file to working directory
cp "$STAN_FILE" "$SCRIPT_DIR/$MODEL_NAME.stan"

# Compile the Stan model using the simpler approach
echo "Compiling Stan model..."
cd "$SCRIPT_DIR"

# Make sure we're in the right directory and use relative path
make -C "$CMDSTAN" "$SCRIPT_DIR/$MODEL_NAME"

if [ $? -ne 0 ]; then
    echo "Error: Failed to compile Stan model. Trying alternative compilation..."
    
    # Alternative compilation method
    cd "$CMDSTAN"
    make "$SCRIPT_DIR/$MODEL_NAME"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to compile Stan model with alternative method"
        exit 1
    fi
fi

cd "$SCRIPT_DIR"

echo "Model compiled successfully!"

# Run sampling
echo "Running MCMC sampling..."
./"$MODEL_NAME" method=sample num_samples=1000 num_warmup=1000 data file="$DATA_FILE" output file="$OUTPUT_DIR/${MODEL_NAME}_samples.csv" random seed=12345

if [ $? -ne 0 ]; then
    echo "Error: Failed to run Stan sampling"
    exit 1
fi

echo "Sampling completed successfully!"
echo "Results saved to: $OUTPUT_DIR/${MODEL_NAME}_samples.csv"

# Move to output directory for diagnostics
cd "$OUTPUT_DIR"

# Optional: Run diagnostics
if [ -f "$CMDSTAN/bin/diagnose" ]; then
    echo "Running diagnostics..."
    "$CMDSTAN/bin/diagnose" "${MODEL_NAME}_samples.csv"
fi

# Optional: Generate summary statistics  
if [ -f "$CMDSTAN/bin/stansummary" ]; then
    echo "Generating summary statistics..."
    "$CMDSTAN/bin/stansummary" "${MODEL_NAME}_samples.csv" > "${MODEL_NAME}_summary.txt"
    echo "Summary saved to: $OUTPUT_DIR/${MODEL_NAME}_summary.txt"
fi

echo "Analysis complete!"
echo "Files generated:"
echo "  - Samples: $OUTPUT_DIR/${MODEL_NAME}_samples.csv"
if [ -f "${MODEL_NAME}_summary.txt" ]; then
    echo "  - Summary: $OUTPUT_DIR/${MODEL_NAME}_summary.txt"
fi
