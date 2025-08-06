#!/bin/bash

# This script sets up and runs ZeroTune with Poetry

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Initialize Poetry if it hasn't been already
if [ ! -f "poetry.lock" ]; then
    echo "Initializing Poetry and installing dependencies..."
    poetry install
fi

# Define help function
function show_help {
    echo "Usage: ./run.sh [OPTION]"
    echo "Options:"
    echo "  --train DATASETS       Build a knowledge base with specified OpenML dataset IDs"
    echo "  --predict DATASET_ID   Predict hyperparameters for the given OpenML dataset ID"
    echo "  --predict-custom       Predict hyperparameters for a custom dataset"
    echo "  --demo                 Run a demonstration of ZeroTune"
    echo "  --datasets             Show recommended datasets for training (default action)"
    echo "  --datasets-default     Explicitly request recommended datasets (same as --datasets)"
    echo "  --datasets-list        List all available datasets in the catalog"
    echo "  --output-dir NAME      Specify a custom output directory name (used with --train and --predict)"
    echo "  --model MODEL_TYPE     Specify ML model type (decision_tree, random_forest, xgboost)"
    echo "  --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh --train \"31 38 44\"                      # Train with datasets 31, 38, and 44"
    echo "  ./run.sh --train \"31 38 44\" --output-dir my_exp  # Train with custom output directory"
    echo "  ./run.sh --train \"31 38 44\" --model random_forest  # Train with Random Forest model"
    echo "  ./run.sh --predict 1464                         # Predict for OpenML dataset 1464"
    echo "  ./run.sh --predict-custom                       # Predict with custom dataset"
    echo "  ./run.sh --demo                                 # Run ZeroTune demo"
    echo "  ./run.sh --datasets                             # Show recommended datasets for training"
    echo "  ./run.sh --datasets-default                     # Explicitly request recommended datasets"
    echo "  ./run.sh --datasets-list                        # List all available datasets"
    echo ""
    echo "Note: After installing the package with 'pip install -e .' or 'poetry install',"
    echo "      you can also run ZeroTune using more Pythonic methods:"
    echo ""
    echo "  1. Direct entry point (recommended):"
    echo "     zerotune train --datasets \"31 38 44\""
    echo ""
    echo "  2. As a Python module:"
    echo "     python -m zerotune train --datasets \"31 38 44\""
    echo ""
    echo "  3. Using the script directly:"
    echo "     ./zerotune_cli.py train --datasets \"31 38 44\""
}

# Set permissions on entry point script
chmod +x zerotune_cli.py

# If no arguments provided, show help
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# Initialize variables
OUTPUT_DIR=""
COMMAND=""
COMMAND_VALUE=""
MODEL_TYPE=""

# Parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --output-dir)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                echo "Error: --output-dir requires a directory name."
                exit 1
            fi
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                echo "Error: --model requires a model type (decision_tree, random_forest, xgboost)."
                exit 1
            fi
            # Validate model type
            case "$2" in
                decision_tree|random_forest|xgboost)
                    MODEL_TYPE="$2"
                    ;;
                *)
                    echo "Error: Invalid model type. Must be one of: decision_tree, random_forest, xgboost."
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --train|--predict)
            COMMAND="$1"
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                echo "Error: $1 requires a value."
                exit 1
            fi
            COMMAND_VALUE="$2"
            shift 2
            ;;
        --predict-custom|--demo|--datasets|--datasets-default|--datasets-list|--help)
            COMMAND="$1"
            if [[ "$1" == "--datasets"* ]] && [ -n "$2" ] && [[ "$2" != --* ]]; then
                # For datasets commands that can take an optional value
                COMMAND_VALUE="$2"
                shift 2
            else
                # Commands without values
                shift 1
            fi
            ;;
        *)
            echo "Error: Unknown option '$1'"
            show_help
            exit 1
            ;;
    esac
done

# Execute command
case "$COMMAND" in
    --train)
        echo "Building knowledge base with datasets: $COMMAND_VALUE"
        
        # Build the command with options
        TRAIN_CMD="poetry run python -m zerotune train --datasets $COMMAND_VALUE"
        
        if [ -n "$OUTPUT_DIR" ]; then
            TRAIN_CMD="$TRAIN_CMD --output-dir \"$OUTPUT_DIR\""
        fi
        
        if [ -n "$MODEL_TYPE" ]; then
            echo "Using model type: $MODEL_TYPE"
            TRAIN_CMD="$TRAIN_CMD --model $MODEL_TYPE"
        fi
        
        # Execute the command
        eval $TRAIN_CMD
        ;;
    --predict)
        echo "Predicting hyperparameters for OpenML dataset $COMMAND_VALUE..."
        
        # Build the command with options
        PREDICT_CMD="poetry run python -m zerotune predict --dataset $COMMAND_VALUE"
        
        if [ -n "$OUTPUT_DIR" ]; then
            PREDICT_CMD="$PREDICT_CMD --output-dir \"$OUTPUT_DIR\""
        fi
        
        if [ -n "$MODEL_TYPE" ]; then
            echo "Using model type: $MODEL_TYPE"
            PREDICT_CMD="$PREDICT_CMD --model $MODEL_TYPE"
        fi
        
        # Execute the command
        eval $PREDICT_CMD
        ;;
    --predict-custom)
        echo "Predicting hyperparameters for custom dataset..."
        
        # Build the command with options
        PREDICT_CUSTOM_CMD="poetry run python -m zerotune predict --dataset custom"
        
        if [ -n "$OUTPUT_DIR" ]; then
            PREDICT_CUSTOM_CMD="$PREDICT_CUSTOM_CMD --output-dir \"$OUTPUT_DIR\""
        fi
        
        if [ -n "$MODEL_TYPE" ]; then
            echo "Using model type: $MODEL_TYPE"
            PREDICT_CUSTOM_CMD="$PREDICT_CUSTOM_CMD --model $MODEL_TYPE"
        fi
        
        # Execute the command
        eval $PREDICT_CUSTOM_CMD
        ;;
    --demo)
        echo "Running ZeroTune demonstration..."
        DEMO_CMD="poetry run python -m zerotune demo"
        
        if [ -n "$MODEL_TYPE" ]; then
            echo "Using model type: $MODEL_TYPE"
            DEMO_CMD="$DEMO_CMD --model $MODEL_TYPE"
        fi
        
        # Execute the command
        eval $DEMO_CMD
        ;;
    --datasets)
        echo "Showing recommended datasets for training..."
        DATASETS_CMD="poetry run python -m zerotune datasets"
        
        if [ -n "$COMMAND_VALUE" ]; then
            DATASETS_CMD="$DATASETS_CMD --count $COMMAND_VALUE"
        fi
        
        eval $DATASETS_CMD
        ;;
    --datasets-default)
        echo "Explicitly showing recommended datasets for training..."
        DATASETS_DEFAULT_CMD="poetry run python -m zerotune datasets --default"
        
        if [ -n "$COMMAND_VALUE" ]; then
            DATASETS_DEFAULT_CMD="$DATASETS_DEFAULT_CMD --count $COMMAND_VALUE"
        fi
        
        eval $DATASETS_DEFAULT_CMD
        ;;
    --datasets-list)
        echo "Listing all available datasets from the catalog..."
        DATASETS_LIST_CMD="poetry run python -m zerotune datasets --list"
        
        if [ -n "$COMMAND_VALUE" ]; then
            DATASETS_LIST_CMD="$DATASETS_LIST_CMD --category $COMMAND_VALUE"
        fi
        
        eval $DATASETS_LIST_CMD
        ;;
    --help)
        show_help
        ;;
    *)
        echo "Error: No valid command specified."
        show_help
        exit 1
        ;;
esac 