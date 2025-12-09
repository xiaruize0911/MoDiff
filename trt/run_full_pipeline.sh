#!/usr/bin/env bash

################################################################################
# MoDiff Complete TensorRT Pipeline
# 
# This script runs the full pipeline:
# 1. Export model to ONNX
# 2. Build FP32 TensorRT engine
# 3. Build INT8 TensorRT engine (native support, MSE calibration)
# 4. Build INT4 TensorRT engine (via INT8 proxy, MSE calibration)
# 5. Run latency/throughput benchmarks
# 6. Generate samples and compute FID scores
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPORT_DIR="$SCRIPT_DIR/export"
CALIB_DIR="$SCRIPT_DIR/calib"
INT4_DIR="$SCRIPT_DIR/int4_output"
INT8_DIR="$SCRIPT_DIR/int8_output"
RESULTS_DIR="$SCRIPT_DIR/benchmark_results"
FID_DIR="$SCRIPT_DIR/fid_comparison"

# Default parameters - 50k samples, 50 steps for proper FID evaluation
NUM_SAMPLES=${NUM_SAMPLES:-50000}
DDIM_STEPS=${DDIM_STEPS:-50}
BENCHMARK_ITERS=${BENCHMARK_ITERS:-100}
SEED=${SEED:-42}
SKIP_ONNX=${SKIP_ONNX:-0}
SKIP_FP32=${SKIP_FP32:-0}
SKIP_INT8=${SKIP_INT8:-0}
SKIP_INT4=${SKIP_INT4:-0}
SKIP_BENCHMARK=${SKIP_BENCHMARK:-0}
SKIP_FID=${SKIP_FID:-0}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --ddim-steps)
            DDIM_STEPS="$2"
            shift 2
            ;;
        --benchmark-iters)
            BENCHMARK_ITERS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --skip-onnx)
            SKIP_ONNX=1
            shift
            ;;
        --skip-fp32)
            SKIP_FP32=1
            shift
            ;;
        --skip-int8)
            SKIP_INT8=1
            shift
            ;;
        --skip-int4)
            SKIP_INT4=1
            shift
            ;;
        --skip-benchmark)
            SKIP_BENCHMARK=1
            shift
            ;;
        --skip-fid)
            SKIP_FID=1
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --num-samples N      Number of samples for FID (default: 1000)"
            echo "  --ddim-steps N       DDIM sampling steps (default: 50)"
            echo "  --benchmark-iters N  Benchmark iterations (default: 100)"
            echo "  --seed N             Random seed (default: 42)"
            echo "  --skip-onnx          Skip ONNX export"
            echo "  --skip-fp32          Skip FP32 engine build"
            echo "  --skip-int8          Skip INT8 engine build"
            echo "  --skip-int4          Skip INT4 engine build"
            echo "  --skip-benchmark     Skip benchmark comparison"
            echo "  --skip-fid           Skip FID computation"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_header() {
    echo ""
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
}

print_step() {
    echo -e "${CYAN}>>> $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Create directories
mkdir -p "$EXPORT_DIR" "$CALIB_DIR" "$INT4_DIR" "$INT8_DIR" "$RESULTS_DIR" "$FID_DIR"

print_header "MoDiff TensorRT Pipeline - FP32, INT8 & INT4"
echo "Configuration:"
echo "  Project root:     $PROJECT_ROOT"
echo "  Export directory: $EXPORT_DIR"
echo "  Samples for FID:  $NUM_SAMPLES"
echo "  DDIM steps:       $DDIM_STEPS"
echo "  Benchmark iters:  $BENCHMARK_ITERS"
echo "  Random seed:      $SEED"
echo ""

################################################################################
# Step 1: Export to ONNX
################################################################################
print_header "Step 1: Export Model to ONNX"

ONNX_FILE="$EXPORT_DIR/modiff_unet_cifar10.onnx"

if [ "$SKIP_ONNX" -eq 1 ]; then
    print_warning "Skipping ONNX export (--skip-onnx)"
else
    # Always re-export ONNX to ensure consistency
    if [ -f "$ONNX_FILE" ]; then
        print_step "Removing old ONNX file and re-exporting..."
        rm -f "$ONNX_FILE"
    fi
    
    print_step "Exporting model to ONNX..."
    cd "$PROJECT_ROOT"
    python trt/export_to_onnx.py \
        --config configs/cifar10.yml \
        --output "$ONNX_FILE" \
        --opset 17
    
    if [ -f "$ONNX_FILE" ]; then
        SIZE=$(du -h "$ONNX_FILE" | cut -f1)
        print_success "ONNX exported: $ONNX_FILE ($SIZE)"
    else
        print_error "Failed to export ONNX"
        exit 1
    fi
fi

################################################################################
# Step 2: Generate Calibration Data
################################################################################
print_header "Step 2: Generate Calibration Data"

CALIB_COUNT=$(find "$CALIB_DIR" -name "*.npz" 2>/dev/null | wc -l)

if [ "$CALIB_COUNT" -ge 100 ]; then
    print_success "Calibration data exists: $CALIB_COUNT samples"
else
    print_step "Generating calibration data..."
    cd "$PROJECT_ROOT"
    
    if [ -f "trt/create_calib_data.py" ]; then
        python trt/create_calib_data.py \
            --output-dir "$CALIB_DIR" \
            --num-samples 512
        print_success "Calibration data generated"
    else
        print_warning "create_calib_data.py not found, using alternative method"
        # Create minimal calibration data from random samples
        python -c "
import numpy as np
import os
os.makedirs('$CALIB_DIR', exist_ok=True)
for i in range(512):
    latent = np.random.randn(1, 3, 32, 32).astype(np.float32)
    timesteps = np.array([np.random.randint(0, 1000)], dtype=np.int64)
    np.savez('$CALIB_DIR/sample_{:04d}.npz'.format(i), latent=latent, timesteps=timesteps)
print('Generated 512 calibration samples')
"
    fi
fi

################################################################################
# Step 3: Build FP32 TensorRT Engine
################################################################################
print_header "Step 3: Build FP32 TensorRT Engine"

FP32_ENGINE="$EXPORT_DIR/modiff_unet_fp32.plan"

if [ "$SKIP_FP32" -eq 1 ]; then
    print_warning "Skipping FP32 engine build (--skip-fp32)"
elif [ -f "$FP32_ENGINE" ]; then
    SIZE=$(du -h "$FP32_ENGINE" | cut -f1)
    print_success "FP32 engine already exists: $FP32_ENGINE ($SIZE)"
else
    print_step "Building FP32 TensorRT engine..."
    cd "$PROJECT_ROOT"
    
    python trt/build_engine.py \
        --onnx "$ONNX_FILE" \
        --calib-dir "$CALIB_DIR" \
        --engine "$FP32_ENGINE" \
        --workspace-gb 8
    
    if [ -f "$FP32_ENGINE" ]; then
        SIZE=$(du -h "$FP32_ENGINE" | cut -f1)
        print_success "FP32 engine built: $FP32_ENGINE ($SIZE)"
    else
        print_error "Failed to build FP32 engine"
        exit 1
    fi
fi

################################################################################
# Step 4: Extract INT8 Scales (Per-Layer from Model)
################################################################################
print_header "Step 4: Extract INT8 Quantization Scales"

INT8_SCALES_DIR="$EXPORT_DIR/extracted_scales_int8"
INT8_SCALES="$INT8_SCALES_DIR/model_scales_int8.json"

if [ "$SKIP_INT8" -eq 1 ]; then
    print_warning "Skipping INT8 scale extraction (--skip-int8)"
elif [ -f "$INT8_SCALES" ]; then
    print_success "INT8 per-layer scales already exist: $INT8_SCALES"
else
    print_step "Extracting per-layer INT8 scales from model (MSE method)..."
    cd "$PROJECT_ROOT"
    
    if [ -f "trt/extract_scales_int8.py" ]; then
        python trt/extract_scales_int8.py \
            --config configs/cifar10.yml \
            --calib-dir "$CALIB_DIR" \
            --output "$INT8_SCALES_DIR" \
            --scale-method mse \
            --num-samples 64 \
            --use-pretrained
        
        if [ -f "$INT8_SCALES" ]; then
            NUM_LAYERS=$(python -c "import json; print(len(json.load(open('$INT8_SCALES'))))" 2>/dev/null || echo "?")
            print_success "INT8 scales extracted for $NUM_LAYERS layers"
        else
            print_warning "Layer scales file not found, will use calibrator during engine build"
        fi
    else
        print_warning "extract_scales_int8.py not found, scales will be computed during engine build"
    fi
fi

################################################################################
# Step 5: Build INT8 TensorRT Engine
################################################################################
print_header "Step 5: Build INT8 TensorRT Engine"

INT8_ENGINE="$INT8_DIR/modiff_unet_int8.plan"

if [ "$SKIP_INT8" -eq 1 ]; then
    print_warning "Skipping INT8 engine build (--skip-int8)"
elif [ -f "$INT8_ENGINE" ]; then
    SIZE=$(du -h "$INT8_ENGINE" | cut -f1)
    print_success "INT8 engine already exists: $INT8_ENGINE ($SIZE)"
else
    print_step "Building INT8 TensorRT engine..."
    cd "$PROJECT_ROOT"
    
    # Use per-layer scales if available
    SCALES_ARG=""
    if [ -f "$INT8_SCALES" ]; then
        SCALES_ARG="--scales-file $INT8_SCALES"
        print_step "Using per-layer scales from $INT8_SCALES"
    fi
    
    python trt/build_engine_int8.py \
        --onnx "$ONNX_FILE" \
        --calib-dir "$CALIB_DIR" \
        $SCALES_ARG \
        --engine "$INT8_ENGINE" \
        --workspace-gb 8 \
        --scale-method mse \
        --fp16
    
    if [ -f "$INT8_ENGINE" ]; then
        SIZE=$(du -h "$INT8_ENGINE" | cut -f1)
        print_success "INT8 engine built: $INT8_ENGINE ($SIZE)"
    else
        print_error "Failed to build INT8 engine"
        exit 1
    fi
fi

################################################################################
# Step 6: Extract INT4 Scales (Per-Layer from Model)
################################################################################
print_header "Step 6: Extract INT4 Quantization Scales"

INT4_SCALES_DIR="$EXPORT_DIR/extracted_scales_int4"
INT4_SCALES="$INT4_SCALES_DIR/model_scales_int4.json"

if [ -f "$INT4_SCALES" ]; then
    print_success "INT4 per-layer scales already exist: $INT4_SCALES"
else
    print_step "Extracting per-layer INT4 scales from model (MSE method)..."
    cd "$PROJECT_ROOT"
    
    if [ -f "trt/extract_scales_int4.py" ]; then
        # Extract scales by running calibration data through the model
        python trt/extract_scales_int4.py \
            --config configs/cifar10.yml \
            --calib-dir "$CALIB_DIR" \
            --output "$INT4_SCALES_DIR" \
            --n-bits 4 \
            --scale-method mse \
            --num-samples 64 \
            --use-pretrained
        
        if [ -f "$INT4_SCALES" ]; then
            NUM_LAYERS=$(python -c "import json; print(len(json.load(open('$INT4_SCALES'))))" 2>/dev/null || echo "?")
            print_success "INT4 scales extracted for $NUM_LAYERS layers"
        else
            print_warning "Layer scales file not found, will use input-only calibration"
        fi
    else
        print_warning "extract_scales_int4.py not found, scales will be computed during engine build"
    fi
fi

################################################################################
# Step 7: Build INT4 TensorRT Engine
################################################################################
print_header "Step 7: Build INT4 TensorRT Engine"

INT4_ENGINE="$INT4_DIR/modiff_unet_int4.plan"

if [ "$SKIP_INT4" -eq 1 ]; then
    print_warning "Skipping INT4 engine build (--skip-int4)"
elif [ -f "$INT4_ENGINE" ]; then
    SIZE=$(du -h "$INT4_ENGINE" | cut -f1)
    print_success "INT4 engine already exists: $INT4_ENGINE ($SIZE)"
else
    print_step "Building INT4 TensorRT engine..."
    cd "$PROJECT_ROOT"
    
    # Use per-layer scales if available
    SCALES_ARG=""
    if [ -f "$INT4_SCALES" ]; then
        SCALES_ARG="--scales-file $INT4_SCALES"
        print_step "Using per-layer scales from $INT4_SCALES"
    elif [ -f "$CALIB_DIR/int4_scales.json" ]; then
        SCALES_ARG="--scales-file $CALIB_DIR/int4_scales.json"
        print_warning "Using input-only scales (may result in higher FID)"
    fi
    
    python trt/build_engine_int4.py \
        --onnx "$ONNX_FILE" \
        --calib-dir "$CALIB_DIR" \
        $SCALES_ARG \
        --engine "$INT4_ENGINE" \
        --workspace-gb 8 \
        --n-bits 4 \
        --scale-method mse \
        --fp16
    
    if [ -f "$INT4_ENGINE" ]; then
        SIZE=$(du -h "$INT4_ENGINE" | cut -f1)
        print_success "INT4 engine built: $INT4_ENGINE ($SIZE)"
    else
        print_error "Failed to build INT4 engine"
        exit 1
    fi
fi

################################################################################
# Step 8: Run Benchmark Comparison
################################################################################
print_header "Step 8: Run Latency/Throughput Benchmark"

if [ "$SKIP_BENCHMARK" -eq 1 ]; then
    print_warning "Skipping benchmark (--skip-benchmark)"
else
    print_step "Running benchmark comparison (FP32 vs INT8 vs INT4)..."
    cd "$PROJECT_ROOT"
    
    python trt/benchmark_comparison.py \
        --fp32-engine "$FP32_ENGINE" \
        --int8-engine "$INT8_ENGINE" \
        --int4-engine "$INT4_ENGINE" \
        --output-dir "$RESULTS_DIR" \
        --warmup 10 \
        --iterations "$BENCHMARK_ITERS" \
        --batch-size 1
    
    print_success "Benchmark completed"
    
    # Display summary if results exist
    if [ -f "$RESULTS_DIR/benchmark_results.csv" ]; then
        echo ""
        echo "Benchmark Results:"
        cat "$RESULTS_DIR/benchmark_results.csv"
    fi
fi

################################################################################
# Step 9: Compute FID Scores
################################################################################
print_header "Step 9: Compute FID Scores"

if [ "$SKIP_FID" -eq 1 ]; then
    print_warning "Skipping FID computation (--skip-fid)"
else
    print_step "Generating samples and computing FID..."
    cd "$PROJECT_ROOT"
    
    python trt/fid_comparison.py \
        --fp32-engine "$FP32_ENGINE" \
        --int8-engine "$INT8_ENGINE" \
        --int4-engine "$INT4_ENGINE" \
        --num-samples "$NUM_SAMPLES" \
        --num-steps "$DDIM_STEPS" \
        --output-dir "$FID_DIR" \
        --seed "$SEED" \
        --cifar-dir "data/cifar-10-batches-py"
    
    print_success "FID computation completed"
    
    # Display results
    if [ -f "$FID_DIR/fid_results.json" ]; then
        echo ""
        echo "FID Results:"
        cat "$FID_DIR/fid_results.json"
    fi
fi

################################################################################
# Summary
################################################################################
print_header "Pipeline Complete - Summary"

echo "Generated Files:"
echo ""

if [ -f "$ONNX_FILE" ]; then
    SIZE=$(du -h "$ONNX_FILE" | cut -f1)
    echo "  ONNX Model:     $ONNX_FILE ($SIZE)"
fi

if [ -f "$FP32_ENGINE" ]; then
    SIZE=$(du -h "$FP32_ENGINE" | cut -f1)
    echo "  FP32 Engine:    $FP32_ENGINE ($SIZE)"
fi

if [ -f "$INT8_ENGINE" ]; then
    SIZE=$(du -h "$INT8_ENGINE" | cut -f1)
    echo "  INT8 Engine:    $INT8_ENGINE ($SIZE)"
fi

if [ -f "$INT4_ENGINE" ]; then
    SIZE=$(du -h "$INT4_ENGINE" | cut -f1)
    echo "  INT4 Engine:    $INT4_ENGINE ($SIZE)"
fi

if [ -f "$RESULTS_DIR/benchmark_results.csv" ]; then
    echo "  Benchmark CSV:  $RESULTS_DIR/benchmark_results.csv"
fi

if [ -f "$FID_DIR/fid_results.json" ]; then
    echo "  FID Results:    $FID_DIR/fid_results.json"
fi

echo ""
print_success "All steps completed successfully!"
echo ""
echo "To re-run specific steps, use:"
echo "  --skip-onnx       Skip ONNX export"
echo "  --skip-fp32       Skip FP32 engine build"
echo "  --skip-int8       Skip INT8 engine build"
echo "  --skip-int4       Skip INT4 engine build"
echo "  --skip-benchmark  Skip benchmark"
echo "  --skip-fid        Skip FID computation"
echo ""
