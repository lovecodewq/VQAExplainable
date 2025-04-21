#!/bin/bash
# train_bottom_up.sh - Script to train the Bottom-Up Attention model

# Default parameters
VG_VERSION="1600-400-20"
SPLIT="minitrain"
OUTPUT_DIR="checkpoints"
GPU_IDS="0"
LOG_LEVEL="ERROR"  # Default log level
DEVICE="cpu"

# Help message
show_help() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -h, --help                 Show this help message"
  echo "  -v, --vg-version VERSION   Visual Genome version (default: $VG_VERSION)"
  echo "  -s, --split SPLIT          Dataset split: train, val, test, minitrain, minival (default: $SPLIT)"
  echo "  -o, --output-dir DIR       Output directory for checkpoints (default: $OUTPUT_DIR)"
  echo "  -g, --gpu-ids IDS          Comma-separated list of GPU IDs to use (default: $GPU_IDS)"
  echo "  -m, --mini                 Use mini training set (shortcut for --split minitrain)"
  echo "  -d, --device DEVICE        Device to use for training: cpu, cuda, mps (default: auto-detect)"
  echo "  -l, --log-level LEVEL      Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: $LOG_LEVEL)"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -h|--help)
      show_help
      exit 0
      ;;
    -v|--vg-version)
      VG_VERSION="$2"
      shift 2
      ;;
    -s|--split)
      SPLIT="$2"
      shift 2
      ;;
    -l|--log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    -o|--output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -g|--gpu-ids)
      GPU_IDS="$2"
      shift 2
      ;;
    -m|--mini)
      SPLIT="minitrain"
      shift
      ;;
    -d|--device)
      DEVICE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Checkpoint filename
CHECKPOINT="$OUTPUT_DIR/bottomup_${SPLIT}_${VG_VERSION}.pth"

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Print training configuration
echo "=== Bottom-Up Attention Training ==="
echo "VG Version:     $VG_VERSION"
echo "Split:          $SPLIT"
echo "Output:         $CHECKPOINT"
echo "GPUs:           $GPU_IDS"
echo "Device:         $DEVICE"
echo "Log Level:      $LOG_LEVEL"
echo "====================================="

# Run training
python train_bottom_up.py \
  --vg-version "$VG_VERSION" \
  --split "$SPLIT" \
  --out-checkpoint "$CHECKPOINT" \
  --device "$DEVICE" \
  --log-level "$LOG_LEVEL"

if [ $? -ne 0 ]; then
  echo "Training failed. Please check the logs for more details."
  exit 1
fi
echo "Training complete. Model saved to $CHECKPOINT"