#!/usr/bin/env bash
set -euo pipefail

# -- CONFIGURATION
# You can override these env vars before calling the script if you like:
VG_ROOT="${VG_ROOT:-$PWD/data/visual_genome}"
IMG_DIR="$VG_ROOT/images"
META_DIR="$VG_ROOT/annotations"
PREPROCESS_SCRIPT="${PREPROCESS_SCRIPT:-scripts/setup_vg.py}"
SETUP_SCRIPT="${SETUP_SCRIPT:-setup.py}"
OUT_JSON="$VG_ROOT/vg_annotations.json"

# -- MAKE DIRS
if [ ! -d "$VG_ROOT" ]; then
  mkdir -p "$VG_ROOT"
fi

# dowload syncgraph data
if [ ! -d "$VG_ROOT/scene_graphs" ]; then
  echo "[0/] Downloading syncgraph data..."
  wget -c "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/scene_graphs.json.zip" -O "$VG_ROOT/scene_graphs.json.zip"
  unzip -q "$VG_ROOT/scene_graphs.json.zip" -d "$VG_ROOT/scene_graphs"
  rm "$VG_ROOT/scene_graphs.json.zip"
fi

# -- DOWNLOAD IMAGES (two  files ~9.7 GB + 5.5 GB)
if [ ! -d "$IMG_DIR" ]; then
  echo "[1] Downloading Visual Genome images part 1..."
  wget -c "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip" -O "$VG_ROOT/images1.zip"
  echo "[2] Downloading Visual Genome images part 2..."
  wget -c "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip" -O "$VG_ROOT/images2.zip"
  # -- UNZIP IMAGES
  mkdir -p "$IMG_DIR"
  unzip -q "$VG_ROOT/images1.zip"  -d "$IMG_DIR"
  unzip -q "$VG_ROOT/images2.zip"  -d "$IMG_DIR"
  rm "$VG_ROOT/images1.zip" "$VG_ROOT/images2.zip"
fi

# -- DOWNLOAD ANNOTATIONS
if [ ! -d "$META_DIR" ]; then
  mkdir -p "$META_DIR"  # Create directory first
  echo "[3] Downloading object & attribute dumps..."
  wget -c "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip"    -O "$META_DIR/objects.json.zip"
  wget -c "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attributes.json.zip" -O "$META_DIR/attributes.json.zip"

  # -- UNZIP ANNOTATIONS
  unzip -q "$META_DIR/objects.json.zip"    -d "$META_DIR"
  unzip -q "$META_DIR/attributes.json.zip" -d "$META_DIR"
  rm "$META_DIR"/*.zip
fi


# download image metadata
IMG_METADATA="$META_DIR/image_data.json"
if [ ! -f "$IMG_METADATA" ]; then
  echo "[4] Downloading image metadata..."
  wget -c "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip" -O "$META_DIR/image_data.json.zip"
  # Unzip the file
  echo "[5] Unpacking image metadata..."
  unzip -q "$META_DIR/image_data.json.zip" -d "$META_DIR"
  rm "$META_DIR/image_data.json.zip"
fi


echo "[6] Generating per-image VG annotations JSON..."
if [ ! -f "$PREPROCESS_SCRIPT" ]; then
  echo "ERROR: preprocessing script not found at $PREPROCESS_SCRIPT"
  exit 1
fi

cmd="python3 $PREPROCESS_SCRIPT"
echo $cmd
$cmd

echo "Building vg dataset extension..."
cmd="python $SETUP_SCRIPT build_ext --inplace"
echo $cmd
$cmd

echo "Finished setup vg data"