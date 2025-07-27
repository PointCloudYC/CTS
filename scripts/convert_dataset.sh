#!/bin/bash

# Convert ArchiStyle-v2 to v1 structure
# Usage: ./scripts/convert_dataset.sh

set -e

# Source and destination paths
SOURCE_DIR="data/ArchiStyle-v2/raw"
DEST_DIR="data/ArchiStyle-v2"
TRAIN_DIR="$DEST_DIR/train"
# test or val, both is fine
TEST_DIR="$DEST_DIR/test"

# Create train and test directories
mkdir -p "$TRAIN_DIR" "$TEST_DIR"

# Class mapping: folder_name -> class_id
declare -A CLASS_MAP=(
    ["chuan"]="0"
    ["jin"]="1"
    ["jing"]="2"
    ["min"]="3"
    ["su"]="4"
    ["wan"]="5"
)

echo "Converting ArchiStyle-v2 dataset..."

# Process each class folder
for class_folder in chuan jin jing min su wan; do
    class_id=${CLASS_MAP[$class_folder]}
    echo "Processing class $class_id ($class_folder)..."
    
    # Get all image files in this class folder
    files=("$SOURCE_DIR/$class_folder"/*)
    total_files=${#files[@]}
    
    # Calculate train/test split (80/20)
    train_count=$((total_files * 8 / 10))
    
    echo "  Total files: $total_files, Train: $train_count, Test: $((total_files - train_count))"
    
    # Shuffle files randomly
    shuffled_files=($(printf '%s\n' "${files[@]}" | shuf))
    
    # Process each file
    for i in "${!shuffled_files[@]}"; do
        file="${shuffled_files[$i]}"
        if [[ -f "$file" ]]; then
            # Extract original filename without path
            original_name=$(basename "$file")
            # Remove extension and get just the name
            name_without_ext="${original_name%.*}"
            # Get the original extension
            extension="${original_name##*.}"
            
            # Create new filename: classID_originalName.originalExt
            new_name="${class_id}_${name_without_ext}.${extension}"
            
            # Determine if this goes to train or test
            if [[ $i -lt $train_count ]]; then
                dest_path="$TRAIN_DIR/$new_name"
            else
                dest_path="$TEST_DIR/$new_name"
            fi
            
            # Copy and rename file
            cp "$file" "$dest_path"
        fi
    done
done

# Count final results
train_count=$(find "$TRAIN_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l)
test_count=$(find "$TEST_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l)
total=$((train_count + test_count))

echo ""
echo "Conversion completed!"
echo "Train files: $train_count"
echo "Test files: $test_count"
echo "Total files: $total"
echo "Train/Test ratio: $(echo "scale=2; $train_count * 100 / $total" | bc -l)% / $(echo "scale=2; $test_count * 100 / $total" | bc -l)%"

echo ""
echo "Dataset structure created at:"
echo "  $TRAIN_DIR"
echo "  $TEST_DIR" 