#!/bin/bash

echo "🔥 FIRE DETECTION PIPELINE 🔥"
echo "======================================"
echo ""

# Step 1: Check Python
echo "📍 Step 1: Checking Python installation..."
python3 --version || { echo "❌ Python3 not found"; exit 1; }
echo ""

# Step 2: Install dependencies
echo "📍 Step 2: Installing dependencies..."
echo "This may take a few minutes..."
pip3 install -r requirements.txt --quiet || { echo "❌ Failed to install dependencies"; exit 1; }
echo "✅ Dependencies installed"
echo ""

# Step 3: Prepare dataset
echo "📍 Step 3: Preparing dataset (organizing images)..."
python3 prepare_data.py || { echo "❌ Failed to prepare data"; exit 1; }
echo ""

# Step 4: Train model
echo "📍 Step 4: Training the model..."
echo "This will take several minutes depending on your hardware..."
python3 train.py || { echo "❌ Training failed"; exit 1; }
echo ""

# Step 5: Explore and evaluate
echo "📍 Step 5: Running exploratory analysis..."
cd notebooks
python3 exploration.py || { echo "❌ Exploration failed"; exit 1; }
cd ..
echo ""

echo "======================================"
echo "✅ PIPELINE COMPLETE!"
echo "======================================"
echo ""
echo "📊 Generated files:"
echo "  - fire_model.h5 (best model checkpoint)"
echo "  - fire_classifier_final/ (final trained model)"
echo "  - notebooks/sample_images.png"
echo "  - notebooks/predictions.png"
echo "  - notebooks/confusion_matrix.png"
echo ""
echo "🧪 Test inference with:"
echo "  python3 infer.py data/fire/test_1.jpg"
echo ""

