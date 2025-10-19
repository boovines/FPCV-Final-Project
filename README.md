# Fire Detection - Phase 1

A deep learning-based fire detection system using EfficientNet-B0 for binary classification (fire vs. non-fire).

## 📁 Project Structure

```
fire_detection_phase1/
│
├── data/                     # Place your fire / non-fire images here
│   ├── fire/
│   └── non_fire/
│
├── models/
│   └── efficientnet_fire_classifier.py
│
├── notebooks/
│   └── exploration.py        # Data exploration and analysis script
│
├── train.py                  # Main training script
├── infer.py                  # Inference / demo
├── exploration.py            # Data exploration and model evaluation
└── requirements.txt          # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Place your images in the following structure:
- `data/fire/` - Images containing fire
- `data/non_fire/` - Images without fire

### 3. Train the Model

```bash
python train.py
```

This will:
- Load images from `data/fire` and `data/non_fire`
- Split them 80/20 for training/validation
- Train an EfficientNet-B0 classifier for 10 epochs
- Save the best model as `fire_model.h5`
- Save the final model as `fire_classifier_final/`

### 4. Run Inference

```bash
python infer.py path/to/test/image.jpg
```

Example:
```bash
python infer.py data/fire/sample_fire.jpg
```

### 5. Explore and Evaluate

```bash
cd notebooks
python exploration.py
```

This will:
- Analyze your dataset distribution
- Visualize sample images
- Evaluate the trained model
- Generate confusion matrix and metrics
- Save visualization plots

## 🧠 Model Architecture

- **Base Model**: EfficientNet-B0 (pretrained on ImageNet)
- **Input Size**: 224x224x3
- **Classification**: Binary (fire vs. not-fire)
- **Output**: Sigmoid activation (probability score 0-1)

### Architecture Details:
1. EfficientNet-B0 base (frozen for transfer learning)
2. Global Average Pooling
3. Dropout (0.3)
4. Dense layer (128 units, ReLU)
5. Output layer (1 unit, Sigmoid)

## 📊 Training Configuration

- **Optimizer**: Adam (learning rate: 1e-4)
- **Loss**: Binary Crossentropy
- **Metrics**: Accuracy, AUC
- **Batch Size**: 32
- **Image Size**: 224x224
- **Epochs**: 10 (with early stopping)

## 🎯 Performance Tips

1. **Data Augmentation**: Consider adding data augmentation for better generalization
2. **Fine-tuning**: Unfreeze EfficientNet layers after initial training for better performance
3. **Class Imbalance**: Use class weights if your dataset is imbalanced
4. **More Data**: Collect more diverse fire/non-fire samples

## 🔮 Future Extensions to Video

See `docs/TODO.md` for detailed plans on extending this to video analysis.

### Roadmap:
1. ✅ Phase 1: Image-based fire detection (current)
2. 🔜 Phase 2: Frame-by-frame video analysis
3. 🔜 Phase 3: Temporal modeling with RNN/LSTM
4. 🔜 Phase 4: Real-time video stream processing

## 📝 Notes

- The model is initially trained with frozen EfficientNet layers for faster convergence
- Best model is saved based on validation loss
- Early stopping prevents overfitting

## 🐛 Troubleshooting

**Issue**: "No images found in data/"
- **Solution**: Make sure your images are in `data/fire/` and `data/non_fire/` directories

**Issue**: "Model not found"
- **Solution**: Train the model first using `python train.py`

**Issue**: Out of memory errors
- **Solution**: Reduce `BATCH_SIZE` in `train.py`

## 📚 References

- EfficientNet: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- Transfer Learning: [Keras Applications](https://keras.io/api/applications/)

