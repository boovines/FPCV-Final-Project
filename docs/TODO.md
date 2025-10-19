# Future Extension to Video

This document outlines the roadmap for extending the fire detection system from static images to video analysis.

## 📋 Phase 2: Frame-by-Frame Video Analysis

### Step 1: Video Frame Extraction
Add `utils/video_utils.py` with the following functionality:

```python
def extract_frames(video_path, stride=5, output_dir="extracted_frames"):
    """
    Extract frames from a video at regular intervals.
    
    Args:
        video_path: Path to the input video
        stride: Extract every Nth frame (default: 5)
        output_dir: Directory to save extracted frames
    
    Returns:
        List of extracted frame paths
    """
    # Implementation using cv2.VideoCapture
    pass
```

### Step 2: Batch Frame Inference
- Feed extracted frames into the current classifier
- Get per-frame logits/probabilities
- Store results in a time series format

### Step 3: Temporal Smoothing
- Average probabilities across sliding time windows
- Apply moving average or Gaussian smoothing
- Reduce false positives from single-frame anomalies

## 📋 Phase 3: Temporal Modeling

For more sophisticated video understanding:

### Architecture Changes:
1. **Keep EfficientNet Base**: Freeze EfficientNet to extract spatial features
2. **Add Temporal Layers**: 
   - Option A: GRU/LSTM for sequential modeling
   - Option B: Temporal Convolutional Networks (TCN)
   - Option C: 3D Convolutions

### Data Format:
- Use clips of length 8-16 frames
- Input shape: `(batch_size, num_frames, 224, 224, 3)`
- Extract features per frame: `(batch_size, num_frames, feature_dim)`
- Process through temporal layer: `(batch_size, output_dim)`

### Example Architecture:
```python
def build_video_fire_classifier(num_frames=16, img_size=224):
    # EfficientNet for feature extraction
    base = EfficientNetB0(include_top=False, weights="imagenet")
    base.trainable = False
    
    # Time-distributed wrapper for frame-wise processing
    frame_input = Input(shape=(num_frames, img_size, img_size, 3))
    features = TimeDistributed(base)(frame_input)
    features = TimeDistributed(GlobalAveragePooling2D())(features)
    
    # Temporal modeling
    x = GRU(128, return_sequences=False)(features)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)
    
    model = Model(frame_input, output)
    return model
```

## 📋 Phase 4: Real-Time Processing

### Requirements:
1. **Optimize Inference Speed**:
   - Use TensorRT or ONNX for model optimization
   - Consider EfficientNet-Lite or MobileNet for faster inference
   - Implement frame skipping strategies

2. **Video Stream Integration**:
   - Add webcam/RTSP stream support
   - Implement sliding window buffer
   - Add queue-based processing for async inference

3. **Alert System**:
   - Threshold-based fire detection
   - Send notifications/alerts
   - Log detection events with timestamps

### Example Real-Time Script:
```python
import cv2
from collections import deque

def process_video_stream(source=0, buffer_size=16):
    cap = cv2.VideoCapture(source)
    frame_buffer = deque(maxlen=buffer_size)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_buffer.append(preprocess(frame))
        
        if len(frame_buffer) == buffer_size:
            # Run inference on buffered frames
            prediction = model.predict(np.array(frame_buffer))
            
            # Display results
            cv2.putText(frame, f"Fire: {prediction:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Fire Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
```

## 🎯 Implementation Priority

1. **High Priority (Next Steps)**:
   - [ ] Create `utils/video_utils.py` for frame extraction
   - [ ] Implement frame-by-frame inference script
   - [ ] Add temporal smoothing to reduce noise

2. **Medium Priority**:
   - [ ] Collect video dataset for training
   - [ ] Implement temporal modeling architecture
   - [ ] Train on video clips instead of single frames

3. **Low Priority (Future Work)**:
   - [ ] Real-time stream processing
   - [ ] Model optimization (TensorRT/ONNX)
   - [ ] Alert and notification system
   - [ ] Web dashboard for monitoring

## 📊 Dataset Considerations

### For Video Training:
- **Fire Videos**: Wildfires, building fires, campfires, controlled burns
- **Non-Fire Videos**: Smoke without fire, sunsets, orange lights, reflections
- **Duration**: 5-30 second clips recommended
- **Diversity**: Different times of day, weather conditions, fire sizes

### Augmentation for Video:
- Temporal jittering
- Speed variation (slow motion / fast forward)
- Frame dropout
- Brightness/contrast variation

## 🔧 Utilities to Build

```
utils/
├── video_utils.py       # Frame extraction, video I/O
├── temporal_smooth.py   # Smoothing algorithms
├── visualization.py     # Plot predictions over time
└── metrics.py          # Video-specific metrics (temporal IoU, etc.)
```

## 📚 References for Video Analysis

- **Temporal Modeling**:
  - Two-Stream CNNs: [Simonyan & Zisserman, 2014]
  - I3D: [Carreira & Zisserman, 2017]
  - SlowFast Networks: [Feichtenhofer et al., 2019]

- **Fire Detection in Video**:
  - [Chen et al., 2020] "Fire Detection Using Deep Learning"
  - [Xu et al., 2021] "Video Fire Detection Methods"

---

**Note**: Start with Phase 2 (frame-by-frame) as it requires minimal code changes and validates the approach before investing in temporal modeling.

