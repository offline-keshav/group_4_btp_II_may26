# Alzheimer's Disease Detection using Deep Learning

A multi-model deep learning approach for Alzheimer's Disease (AD) detection using EEG data. This project implements three specialized neural network architectures (BiLSTM, CNN-Spatial, and CNN-Spectral) with a transformer-based fusion model for improved classification accuracy.

## Project Overview

This project develops a state-of-the-art system for detecting Alzheimer's Disease from EEG signals using ensemble deep learning techniques. The approach combines:
- **BiLSTM Model**: Captures temporal dependencies in EEG sequences
- **CNN-Spatial Model**: Extracts spatial features across EEG channels
- **CNN-Spectral Model**: Analyzes frequency domain characteristics
- **End-to-End Fusion Model**: Integrates predictions from all three models using multi-head attention

## Dataset

**Data Statistics:**
- **Subjects**: 88 individuals
- **Classes**: Alzheimer's Disease (AD) vs. Healthy controls
- **EEG Channels**: 19
- **Sampling Rate**: 128 Hz
- **Data Format**: Segmented into 1024-sample windows for neural network input

**Data Organization:**
- `Data_sampled_128HZ/`: Raw EEG data for each subject (sub-001 to sub-088)
- `Train_Test_Split.csv`: Subject labels and train/test split assignments
- `subjects_metadata.csv`: Additional subject metadata

## Project Structure

```
.
├── README.md                           # This file
├── BiLSTM_Model.ipynb                  # BiLSTM architecture implementation
├── CNNSpatial_Model.ipynb              # Spatial CNN model
├── CNNSpectral_Model.ipynb             # Spectral (frequency domain) CNN model
├── End_to_End_Model.ipynb              # Transformer-based fusion model
├── Train_Test_Split.csv                # Train/test split assignments
├── Data_sampled_128HZ/                 # EEG data directory
│   ├── sub-001_data.npy to sub-088_data.npy
│   └── subjects_metadata.csv
└── Models/                             # Pre-trained model weights
    ├── Final_Bilstm_model.keras
    ├── Final_CNNSpatial_model.keras
    ├── Final_CNNSpectral_model.keras
    └── Final_End_to_End_model.keras
```

## Model Descriptions

### BiLSTM Model (`BiLSTM_Model.ipynb`)
- **Architecture**: Bidirectional LSTM layers
- **Purpose**: Captures temporal patterns and long-range dependencies in EEG sequences
- **Input**: (batch_size, 19, 1024) - 19 channels × 1024 time samples
- **Output**: Feature vector for classification

### CNN-Spatial Model (`CNNSpatial_Model.ipynb`)
- **Architecture**: Convolutional layers optimized for spatial domain
- **Purpose**: Extracts features from EEG channel relationships (spatial structure)
- **Input**: (batch_size, 19, 1024)
- **Output**: Feature vector for classification

### CNN-Spectral Model (`CNNSpectral_Model.ipynb`)
- **Architecture**: Convolutional layers for frequency domain analysis
- **Purpose**: Analyzes spectral characteristics and frequency components of EEG signals
- **Input**: (batch_size, 19, 1024)
- **Output**: Feature vector for classification

### End-to-End Fusion Model (`End_to_End_Model.ipynb`)
- **Architecture**: Transformer-based multi-modal fusion
- **Components**:
  - Dense projection layers to align feature dimensions (128D)
  - Multi-head self-attention (4 heads) across model outputs
  - Feed-forward layers for feature refinement
  - Global average pooling and softmax classification
- **Purpose**: Combines predictions from all three models for improved accuracy
- **Input**: Feature vectors from BiLSTM, CNN-Spatial, and CNN-Spectral models
- **Output**: Binary classification (AD vs. Healthy)

## Requirements

```
tensorflow>=2.10.0
keras>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
```

## Usage

### 1. Data Preparation
The data is pre-processed and segmented into 1024-sample windows. Each subject's EEG data is loaded from `.npy` files and normalized.

```python
# Example: Load and segment data
from End_to_End_Model import load_and_segment

segments = load_and_segment('sub-001', data_dir='Data_sampled_128HZ')
# Returns array of shape (num_segments, 19, 1024)
```

### 2. Training Individual Models
Run each notebook to train individual models:

```bash
# Train BiLSTM model
jupyter notebook BiLSTM_Model.ipynb

# Train CNN-Spatial model
jupyter notebook CNNSpatial_Model.ipynb

# Train CNN-Spectral model
jupyter notebook CNNSpectral_Model.ipynb
```

### 3. Running the Fusion Model
Execute the End-to-End model notebook which:
- Loads pre-trained individual models
- Removes the output layers for feature extraction
- Creates a transformer-based fusion architecture
- Trains the fusion model on combined features

```bash
jupyter notebook End_to_End_Model.ipynb
```

### 4. Making Predictions
```python
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained fusion model
fusion_model = load_model('Models/Final_End_to_End_model.keras')

# Load features from individual models
# ... (process EEG data through each model)

# Make prediction
predictions = fusion_model.predict([bilstm_features, cnn_spatial_features, cnn_spectral_features])
# Returns shape (batch_size, 2) with probabilities [P(Healthy), P(AD)]
```

## Data Processing Pipeline

1. **Load Data**: Raw EEG data (.npy format) with shape (19, time_steps)
2. **Segmentation**: Divide time series into non-overlapping 1024-sample windows
3. **Normalization**: 
   - Multiply by 1e6 (convert to microvolts)
   - Remove channel-wise mean
4. **Train/Test Split**: Stratified split based on `Train_Test_Split.csv`

## Key Features

- ✅ Pre-trained models available for immediate inference
- ✅ Modular architecture supporting individual and ensemble predictions
- ✅ Transformer-based multi-head attention fusion
- ✅ Reproducible results with fixed random seeds
- ✅ EEG data from 88 subjects with clear train/test division
- ✅ Support for both spatial and spectral domain analysis

## Results

The fusion model leverages complementary strengths:
- **BiLSTM**: Temporal dependencies (sequential patterns)
- **CNN-Spatial**: Inter-channel relationships (spatial structure)
- **CNN-Spectral**: Frequency domain characteristics (oscillations)

The transformer attention mechanism learns optimal feature weights for classification.

## File Details

### Data Files
- **EEG Data**: Each `.npy` file contains shape (19, variable_time_steps)
  - 19 EEG channels
  - Variable-length time series at 128 Hz sampling rate

- **Train_Test_Split.csv**: Columns
  - `subject_id`: Identifier (sub-001 to sub-088)
  - `label`: AD or Healthy
  - `split`: train or test

- **subjects_metadata.csv**: Additional demographic/clinical information

### Model Files (.keras format)
- Trained weights compatible with TensorFlow/Keras
- Use `load_model()` to restore weights
- Models expect preprocessed input of shape (batch_size, 19, 1024)

## Configuration

Key hyperparameters in the fusion model:
- **Common dimension**: 128 (projection dimension)
- **Attention heads**: 4
- **Feed-forward dimension**: 256
- **Dropout rate**: 0.3
- **Normalization**: LayerNormalization with ε=1e-6

## Reproducibility

Random seeds are set for reproducibility:
```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
```

## Notes

- Segment length (1024 samples @ 128Hz) ≈ 8 seconds of EEG data
- Data is z-scored within each segment for normalization
- All models are frozen (trainable=False) in the fusion architecture for transfer learning
- The final fusion layer has softmax activation for binary classification

## Future Improvements

- [ ] Implement cross-validation for more robust evaluation
- [ ] Add explainability techniques (attention visualization)
- [ ] Extend to multi-class classification (MCI, vAD)
- [ ] Optimize for deployment on edge devices
- [ ] Conduct statistical significance testing on improvements

## Author Notes

This project implements a Bachelor's Thesis on Alzheimer's Disease detection. The ensemble approach demonstrates that combining multiple neural network architectures can improve classification performance for neurological disorder detection from EEG signals.

---

**Last Updated**: May 2026
