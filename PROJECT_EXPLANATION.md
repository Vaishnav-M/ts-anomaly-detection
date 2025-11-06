# PROJECT SETUP SUMMARY & DETAILED EXPLANATION

## 📋 What Has Been Set Up

### ✅ Project Structure Created
```
ts-works-assignment/
├── venv/                          # Virtual environment (Python 3.10)
├── data/
│   ├── raw/                       # Place your datasets here
│   └── processed/                 # Cleaned/processed data goes here
├── notebooks/                     # Jupyter notebooks for analysis
├── src/                          # Python source code modules
│   ├── __init__.py
│   └── models/
│       └── __init__.py
├── outputs/
│   ├── plots/                    # Visualizations will be saved here
│   ├── models/                   # Trained models will be saved here
│   └── results/                  # Metrics and reports go here
├── .gitignore                    # Git ignore configuration
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── verify_setup.py              # Setup verification script
```

### ✅ Virtual Environment (venv)
- **Python Version**: 3.10.0 (perfect for compatibility)
- **Name**: `venv` (as requested)
- **Status**: Created and activated
- **Location**: `e:\workout_programs\VMS_PROJECTS\ts-works-assignment\venv`

### ✅ Dependencies Being Installed
The following packages are being installed:
- **Data Processing**: numpy, pandas, scipy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn (Isolation Forest, LOF)
- **Deep Learning**: TensorFlow 2.13 (with GPU support)
- **Time Series**: statsmodels
- **Development**: Jupyter, notebook, ipykernel
- **Utilities**: tqdm, joblib, python-dateutil

---

## 🎯 DETAILED PROJECT EXPLANATION

### Problem Statement
You're building an **anomaly detection system** for IoT sensors in a manufacturing facility. The goal is to identify unusual sensor readings that indicate:
- Equipment failure
- Maintenance needs
- Operational problems

Think of it like a "health monitoring system" for machines.

---

## 🏗️ ARCHITECTURE & APPROACH

### Phase 1: Data Preparation & Exploration
**What**: Load sensor data, clean it, understand patterns

**Key Activities**:
1. **Load Dataset**: Import time series sensor readings
2. **Data Quality Check**:
   - Missing values? → Interpolate or forward-fill
   - Outliers? → Identify using statistical methods
   - Inconsistent timestamps? → Resample to regular intervals
3. **Exploratory Data Analysis (EDA)**:
   - Plot time series to see patterns
   - Check distributions (normal? skewed?)
   - Look for seasonality (daily/weekly patterns)
   - Compute basic statistics (mean, std, min, max)

**Output**: Clean dataset + understanding of normal behavior

---

### Phase 2: Feature Engineering
**Why**: Raw sensor readings aren't enough. We need to create features that capture patterns.

**Feature Types**:

1. **Rolling Statistics** (capture trends):
   - Rolling mean (average over last N readings)
   - Rolling std (volatility over last N readings)
   - Rolling min/max (range over last N readings)
   - Example: `rolling_mean_5 = sensor_value.rolling(window=5).mean()`

2. **Lag Features** (capture history):
   - Previous values: `lag_1`, `lag_2`, etc.
   - Rate of change: `value - previous_value`

3. **Time-based Features**:
   - Hour of day (0-23)
   - Day of week (0-6)
   - Is weekend? (0 or 1)

4. **Domain-specific**:
   - For vibration sensors: FFT features (frequency domain)
   - For temperature: rate of temperature change
   - Interaction features: sensor1 * sensor2

**Output**: Enhanced dataset with 20-30 features instead of just raw values

---

### Phase 3: Anomaly Detection Models

#### 🔷 **Approach 1: Statistical/Unsupervised Methods**

##### **Model 1: Isolation Forest**
**Intuition**: 
- Normal data points are clustered together
- Anomalies are isolated and far from clusters
- Algorithm randomly creates decision trees
- Anomalies require fewer splits to isolate
- Anomaly score = how easily a point is isolated

**Why it works**:
- Doesn't assume data distribution
- Fast and scalable
- Works well for high-dimensional data
- Natural outlier detector

**Hyperparameters**:
- `n_estimators`: Number of trees (100-500)
- `contamination`: Expected % of anomalies (0.01-0.1)
- `max_features`: Features per tree (1.0 = all features)

##### **Model 2: Local Outlier Factor (LOF)**
**Intuition**:
- Compares local density of a point to its neighbors
- Normal points: similar density to neighbors
- Anomalies: much lower density (isolated)
- LOF score > 1 = anomaly

**Why it works**:
- Captures local anomalies (context-aware)
- Good for varying density clusters
- Finds subtle anomalies

**Hyperparameters**:
- `n_neighbors`: How many neighbors to compare (20-50)
- `contamination`: Expected % of anomalies
- `metric`: Distance metric (euclidean, manhattan)

---

#### 🔷 **Approach 2: Deep Learning (LSTM Autoencoder)**

**Intuition**:
1. **Training Phase**:
   - Feed normal data to autoencoder
   - Network learns to compress and reconstruct normal patterns
   - Learns what "normal" looks like

2. **Detection Phase**:
   - Feed new data point
   - Autoencoder tries to reconstruct it
   - Normal data → low reconstruction error
   - Anomaly → high reconstruction error (network never learned this pattern)

**Architecture**:
```
Input → LSTM Encoder → Bottleneck → LSTM Decoder → Output
      (compress)      (latent)      (reconstruct)

Example:
Input: [10 features]
Encoder: LSTM(64) → LSTM(32) → Dense(16)  ← bottleneck
Decoder: Dense(32) → LSTM(32) → LSTM(64) → Dense(10)
```

**Why LSTM?**
- Captures temporal dependencies
- Remembers patterns over time
- Understands sequences (t-5, t-4, ..., t)

**Why it works**:
- Learns complex non-linear patterns
- Captures temporal relationships
- No manual feature engineering needed
- Can detect subtle anomalies

**Hyperparameters**:
- `sequence_length`: How many time steps (30-100)
- `encoding_dim`: Bottleneck size (8-32)
- `epochs`: Training iterations (50-200)
- `batch_size`: Samples per update (32-128)
- `learning_rate`: Step size (0.001-0.01)

**Loss Function**: Mean Squared Error (MSE)
**Threshold**: 95th percentile of reconstruction errors on normal data

---

### Phase 4: Model Evaluation

#### **Metrics** (if we have labels):
- **Precision**: Of flagged anomalies, how many are real?
  - Formula: `TP / (TP + FP)`
  - Important when false alarms are costly

- **Recall**: Of real anomalies, how many did we catch?
  - Formula: `TP / (TP + FN)`
  - Important when missing anomalies is costly

- **F1-Score**: Balance between precision and recall
  - Formula: `2 * (Precision * Recall) / (Precision + Recall)`

- **AUC-ROC**: Overall discrimination ability

#### **Validation** (if we don't have labels):
1. **Visual Inspection**: Plot detected anomalies, do they look unusual?
2. **Domain Reasoning**: Do anomalies align with known events?
3. **Clustering**: Do anomalies form separate clusters?
4. **Statistical Tests**: Are anomalies statistically significant?

#### **Comparison**:
- Statistical methods: Fast, interpretable, good for simple patterns
- Deep learning: Captures complex patterns, better for multivariate data

---

## 🎮 YOUR GPU & HARDWARE

You mentioned:
- **GPU**: 8GB (excellent for training LSTM Autoencoder!)
- **RAM**: 16GB (sufficient for this project)

**TensorFlow will automatically use your GPU** for:
- Training the LSTM Autoencoder (5-10x faster than CPU)
- Batch processing during inference

**Memory Management**:
- Your dataset should fit in RAM easily
- For very large datasets, use batch processing
- TensorFlow will use 6-7GB of GPU memory

---

## 📊 DATASET RECOMMENDATIONS

### **Option 1: NASA Bearing Dataset** ✅ RECOMMENDED
**Why**: Real industrial data, perfect for this use case

**Details**:
- **Source**: https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset
- **Type**: Vibration sensor data from bearings
- **Features**: Multiple sensor channels
- **Size**: ~150MB
- **Anomalies**: Bearing degradation over time
- **Use Case**: Perfect match for your assignment

**What you'll detect**: Bearing wear and approaching failure

---

### **Option 2: AWS Server Metrics**
**Details**:
- **Source**: https://github.com/numenta/NAB/tree/master/data
- **Type**: Server CPU, memory, latency metrics
- **Features**: Multiple metrics over time
- **Size**: Small (~10MB)
- **Anomalies**: Labeled anomalies included!

---

### **Option 3: Generate Synthetic Data**
**Approach**:
```python
# Normal sine wave + noise
normal = np.sin(2 * np.pi * t) + noise

# Inject anomalies
anomalies = random_spikes + random_drops
```

**Pros**: Full control, guaranteed anomalies
**Cons**: Less realistic than real data

---

## 🚀 NEXT STEPS (After Installation Completes)

### Step 1: Verify Setup
```powershell
# Run verification script
E:/workout_programs/VMS_PROJECTS/ts-works-assignment/venv/Scripts/python.exe verify_setup.py
```

This checks:
- Python version
- All packages installed
- GPU availability
- Directory structure

### Step 2: Download Dataset
1. Go to https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset
2. Download the dataset
3. Extract to `data/raw/bearing_dataset/`

### Step 3: Start Jupyter
```powershell
# Make sure you're in the project directory
cd e:\workout_programs\VMS_PROJECTS\ts-works-assignment

# Start Jupyter
E:/workout_programs/VMS_PROJECTS/ts-works-assignment/venv/Scripts/jupyter.exe notebook
```

### Step 4: Create Notebooks (I'll help you with this next!)
We'll create 5 notebooks:
1. `01_data_exploration.ipynb` - Load, clean, EDA
2. `02_feature_engineering.ipynb` - Create features
3. `03_statistical_models.ipynb` - Isolation Forest, LOF
4. `04_deep_learning_models.ipynb` - LSTM Autoencoder
5. `05_model_comparison.ipynb` - Compare & visualize

---

## 📈 PROJECT TIMELINE (4-8 hours)

### Hour 1-2: Data Preparation
- Load dataset
- Handle missing values
- EDA with plots
- Document findings

### Hour 3-4: Feature Engineering
- Create rolling features
- Create lag features
- Normalize/scale
- Save processed data

### Hour 5-6: Model Development
- Implement Isolation Forest
- Implement LOF
- Implement LSTM Autoencoder
- Train models

### Hour 7-8: Evaluation & Documentation
- Compare models
- Create visualizations
- Write summary document
- Finalize README

---

## 🔧 DEVELOPMENT WORKFLOW

### Git Workflow:
```powershell
# Initialize git (if not done)
git init
git add .
git commit -m "Initial project setup"

# After each major milestone
git add .
git commit -m "Completed data exploration"
```

### Coding Workflow:
1. **Explore in notebooks** (interactive, visual)
2. **Move working code to src/** (modular, reusable)
3. **Test in notebooks** (validate)
4. **Document** (comments, docstrings)

---

## 📝 DELIVERABLES CHECKLIST

- [ ] Jupyter notebooks with clear documentation
- [ ] Python modules in `src/` (production-ready)
- [ ] Summary document (2-3 pages):
  - [ ] Problem understanding
  - [ ] Feature engineering rationale
  - [ ] Model selection & comparison
  - [ ] Key findings
  - [ ] Limitations & improvements
- [ ] Visualizations (minimum 3-4):
  - [ ] Time series with anomalies
  - [ ] Feature distributions
  - [ ] Model comparison charts
  - [ ] Reconstruction error plots
- [ ] README with setup instructions
- [ ] Requirements.txt with dependencies

---

## 💡 KEY INSIGHTS TO INCLUDE

When writing your summary document, explain:

1. **Why Isolation Forest?**
   - Fast, scalable, no assumptions about distribution
   - Works well for high-dimensional data

2. **Why LSTM Autoencoder?**
   - Captures temporal patterns
   - Learns complex relationships
   - Works well for multivariate time series

3. **Feature Engineering Rationale**:
   - Rolling statistics capture trends
   - Lag features capture history
   - Time features capture seasonality

4. **Business Value**:
   - Early warning system for equipment failure
   - Reduces downtime and maintenance costs
   - Enables predictive maintenance

---

## ⚠️ IMPORTANT NOTES

1. **Git Ignore**: Large files (data, models) won't be committed
2. **GPU Usage**: TensorFlow will auto-detect and use your GPU
3. **Modularity**: Write reusable functions in `src/`
4. **Documentation**: Comment your code thoroughly
5. **Error Handling**: Add try-except blocks for robustness

---

## 🎓 LEARNING RESOURCES

While working:
- Isolation Forest paper: Liu et al., 2008
- LSTM Autoencoders: Malhotra et al., 2016
- Time Series Anomaly Detection: surveys available online

---

## 🤝 NEXT INTERACTION

Once the installation completes, tell me:
1. "Setup complete, let's start coding!"
2. Which dataset you want to use (NASA/AWS/Synthetic)
3. Any specific requirements or constraints

Then I'll help you:
1. Create the first notebook (data exploration)
2. Build the Python modules
3. Implement the models
4. Create visualizations
5. Write the summary document

---

## 📞 QUESTIONS TO ASK ME

Feel free to ask:
- "How do I implement [specific feature]?"
- "What's the best approach for [specific problem]?"
- "Can you explain [concept] in more detail?"
- "Help me debug this error"
- "How should I structure [component]?"

I'm here to help you build an excellent project! 🚀
