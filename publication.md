# Deep Learning Based Satellite Image Classification Using Hyperspectral Data

## Abstract

This work focuses on automatically classifying different land types and crops in satellite images using deep learning techniques. We took hyperspectral satellite data from earth observation missions and built a complete system that learns to identify what each small area in the image represents - whether it's a corn field, a water body, or a road. Our approach uses a simple but effective neural network called SimpleCNN that learns patterns from training patches and then applies this knowledge to classify entire satellite maps. We tested our system on standard satellite datasets (Indian Pines, Salinas, and PaviaU) and created a web API for easy prediction. The entire work is documented and reproducible, making it suitable for academic research and practical applications.

---

## 1. Introduction - Why This Work Matters

### 1.1 The Real Problem

Think about this: a farmer in Punjab wants to know if his wheat crop is healthy this season. A city planner needs to track how much forest cover is left. A disaster management team needs to quickly identify areas affected by floods. How do these people get this information quickly and accurately?

The traditional way is to send people to check on the ground or use simple cameras. But this is slow, expensive, and sometimes risky. Since the 1970s, satellites have been orbiting Earth and taking pictures. But here's the challenge - these satellite images have **hundreds of different color channels** (not just Red-Green-Blue like normal photos), each telling something different about the land. Sorting through all this information manually is basically impossible.

### 1.2 Why Machine Learning?

Here's where the smart approach comes in. Instead of humans looking at each pixel and deciding "this is forest" or "this is water", we can train a computer model to do this automatically. The model learns by looking at thousands of examples: "In these images with these data patterns, it was always forest. In different patterns, it was always agricultural land." Once trained properly, this model can classify new satellite images in minutes, not weeks.

### 1.3 Why Deep Learning Specifically?

Deep learning (neural networks that have multiple layers) is particularly good at finding patterns in complex data. When you have hyperspectral data with 200+ channels, a simple mathematical formula cannot capture all the relationships. But a neural network with multiple layers can learn step-by-step: first layer learns simple color patterns, second layer combines those patterns, third layer recognizes complex objects. This is exactly what we need for satellite image classification.

---

## 2. Understanding the Data - What is Hyperspectral Imaging

### 2.1 Normal Cameras vs Hyperspectral Sensors

Let me explain with an easy comparison:

- **Your smartphone camera:** Takes 3 color photos - Red channel, Green channel, Blue channel. So one pixel in the image has 3 numbers (RGB values).

- **Hyperspectral satellite sensor:** Takes photos in 200+ narrow wavelength bands across visible light, near-infrared, shortwave-infrared, and thermal regions. So one pixel has 200+ numbers, each telling something slightly different.

Think of it like this: a normal photo is like a person described by 3 traits (height, weight, skin color). A hyperspectral image is like that person described by 200 traits (each tiny detail from top to bottom). The more details you have, the better you can identify them.

### 2.2 What These Channels Tell Us

Different materials reflect light differently at different wavelengths:
- **Green leaves** - they absorb red light but reflect a lot of near-infrared (this is how vegetation is detected)
- **Water** - absorbs most light except some blue wavelengths
- **Urban areas (concrete/brick)** - different pattern than vegetation
- **Soil** - different again

So if a scientist or computer looks at the pattern of values across all 200+ channels, it can figure out: "This is definitely vegetation" or "This is definitely water", because the pattern is unique.

### 2.3 The Datasets We Used

We worked with three standard satellite datasets that all researchers use:

**Indian Pines Dataset:**
- Collected over agricultural area in Indiana, USA
- Size: 145 rows × 145 columns (21,025 pixels total)
- Spectral bands: 200 channels
- Classes: 16 different land types (corn, soybeans, grass pasture, trees, etc.)
- Ground truth: Expert labeled map showing what each pixel actually is

**Salinas Dataset:**
- Collected over agricultural valley in California, USA  
- Size: 512 × 217 pixels
- Spectral bands: 224 channels
- Classes: 16 agricultural crops and classes
- Ground truth: Already labeled training map

**Pavia University Dataset:**
- Collected over urban area in Italy
- Size: 610 × 340 pixels
- Spectral bands: 103 channels (some were removed as noise)
- Classes: 9 different urban classes (asphalt, concrete, trees, etc.)
- Ground truth: Labeled by domain experts

All three datasets are freely available and used worldwide in research. This makes our results comparable and reproducible.

### 2.4 What is "Ground Truth"?

Ground truth is the **correct answer** for each pixel. It's like the answer key in an exam. For satellite data, domain experts (remote sensing scientists) manually look at the satellite image combined with field visits and other sources, and label each pixel: "This one is definitely wheat", "This one is definitely water", etc.

In our `.mat` files:
- `Indian_pines_corrected.mat` - contains the 200 spectral channels for each pixel
- `Indian_pines_gt.mat` - contains a map where each pixel has a number (1-16) indicating the true class

The ground truth is crucial because without it, the computer cannot learn.

---

## 3. Our Approach - The Complete Pipeline

### 3.1 What is an "End-to-End Pipeline"?

End-to-end means: starting from raw data files → finishing with predictions. Every step is connected like a chain. If any step is weak, the final result suffers. Our pipeline has these clear steps:

```
Raw .mat Files 
    ↓
Loading & Basic Checks
    ↓
Preprocessing (Normalization + PCA)
    ↓
Patch Extraction
    ↓
Train/Validation/Test Split
    ↓
Neural Network Training
    ↓
Model Saving
    ↓
Predictions on New Data
    ↓
API for Easy Access
```

Each step solves a specific problem. Let me explain why each is necessary.

### 3.2 Step 1: Loading & Understanding Data

First, we load the `.mat` files (MATLAB format, commonly used in research). We check:
- Is the data 3D? (height, width, spectral_channels)
- Is the ground truth 2D? (height, width with class labels)
- Do the sizes match?

This might seem obvious, but in real work, files are sometimes corrupted or have unexpected formats. Checking early saves hours of debugging.

### 3.3 Step 2: Why Preprocessing is Critical

**The problem:** Imagine you have 200 columns of data. Column 1 has values 0-100. Column 50 has values 0-10000. Our neural network will think column 50 is "more important" just because the numbers are bigger. This is wrong.

**Our solution:** We do two things:

**Normalization:** Scale all channels to 0-1 range using MinMax scaling. Formula:
```
normalized_value = (value - min) / (max - min)
```
Now every channel is equally important numerically.

**PCA (Principal Component Analysis):** Our 200 channels have a lot of redundancy. PCA finds 30 most important "combinations" of these channels that carry 95% of the information. This does two things:
- Reduces computation time (200 → 30 is 6.67x faster)
- Sometimes removes noise (less important channels often contain noise)

Think of it like: if you have 200 witnesses to an event, but 30 of them saw everything clearly while the others just noticed small details or were confused, you'd focus on those 30 main witnesses.

### 3.4 Step 3: Patch Extraction - Why We Slice the Image

Here's an important concept: we don't train the network on individual pixels. We train on **small patches** (9×9 pixel squares in our case).

Why? Single pixels are too small to contain useful information. A 9×9 patch (81 pixels total) gives context: "Is this pixel in the middle of a field, or at the edge between two fields?" This spatial context is crucial.

Process:
1. For each labeled pixel in the image, take a 9×9 region centered on it
2. Convert the 3D patch (9×9×30 channels) into a vector (810 numbers)
3. Pair it with the label ("This is corn" = class 1, etc.)

Result: thousands of training examples, each exactly the same size and format.

- **Indian Pines:** ~10,000 patches extracted from ~13,000 labeled pixels
- Patches are balanced by class (we use stratified splitting to ensure training data has all classes well-represented)

### 3.5 Step 4: Data Split - Why 70/15/15?

Different sets serve different purposes:

**Training Set (70%):**
- The network learns from this
- Sees millions of times during training
- Updates its weights based on mistakes

**Validation Set (15%):**
- Network doesn't learn from this
- Used to check: "Is the network actually improving?"
- Tells us when to stop training (if validation accuracy drops while training accuracy increases, the network is overfitting)

**Test Set (15%):**
- Network never sees this data, ever
- Used to measure the real-world performance
- Simulates: "Will this network work on new satellite images?"

We use **stratified split**, meaning each set gets roughly equal proportions of every class. If agricultural class is 5% of all pixels, it should be 5% in train, validation, and test.

Why 70/15/15 and not 80/10/10? With smaller datasets like Indian Pines (~13K labeled pixels), 70/15/15 gives enough validation and test samples for reliable estimates. With larger datasets, 80/10/10 works too.

---

## 4. Our Research Objective

**Goal:** To develop an automated, reproducible system for hyperspectral satellite image classification that starts from raw data files and produces predictions through a modern web API, demonstrating high accuracy while maintaining complete code transparency for academic research.

**Specific contributions of this work:**
1. **Complete pipeline documentation** - Every step from data loading to predictions is explicit and reproducible
2. **Modular, testable code** - Services, utilities, and routers separated for clarity and reusability
3. **Practical API endpoint** - Predictions accessible through a web service (FastAPI)
4. **Metadata tracking** - Each trained model saves information about how it was trained (seed, PCA components, split ratios) for reproducibility

---

## 5. Technical Foundation - The SimpleCNN Model

### 5.1 What is a CNN and Why It Works Here

CNN = Convolutional Neural Network. The idea is simple:

- **Convolution layer:** Applies small "filters" (like magnifying glasses) across the patch. Each filter looks for different patterns (e.g., sharp edges, smooth regions, specific color combinations).
- **ReLU activation:** Just makes negative values zero (simple non-linearity, allows learning of complex patterns)
- **Pooling layer:** Compresses information (downsamples the spatial size, keeping only the strongest signals)
- **Fully connected layer:** Converts the learned features into class probabilities

**Why for satellite data?** Spatial relationships matter. A pixel's class depends on its neighborhood, not just itself. Convolutions are specifically designed to learn these local patterns.

### 5.2 Our SimpleCNN Architecture

We use a straightforward architecture:

```
Input: 30×9×9 patch (30 spectral channels, 9×9 spatial)
  ↓
Conv2D (32 filters, 3×3 kernel) → ReLU → MaxPool (2×2)
  ↓
Conv2D (64 filters, 3×3 kernel) → ReLU → MaxPool (2×2)
  ↓
Flatten → Dense (128) → ReLU → Dropout (0.5)
  ↓
Dense (num_classes) → Softmax
  ↓
Output: Probability for each class
```

- **32 filters in layer 1:** Learn 32 different simple patterns
- **64 filters in layer 2:** Combine those patterns into 64 more complex patterns
- **Dropout(0.5):** Random 50% of connections disabled during training to prevent overfitting
- **Softmax:** Converts logits to probabilities (sums to 1)

Total trainable parameters: ~50,000. Small enough to train quickly, large enough to learn patterns.

---

## 6. Training Strategy

### 6.1 How the Network Learns

Training works in iterations:
1. **Forward pass:** Feed a batch of patches through the network → get predictions
2. **Loss calculation:** Measure how wrong the predictions are using Cross-Entropy Loss (standard for classification)
3. **Backward pass:** Calculate gradients (which direction to adjust weights)
4. **Update weights:** Adam optimizer adjusts all weights slightly to reduce loss

This repeats thousands of times. With 10,000 patches and batch size 64:
- Each epoch = ~156 iterations
- 10 epochs = ~1,560 iterations of weight updates

### 6.2 Validation During Training

After each epoch:
- Measure accuracy on validation set
- If validation accuracy improves → save the model weights
- If validation accuracy decreases for N consecutive epochs → stop training (early stopping)

This ensures we get the best model, not an overfitted one.

### 6.3 Why These Hyperparameters?

- **Learning rate = 0.001:** Standard for Adam optimizer. Too high = unstable, too low = slow convergence
- **Batch size = 64:** Balance between memory usage and gradient stability
- **Epochs = 10:** Enough for convergence on smaller datasets (you can increase for better results)
- **PCA components = 30:** Reduces from 200 to 30 channels, removing noise while keeping 95%+ information

---

## 7. Evaluation and Predictions

### 7.1 How We Measure Success

On test set, we calculate:

**Overall Accuracy (OA):** `(correct predictions) / (total test samples)`

The most basic metric. Simple to understand: "Out of 100 test images, how many did the network get right?"

**Per-Class Accuracy:** For each class separately

Important because some classes might be easy to classify (large fields) while others are hard (small scattered areas). If the network gets 80% overall but only 20% on a rare class, that's not good.

**Average Accuracy (AA):** Average of per-class accuracies

If you have 16 classes and get 70% on class 1, 75% on class 2, ..., AA is the average of all 16.

### 7.2 Full Image Predictions

After training, the model can classify a complete 145×145 image in one go:

1. Extract all overlapping 9×9 patches from the image
2. Feed each through the network
3. Collect all predictions into a 145×145 map
4. This becomes our predicted classification map

Compare this with true ground truth → get all the accuracy metrics.

---

