# Satellite Image Classification Project

## 📁 Project Structure

```
app/
  routers/
    __init__.py
    predicting.py        # FastAPI endpoint for inference
  schemas/
    __init__.py
    predicting_schemas.py# Pydantic models for request/response
  services/
    __init__.py
    pytorch_training.py  # training and inference logic
  utils/
    __init__.py
    global_utils.py      # helpers for file I/O, dataset loading
  config.py             # settings and environment config
  model_save.py         # wrappers around saving utilities

main.py                 # entry point for the FastAPI application
.env                    # environment variables (model & data paths)
requirements.txt        # Python dependencies
```

> The `app` package contains all the domain logic, while `main.py` boots
> up the HTTP server and wires routers.

---

## 🧠 Data & Problem Overview

We are working with **hyperspectral satellite imagery** datasets (e.g. Indian
Pines, PaviaU, Salinas). Each dataset consists of multi-band `.mat` files
containing tens to hundreds of spectral channels, plus ground-truth maps
(`*_gt.mat`) with classification labels for each pixel.

The goal is to train a deep learning model (PyTorch) that can take a patch or
full image as input and output a land-cover class label for each pixel or
region.

---

## 🔄 End‑to‑End Pipeline

1. **Data ingestion** – use helpers in `app.utils.global_utils` to load `.mat`
   files and build `torch.utils.data.Dataset`/`DataLoader` objects.
2. **Model training** – `app.services.pytorch_training.train_model` encapsulates
   training loops, loss computation, and checkpointing. Trained weights are
   persisted via `app.services.pytorch_training.save_model` or
   `app.model_save`.
3. **Model storage** – serialized `.pth` files live in the directory specified
   by the `MODEL_DIR` environment variable (`models/` by default).
4. **Serving predictions** – a FastAPI server (`main.py`) exposes an endpoint
   under `/api/predict` that accepts JSON with paths to an image and a model
   file. The request body is validated by Pydantic schemas in
   `app.schemas.predicting_schemas.py`.
5. **Inference logic** – the router calls `load_model`/`predict_image` from
   `app.services.pytorch_training` to run a forward pass and return a label.

> This modular layout makes it easy to reuse training code offline, while the
> API only needs a lightweight dependency on the same service functions.

---

## 🚀 Getting Started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Populate `.env`** with dataset and model directories (already stubbed).
3. **Train a model** by writing a simple Python script that imports
   `train_model` from `app.services.pytorch_training` and supplies a dataset.
   Example:
   ```python
   from app.services.pytorch_training import train_model, save_model
   from app.utils.global_utils import load_mat_file

   data = load_mat_file('Indian_pines_corrected.mat')
   model = train_model(data, epochs=50, lr=1e-3)
   save_model(model, 'models/indian_pines.pth')
   ```
4. **Start the API**
   ```bash
   uvicorn main:app --reload
   ```
5. **Call the prediction endpoint**
   ```bash
   curl -X POST "http://localhost:8000/api/predict" \
        -H "Content-Type: application/json" \
        -d '{"image_path":"path/to/sample.mat","model_path":"models/indian_pines.pth"}'
   ```

---

## 📄 For Your Paper/Publication

The repository is built with clarity and reproducibility in mind:

- **Modular architecture:** separating routing, schema validation,
  service logic, and utilities supports both experimentation and deployment.
- **Data-centric design:** utilities focus on standard hyperspectral file
  formats (.mat), making it easy to adapt to new datasets.
- **Clear pipeline:** from raw data to model weights to API inference, each
  step is documented and testable.

Include a figure in your paper showing the folder layout and data flow (e.g.
train script producing a `.pth` file, then FastAPI serving predictions). The
README here can serve as a supplemental material describing implementation
details.

---

## 📌 Future Enhancements

- Add unit tests for each module and an integration test for the API.
- Expand `pytorch_training` with dataset classes, augmentation, evaluation
  metrics, and checkpoint/resume support.
- Implement caching, authentication or batch prediction endpoints.

---

Feel free to tailor the structure and narrative to match your publication
requirements. Good luck with the paper! 😊

---

## Experimental Protocol (Suggested for Publication)

1. Dataset selection: choose one of the provided datasets (Indian Pines, Salinas, PaviaU). Document the `.mat` filenames, spatial resolution, number of bands and any bands removed.
2. Preprocessing: apply MinMax normalization across pixels per band and optionally reduce spectral bands using PCA to `n_components` (30 is common). Record the random seed and PCA explained variance.
3. Patch extraction: extract centered square patches (e.g., 9×9) from labeled pixels only (ignore label 0). Convert labels to zero-based indexing internally.
4. Split: stratified train/val/test split by pixel (common split: 70/15/15), seed fixed for reproducibility.
5. Model: document model architecture (example: `SimpleCNN` or ViT used in notebook), number of parameters and initialization.
6. Training: specify optimizer (Adam), initial learning rate, scheduler, batch size, number of epochs, and class weighting strategy. Save best checkpoint by validation accuracy.
7. Evaluation: report Overall Accuracy (OA), Average Accuracy (AA), Cohen's Kappa, and per-class accuracies. Include confusion matrix and visual predicted map.

## Reproducibility Checklist

- Code: share this repository (or supplemental code archive) with a README and a requirements file.
- Data: specify dataset filenames and exact pre-processing steps (including PCA parameters).
- Random seeds: include seeds for Python, NumPy and PyTorch and note deterministic CuDNN settings if used.
- Environment: state Python version and key library versions (`torch`, `numpy`, `scikit-learn`). Use `pip freeze` output when possible.
- Checkpoints: provide the final `.pth` files and accompanying metadata JSON describing dataset and training run.

## Running Tests & CI

Run unit tests locally with:

```bash
pip install -r requirements.txt
pytest -q
```

The repository includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs tests on pushes and pull requests.

## Docker (quick start)

Build the container and run the API locally:

```bash
docker build -t satellite-api:latest .
docker run -p 8000:8000 satellite-api:latest
```

Then call the API at `http://localhost:8000/api/predict` as described earlier.
