# Heart Disease Prediction — ML Model + Flask API + Render Deployment

A machine learning system that predicts whether a patient is at risk of heart
disease based on clinical parameters, exposed as a REST API and deployed as a
live web service.

**Live Demo (Render):** `<ADD YOUR RENDER URL HERE AFTER DEPLOYMENT>`
*(e.g. https://heart-disease-deployment.onrender.com)*

---

## 1. Problem Statement

A healthcare organization wants to deploy a machine learning model that
predicts whether a patient is at risk of heart disease based on clinical
parameters (age, blood pressure, cholesterol, ECG results, etc.).

---

## 2. Dataset

- Source: [Heart Disease Dataset — Kaggle (johnsmith88)](https://www.kaggle.com/datasets/johnsmith88/heart-diseasedataset)
- 13 clinical/numerical features + 1 binary target (`target`: 1 = disease
  present, 0 = no disease)
- Features: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
  oldpeak, slope, ca, thal`

> Note: `heart.csv` in this repo uses the standard UCI Cleveland heart-disease
> schema (identical 14 columns to the Kaggle dataset above). If you have
> downloaded the Kaggle CSV directly, simply replace `heart.csv` with it and
> re-run `train_model.py` — the column names match exactly, so no code
> changes are required.

---

## 3. Repository Structure

```
HeartDiseaseDeployment/
│
├── app.py                # Flask REST API
├── train_model.py        # Data preprocessing + model training script
├── model.pkl              # Trained classifier (RandomForest)
├── scaler.pkl              # Fitted StandardScaler used at inference time
├── feature_names.pkl       # Ordered list of expected input features
├── heart.csv                # Training dataset
├── requirements.txt        # Python dependencies
├── Procfile                # Render/gunicorn start command
├── .gitignore
├── README.md
├── templates/
│   └── index.html          # Optional simple web UI for the API
└── static/                 # (reserved for CSS/JS assets)
```

---

## 4. Task 1 — Data Understanding & Preprocessing

Implemented in `train_model.py`:

1. Loads `heart.csv` with Pandas.
2. Prints the first five records (`df.head()`).
3. Identifies:
   - **Numerical features:** `age, sex, cp, trestbps, chol, fbs, restecg,
     thalach, exang, oldpeak, slope, ca, thal`
   - **Target variable:** `target`
4. Checks for missing values with `df.isnull().sum()` — dataset has **no
   missing values**.
5. Splits data into **80% train / 20% test** using
   `train_test_split(..., test_size=0.20, random_state=42, stratify=y)`.

---

## 5. Task 2 — Model Development

- **Algorithm used:** Random Forest Classifier (`n_estimators=200,
  max_depth=6`)
- Features are standardized with `StandardScaler` before training.
- **Test Accuracy:** ~0.84 (83.6%) on the held-out 20% test split.
- Model, scaler, and feature-order list are persisted with **Joblib** as
  `model.pkl`, `scaler.pkl`, and `feature_names.pkl`.

To retrain from scratch:

```bash
pip install -r requirements.txt
python train_model.py
```

---

## 6. Task 3 — REST API (Flask)

`app.py` exposes:

| Method | Route      | Description                                   |
|--------|-----------|------------------------------------------------|
| GET    | `/`        | Health/info page, or simple HTML form           |
| GET    | `/health`  | Health check → `{"status": "ok"}`               |
| POST   | `/predict` | Accepts patient JSON, returns prediction JSON   |

### Example request

```bash
curl -X POST https://<your-render-url>/predict \
     -H "Content-Type: application/json" \
     -d '{
           "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
           "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
           "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
         }'
```

### Example response

```json
{
  "prediction": "Heart Disease Detected",
  "probability_of_heart_disease": 0.6804
}
```

### Run locally

```bash
pip install -r requirements.txt
python app.py
# API available at http://127.0.0.1:5000
```

---

## 7. Task 4 — GitHub & Render Deployment

### GitHub

This project lives as a **subfolder** inside the
[`AnshumanJ28/Assignments`](https://github.com/AnshumanJ28/Assignments)
repository, at `Assignments/HeartDiseaseDeployment/`.

To clone and run it locally:

```bash
git clone https://github.com/AnshumanJ28/Assignments.git
cd Assignments/HeartDiseaseDeployment
pip install -r requirements.txt
python app.py
```

(See the repo root for push instructions if you're updating this folder.)

### Render

1. Go to [render.com](https://render.com) and sign in (GitHub login is
   easiest).
2. Click **New → Web Service** and connect the `AnshumanJ28/Assignments`
   repository.
3. Configure:
   - **Root Directory:** `HeartDiseaseDeployment` (**required** — this repo
     contains other assignment notebooks at the root, so Render must be
     pointed at this subfolder to find `app.py`)
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app` (already declared in `Procfile`)
   - **Instance Type:** Free (or paid, to avoid the free-tier sleep/spin-down
     during evaluation)
4. Deploy. Render will build the service and give you a public URL such as
   `https://heart-disease-deployment.onrender.com`.
5. Verify it works:
   ```bash
   curl https://<your-render-url>/health
   curl -X POST https://<your-render-url>/predict -H "Content-Type: application/json" -d '{...}'
   ```
6. Paste the live URL at the top of this README.

> **Tip:** Free Render web services spin down after inactivity, causing the
> first request to be slow (cold start). For evaluation, either upgrade to a
> paid instance, or ping the `/health` endpoint a minute before evaluation to
> "wake" the service.

---

## 8. Task 5 — Conclusion

The Random Forest classifier achieved **~84% accuracy** on the held-out test
set, with strong recall for patients who do have heart disease — an
important property for a clinical screening tool, since missing a positive
case is costlier than a false alarm. Feature scaling and a modest tree depth
kept the model from overfitting the relatively small dataset.

The biggest deployment challenges were ensuring the exact same
preprocessing (feature order and scaling) used in training was reproduced at
inference time inside the Flask API, and packaging dependencies so the
Render build environment matched the local one exactly, since a version
mismatch in scikit-learn can break `joblib.load()`. Free-tier cold starts on
Render also required care so the service would respond promptly during
evaluation.

This project highlights why **MLOps** matters: a model is only useful once
it is versioned, reproducibly trained, packaged, and served reliably behind
an API with monitoring and CI/CD. Treating data prep, training, and
deployment as one pipeline — rather than a one-off notebook — is what turns
a good model into a dependable healthcare product.

---

## 9. Tech Stack

- Python, Pandas, scikit-learn, Joblib
- Flask, Gunicorn
- Render (deployment), GitHub (version control)
