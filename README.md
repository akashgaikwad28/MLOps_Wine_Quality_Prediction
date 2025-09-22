
# 🍷 Wine Quality Prediction — End-to-End MLOps Pipeline

This project demonstrates a complete machine learning workflow for predicting wine quality using physicochemical properties. It follows MLOps best practices with modular pipeline design, experiment tracking, version control, CI/CD automation, and collaborative tooling.

---

## 🚀 Project Overview

- **Goal**: Predict wine quality (score 0–10) based on chemical features
- **Model**: Supervised learning using Scikit-learn
- **Pipeline Stages**:
  - Data Ingestion
  - Data Validation
  - Data Transformation
  - Model Training
  - Model Evaluation
  - Prediction & Deployment

---

## 🧠 Key Features

- ✅ Modular pipeline using Python OOP
- ✅ Configuration-driven architecture (YAML)
- ✅ Schema-based data validation
- ✅ Unit testing with Pytest
- ✅ CI/CD with GitHub Actions
- ✅ Experiment tracking with MLflow
- ✅ Data & model versioning with DVC
- ✅ Remote collaboration via DagsHub

---

## 🛠️ Tech Stack

| Category             | Tools Used                          |
|----------------------|-------------------------------------|
| Language             | Python                              |
| ML Framework         | Scikit-learn                        |
| Experiment Tracking  | MLflow                              |
| Versioning           | DVC                                 |
| Collaboration        | DagsHub                             |
| CI/CD                | GitHub Actions                      |
| Testing              | Pytest                              |
| Configuration        | YAML                                |
| Deployment           | Docker, FastAPI (planned)           |

---

## 📁 Project Structure

```
MLOps_Wine_Quality_Prediction/
│
├── src/
│   └── Wine_Quality_Prediction/
│       ├── components/
│       ├── config/
│       ├── entity/
│       ├── pipeline/
│       └── utils/
│
├── tests/
│   └── unit/
│
├── params.yaml
├── schema.yaml
├── requirements.txt
├── setup.py
├── Dockerfile
├── app.py
└── main.py
```

---

## ⚙️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/akashgaikwad28/MLOps_Wine_Quality_Prediction.git
cd MLOps_Wine_Quality_Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python main.py
```

---

## 🧪 Run Tests

```bash
pytest tests/
```

---

## 📊 MLflow Tracking

- All experiments are logged using MLflow.
- Metrics, parameters, and artifacts are tracked and visualized.
- To launch MLflow UI:
```bash
mlflow ui
```

---

## 📦 DVC Integration

- Data and model artifacts are versioned using DVC.
- Remote storage is configured via DagsHub.
- To pull latest data:
```bash
dvc pull
```

---

## 🌐 DagsHub Collaboration

- Git + DVC + MLflow integration
- Remote tracking of experiments and data
- Project hosted at: [DagsHub Repository](https://dagshub.com/akashgaikwad28/MLOps_Wine_Quality_Prediction)

---

## 📈 Model Performance

- Accuracy: *X.XX* (replace with actual)
- F1 Score: *X.XX*
- Evaluation metrics tracked via MLflow

---

## 📌 Future Enhancements

- [ ] Dockerize the pipeline
- [ ] Deploy model via FastAPI
- [ ] Add monitoring for data drift
- [ ] Integrate model registry

---

## 👨‍💻 Author

**Akash Gaikwad**  
Data Scientist & MLOps Engineer  
[GitHub Profile](https://github.com/akashgaikwad28)

