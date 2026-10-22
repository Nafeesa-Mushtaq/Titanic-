# Titanic Passenger Survival Prediction

**Live app:** [Titanic Survival Predictor](https://2vqhfvhbaaaxnuxmvnws68.streamlit.app/)

An educational machine learning project for **Assignment 2: Developing an End-to-End Machine Learning System**. This project runs and evaluates the instructor-provided Titanic FullEnhancement notebook and deploys its trained Support Vector Classifier (SVC) using Streamlit.

## Inputs and Output

| Input | Available values |
|---|---|
| Passenger class (`PClass`) | First, Second, Third |
| Gender (`Gender`) | Male, Female |
| Sibling count (`Sibling`) | Zero, One, Two, Three |
| Embarkation port (`Embarked`) | Southampton, Cherbourg, Queenstown |

The model predicts **Survived (1)** or **Not Survived (0)**.

## Dataset and Workflow

The notebook uses the instructor-provided 100-row teaching dataset, `sample-data.csv`.

The workflow includes loading and inspecting data, visualizing survival by gender and passenger class, encoding categorical inputs and the target, splitting training and testing data, training SVC, comparing classifiers, evaluating predictions, saving the model, and deploying a web application.

The application reproduces the notebook's label encodings and supplies inputs in the order `PClass`, `Gender`, `Sibling`, `Embarked`.

## Evaluation Results

SVC achieved **80% accuracy**, correctly classifying 16 of 20 held-out test records.

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Not Survived (0) | 0.71 | 1.00 | 0.83 | 10 |
| Survived (1) | 1.00 | 0.60 | 0.75 | 10 |

| Actual / Predicted | Not Survived (0) | Survived (1) |
|---|---|---|
| Not Survived (0) | 10 | 0 |
| Survived (1) | 4 | 6 |

### Model Comparison

| Model | Test accuracy |
|---|---|
| SVC | 80% |
| Logistic Regression | 75% |
| Decision Tree | 80% |

SVC was retained for deployment, following the original notebook workflow. These comparisons use a small shared test set and do not establish that any classifier will perform best on new datasets.

## Project Files

| File | Purpose |
|---|---|
| `Titanic_Passenger_Survival_FullEnhancement.ipynb` | Training, evaluation, feedback, and conclusion |
| `sample-data.csv` | Teaching dataset |
| `svc_trained_model.pkl` | Saved SVC model |
| `app.py` | Streamlit interface |
| `requirements.txt` | Application dependencies |

Notebook and dataset files may be kept separately from the deployment folder. Keep `app.py`, `requirements.txt`, and `svc_trained_model.pkl` together.

## Run Locally

Use the project's Python 3.13 environment. From the folder containing `app.py`, run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the Local URL printed in the terminal, choose passenger details, and click **Predict Survival**.

To regenerate the model, run the notebook with its dataset available, then copy the generated `svc_trained_model.pkl` into the app folder. Use the same scikit-learn version during training and deployment.

## Deployment

The app is deployed through Streamlit Community Cloud from GitHub using Python 3.13. The entrypoint is `app.py` if the files are in the repository root, or `streamlit-app/app.py` if they are inside that folder.

The public deployment link is also included at the top of the submitted notebook.

## Feedback and Limitations

Local testing confirmed that the application displays predictions. Evaluation identified four actual survivors that the model predicted as not survived. Future improvements include testing more passenger combinations, using a larger dataset, investigating incorrect predictions, and tuning parameters using cross-validation on training data.

The 100-row teaching sample and 20-record test set limit the reliability and generalizability of the results. Predictions are for educational demonstration and should not be interpreted as definitive historical conclusions.

## Acknowledgment

The notebook, teaching dataset, and original application were provided through the instructor's [ML Teaching Notebooks repository](https://github.com/haqnawaz99/ml-teaching-notebooks). This assignment builds on that supplied work.
