# ML Teaching Notebooks

A collection of end-to-end, step-by-step machine learning notebooks for students, originally developed under the supervision of Dr. Rao Muhammad Adeel Nawab. Each project follows the same teaching structure (Import Libraries -> Load Data -> Preprocess -> Encode -> Train -> Test -> Deploy -> Collect Feedback) so students can compare approaches across problem types.

## Projects

### 1. Titanic Passenger Survival Prediction (Binary Classification)

Predicts whether a passenger survived the Titanic disaster from PClass, Gender, Sibling count, and Embarked port, using a Support Vector Classifier.

All files live in one folder, [`1 - Titanic Survival Prediction`](./1%20-%20Titanic%20Survival%20Prediction), since they share the same dataset, trained model, and `requirements.txt`. Two notebooks are included inside it:

| Notebook | Description |
|---|---|
| `Titanic_Passenger_Survival_Recommended.ipynb` | The original notebook with fixes: proper evaluation metrics (precision/recall/F1/confusion matrix) and a filled-in feedback section. Minimal changes, same modeling approach. |
| `Titanic_Passenger_Survival_FullEnhancement.ipynb` | Everything in the Recommended notebook, plus a data visualization step (survival rate by Gender/PClass) and a model comparison step (SVC vs. Logistic Regression vs. Decision Tree). |

Both achieve **0.80 accuracy** on the held-out test set and run top-to-bottom with zero errors.

### 3. Car Price Prediction with K-Fold Cross Validation (Regression)

Compares **5 regression algorithms** (Linear Regression, K-Nearest Neighbors, Decision Tree, Random Forest, Gradient Boosting) for predicting a car's price (MSRP), using **K-fold cross validation** and **Root Mean Squared Error (RMSE)**. Each notebook ends with per-fold and summary comparison tables, charts, and the best model trained on all the data.

All files live in [`3 - Car Price Prediction K-Fold Cross Validation`](./3%20-%20Car%20Price%20Prediction%20K-Fold%20Cross%20Validation). The notebooks are designed for **Google Colab**:

| Notebook | Data | Validation | Open |
|---|---|---|---|
| `car_price_kfold_100_instances_3fold.ipynb` | 100 instances | 3-fold | [Open in Colab](https://colab.research.google.com/github/haqnawaz99/ml-teaching-notebooks/blob/main/3%20-%20Car%20Price%20Prediction%20K-Fold%20Cross%20Validation/car_price_kfold_100_instances_3fold.ipynb) |
| `car_price_kfold_full_dataset_10fold.ipynb` | Full dataset (~11,900 cars) | 10-fold | [Open in Colab](https://colab.research.google.com/github/haqnawaz99/ml-teaching-notebooks/blob/main/3%20-%20Car%20Price%20Prediction%20K-Fold%20Cross%20Validation/car_price_kfold_full_dataset_10fold.ipynb) |

To run: open a notebook in Colab, click **Runtime → Run all**, and upload [`car_price.csv`](./3%20-%20Car%20Price%20Prediction%20K-Fold%20Cross%20Validation/car_price.csv) when Step 2.1 asks for it. The trained `.pkl` model is created when the notebook runs, so it is not stored in this repo.

| Algorithm | Mean RMSE – 100 cars, 3-fold | Mean RMSE – full data, 10-fold |
|---|---|---|
| Decision Tree | **$22,317** | $15,595 |
| Gradient Boosting | $25,122 | $18,675 |
| Random Forest | $30,100 | **$15,362** |
| K-Nearest Neighbors | $34,584 | $23,680 |
| Linear Regression | $40,466 | $33,534 |

With more data, every algorithm's error drops, and the best algorithm changes from Decision Tree to Random Forest.

## Try It Online (No Setup Required)

A live demo of the Titanic survival predictor is deployed on Streamlit Community Cloud — students can try it directly in a browser, no Python or Jupyter install needed:

**[Titanic Survival Predictor — Live App](https://ml-teaching-notebooks-jrrdwercxskw5kwc6rufvp.streamlit.app/)**

The app code lives in [`streamlit-app/`](./streamlit-app) and uses the trained model from the Recommended notebook.

## Getting Started (each project folder)

Every project folder is self-contained with its own `requirements.txt`. From inside a project folder:

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
# Windows (cmd):
venv\Scripts\activate.bat
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

Full setup instructions (including troubleshooting) are also included as the first section inside each notebook. The car price notebooks (project 3) run on Google Colab instead, so they need no local setup.

## Roadmap

More projects will be added here as they are finished:
- English Sentiment Analysis (Binary Classification)
- Car Price Prediction (Regression), single train/test split version
