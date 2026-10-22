# Titanic Survival Predictor — Streamlit App

A small web app for students to try the trained Titanic survival model in a browser, with no install required. Built from the model trained in [`1 - Titanic Survival Prediction`](../1%20-%20Titanic%20Survival%20Prediction) (the `Titanic_Passenger_Survival_Recommended.ipynb` notebook).

## Deploying to Streamlit Community Cloud (free)

1. Go to **https://share.streamlit.io** and sign in with your GitHub account.
2. Click **"Create app"** (or **"New app"**).
3. Choose **"Deploy a public app from GitHub"**.
4. Fill in:
   - **Repository:** `haqnawaz99/ml-teaching-notebooks`
   - **Branch:** `main`
   - **Main file path:** `streamlit-app/app.py`
5. Click **Deploy**. The first deploy takes a minute or two while it installs `requirements.txt`.
6. You'll get a public URL like `https://<something>.streamlit.app` — share that with students. It stays live and free, and **automatically redeploys** whenever you push changes to this repo.

Once deployed, paste the URL into the main [`README.md`](../README.md) "Try It Online" section so students can find it from the repo homepage.

## Running It Locally First (optional, to test changes)

```bash
cd streamlit-app
python -m venv venv
venv\Scripts\activate   # Windows; use "source venv/bin/activate" on macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

This opens the app at `http://localhost:8501`.

## How It Works

- `app.py` loads `svc_trained_model.pkl` (the same file the notebook produces) and recreates the exact same `LabelEncoder`s used during training, so predictions match the notebook exactly.
- Students pick PClass, Gender, Sibling count, and Embarked port from dropdowns and click **Predict Survival**.
- If you retrain the model in the notebook and want the app to reflect it, copy the new `svc_trained_model.pkl` into this folder and push — Streamlit Cloud will auto-redeploy.
