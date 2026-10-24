# Mental Health Predictor (API-driven)

Files:
- dataset_api_loader.py    # fetch dataset from Kaggle / Reddit / Twitter or fallback
- train_models_api.py      # train TF-IDF+ANN and Tokenizer+LSTM
- app_mental_api.py        # Flask app for predictions
- templates/index_mental.html
- requirements.txt

Quick start:
1. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate   # Windows: venv\\Scripts\\activate

2. Install dependencies:
   pip install -r requirements.txt

3. (Optional) Set API credentials as environment variables:
   - Kaggle: place kaggle.json in project root or ~/.kaggle/kaggle.json
   - Reddit:
       export REDDIT_CLIENT_ID="..."
       export REDDIT_CLIENT_SECRET="..."
       export REDDIT_USER_AGENT="myapp by /u/you"
   - Twitter/X:
       export TWITTER_BEARER_TOKEN="..."

4. Fetch dataset (will merge Kaggle/Reddit/Twitter or fall back to synthetic):
   python generate_synthetic_data.py
   -> produces mental_dataset.csv

5. Train models:
   python train_models_api.py
   -> saves tfidf_vectorizer.joblib, ann_model.h5, tokenizer.joblib, lstm_model.h5

6. Run the app:
   python app_mental_api.py
   Open http://127.0.0.1:5000/

Notes:
- If you lack API keys, dataset_api_loader.py will generate a synthetic dataset so you can run and test the full pipeline.
- Replace `KAGGLE_DATASET_SLUG` in dataset_api_loader.py with a real slug if you want a particular Kaggle dataset.
