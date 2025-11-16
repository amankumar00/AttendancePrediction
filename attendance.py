!pip install -q joblib scikit-learn pandas

import os, time, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
from google.colab import files
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier

OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
VAL_SIZE = 0.20
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}
JOBLIB_COMPRESS = 9

uploaded = files.upload()
csv_filename = list(uploaded.keys())[0]
df = pd.read_csv(csv_filename)

numeric_cols = ['past_meetings', 'past_attended', 'attendance_rate', 'importance']
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

if 'past_meetings' in df.columns:
    df = df[df['past_meetings'] >= 0].reset_index(drop=True)

if 'company' in df.columns and 'attended' in df.columns:
    comp_stats = df.groupby('company')['attended'].agg(['mean','count'])
    comp_stats = comp_stats.rename(columns={'mean':'company_att_mean','count':'company_count'})
    df = df.merge(comp_stats, on='company', how='left')

if 'role' in df.columns and 'attended' in df.columns:
    role_stats = df.groupby('role')['attended'].agg(['mean','count'])
    role_stats = role_stats.rename(columns={'mean':'role_att_mean','count':'role_count'})
    df = df.merge(role_stats, on='role', how='left')

if {'attendance_rate','company_att_mean'}.issubset(df.columns):
    df['attendance_vs_company'] = df['attendance_rate'] - df['company_att_mean']

if 'importance' in df.columns:
    df['is_high_importance'] = (df['importance'] >= df['importance'].quantile(0.75)).astype(int)

TARGET = 'attended'
candidate_features = [
    'company','role','meeting_type','time_of_day',
    'past_meetings','past_attended','attendance_rate','importance',
    'company_att_mean','company_count','role_att_mean','role_count',
    'attendance_vs_company','is_high_importance'
]

features = [c for c in candidate_features if c in df.columns]
X = df[features].copy()
y = df[TARGET].copy()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VAL_SIZE, stratify=y, random_state=RANDOM_STATE
)

class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, n_splits=5, shuffle=True, random_state=42):
        self.cols = cols
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        X = X.copy().reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        self.cols_ = self.cols or [
            c for c in X.columns 
            if X[c].dtype == 'object' or str(X[c].dtype).startswith('category')
        ]
        
        self.global_mean_ = float(y.mean())
        self.mappings_ = {}
        self._oof_ = pd.DataFrame(index=X.index)

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )

        for col in self.cols_:
            oof_col = pd.Series(index=X.index, dtype=float)
            for tr, vl in kf.split(X):
                fold_means = y.iloc[tr].groupby(X.iloc[tr][col]).mean()
                oof_col.iloc[vl] = X.iloc[vl][col].map(fold_means)
            oof_col.fillna(self.global_mean_, inplace=True)
            self._oof_[col + '_te'] = oof_col
            self.mappings_[col] = y.groupby(X[col]).mean()
        return self

    def transform(self, X):
        X = X.copy().reset_index(drop=True)
        for col in self.cols_:
            mapping = self.mappings_.get(col, pd.Series(dtype=float))
            X[col + '_te'] = X[col].map(mapping).fillna(self.global_mean_)
        return X.drop(columns=self.cols_, errors='ignore')

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        X_oof = X.copy().reset_index(drop=True)
        for col in self.cols_:
            X_oof[col + '_te'] = self._oof_[col + '_te']
        return X_oof.drop(columns=self.cols_, errors='ignore')

cat_cols = [c for c in ['company','role','meeting_type','time_of_day'] if c in X_train.columns]
te = KFoldTargetEncoder(cols=cat_cols, n_splits=5, random_state=RANDOM_STATE)

X_train_enc = te.fit_transform(X_train, y_train)
X_val_enc = te.transform(X_val)

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_train_num = num_pipeline.fit_transform(X_train_enc)
X_val_num = num_pipeline.transform(X_val_enc)

X_train_df = pd.DataFrame(X_train_num, columns=X_train_enc.columns, index=X_train_enc.index)
X_val_df = pd.DataFrame(X_val_num, columns=X_val_enc.columns, index=X_val_enc.index)

model = RandomForestClassifier(**RF_PARAMS)
model.fit(X_train_df, y_train)

probs = model.predict_proba(X_val_df)[:, 1]
preds = (probs >= 0.5).astype(int)

acc = accuracy_score(y_val, preds)
auc = roc_auc_score(y_val, probs)

artifact = {
    "model": model,
    "target_encoder": te,
    "numeric_pipeline": num_pipeline,
    "features": features,
    "categorical_cols": cat_cols,
    "trained_at_utc": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    "train_rows": int(len(df))
}

artifact_path = os.path.join(OUT_DIR, "randomforest_meeting_model.joblib")
joblib.dump(artifact, artifact_path, compress=JOBLIB_COMPRESS)

meta = {
    "model_type": "RandomForestClassifier",
    "accuracy": float(acc),
    "roc_auc": float(auc),
    "train_rows": int(len(df)),
    "features": features,
    "saved_at": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
}

with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
