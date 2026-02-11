"""
model_train.py
----------------------------------------
Crop Yield Prediction System
FINAL clean training script with:
- Categorical text normalization
- Safe label encoding
- Regression + Classification
- Progress logging
- Streamlit-compatible artifacts
"""

import pandas as pd
import joblib
import logging
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    accuracy_score, f1_score
)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# =====================================================
# LOGGING CONFIG
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger()

def step(title):
    log.info("=" * 70)
    log.info(title)
    log.info("=" * 70)


# =====================================================
# STEP 1: LOAD DATA
# =====================================================
step("STEP 1: Loading dataset")

df = pd.read_csv("./crop-yield-in-indian-states-dataset/crop_yield.csv")
df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True, errors="ignore")

log.info(f"Dataset shape: {df.shape}")


# =====================================================
# STEP 2: CLEAN CATEGORICAL TEXT (ROOT FIX)
# =====================================================
step("STEP 2: Cleaning categorical text")

for col in ["Crop", "Season", "State"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()      # removes padded spaces
        .str.title()      # normalizes casing
    )

log.info("Categorical columns cleaned (spaces + casing normalized)")


# =====================================================
# STEP 3: PRESERVE RAW VALUES
# =====================================================
df["Crop_raw"] = df["Crop"]
df["Season_raw"] = df["Season"]
df["State_raw"] = df["State"]


# =====================================================
# STEP 4: LABEL ENCODING (SAFE)
# =====================================================
step("STEP 4: Encoding categorical features")

le_crop = LabelEncoder()
le_season = LabelEncoder()
le_state = LabelEncoder()

df["Crop"] = le_crop.fit_transform(df["Crop_raw"])
df["Season"] = le_season.fit_transform(df["Season_raw"])
df["State"] = le_state.fit_transform(df["State_raw"])

# Sanity check (fail fast if dirty)
assert all(s == s.strip() for s in le_season.classes_), "Season labels contain spaces"
assert all(s == s.strip() for s in le_state.classes_), "State labels contain spaces"
assert all(s == s.strip() for s in le_crop.classes_), "Crop labels contain spaces"

joblib.dump(le_crop, "le_item.pkl")
joblib.dump(le_season, "le_season.pkl")
joblib.dump(le_state, "le_state.pkl")

log.info("Encoders saved successfully")


# =====================================================
# STEP 5: REGRESSION DATASET
# =====================================================
step("STEP 5: Preparing regression dataset")

FEATURES = [
    "Crop",
    "Crop_Year",
    "Season",
    "State",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]

X_reg = df[FEATURES]
y_reg = df["Yield"]

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler_reg = StandardScaler()
X_train_scaled = scaler_reg.fit_transform(X_train)
X_test_scaled = scaler_reg.transform(X_test)

joblib.dump(scaler_reg, "scaler_reg.pkl")


# =====================================================
# STEP 6: TRAIN REGRESSION MODELS
# =====================================================
step("STEP 6: Training regression models")

reg_models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression": Pipeline([
        ("poly", PolynomialFeatures(degree=2)),
        ("lr", LinearRegression())
    ]),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

reg_results = []

for name, model in reg_models.items():
    log.info(f"Training {name}...")
    start = time.time()

    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    reg_results.append({"Model": name, "R2": r2, "MAE": mae})

    log.info(
        f"{name} | R2={r2:.4f} | MAE={mae:.2f} | Time={time.time()-start:.1f}s"
    )

reg_df = pd.DataFrame(reg_results)
best_reg_name = reg_df.sort_values("R2", ascending=False).iloc[0]["Model"]
best_reg_model = reg_models[best_reg_name]

joblib.dump(best_reg_model, "best_reg_model.pkl")

log.info(f"Best regression model selected: {best_reg_name}")


# =====================================================
# STEP 7: CLASSIFICATION DATASET
# =====================================================
step("STEP 7: Preparing classification dataset")

q1 = df["Yield"].quantile(0.33)
q2 = df["Yield"].quantile(0.67)

df_clf = df[(df["Yield"] <= q1) | (df["Yield"] >= q2)].copy()
df_clf["yield_category"] = df_clf["Yield"].apply(
    lambda x: "Low" if x <= q1 else "High"
)

X_clf = df_clf[FEATURES]
y_clf = df_clf["yield_category"]

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf,
    test_size=0.2,
    random_state=42,
    stratify=y_clf
)


# =====================================================
# STEP 8: TRAIN CLASSIFICATION MODELS
# =====================================================
step("STEP 8: Training classification models")

clf_models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=7))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", C=2))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    ),
    "Naive Bayes": GaussianNB()
}

clf_results = []

for name, model in clf_models.items():
    log.info(f"Training {name}...")
    start = time.time()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, pos_label="High")

    clf_results.append({"Model": name, "Accuracy": acc, "F1": f1})

    log.info(
        f"{name} | Accuracy={acc:.4f} | F1={f1:.4f} | Time={time.time()-start:.1f}s"
    )

clf_df = pd.DataFrame(clf_results)
best_clf_name = clf_df.sort_values("F1", ascending=False).iloc[0]["Model"]
best_clf_model = clf_models[best_clf_name]

joblib.dump(best_clf_model, "best_classification_model.pkl")

log.info(f"Best classification model selected: {best_clf_name}")


# =====================================================
# DONE
# =====================================================
step("TRAINING COMPLETE")

log.info("All models, scalers, and encoders saved successfully.")
log.info("Artifacts ready for Streamlit + verification.")
