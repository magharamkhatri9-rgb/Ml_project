"""
=============================================================
  ML Lab – Midterm Project
  Domain  : House Prices / Real Estate (Pakistan)
  Dataset : house_prices.csv  (300 records, 13 columns)
  Tasks   : Regression  → predict price_pkr_lakhs
            Classification → predict price_category (Low/Med/High)
  Algorithms (Regression)      : Linear Regression, Random Forest Regressor
  Algorithms (Classification)  : Logistic Regression, Random Forest Classifier
=============================================================
"""

# ─── 0. IMPORTS ──────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

# ─── 1. LOAD DATASET ─────────────────────────────────────────────────────────
df = pd.read_csv("house_prices.csv")
print("=" * 60)
print("SECTION 1 – DATASET OVERVIEW")
print("=" * 60)
print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns\n")
print(df.head())

# ─── 2. BASIC EDA ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2 – EDA")
print("=" * 60)
print("\n[2.1] Data Types & Non-Null Counts:")
print(df.info())
print("\n[2.2] Statistical Summary:")
print(df.describe().round(2))
print("\n[2.3] Missing Values:")
print(df.isnull().sum())
print("\n[2.4] Price Category Distribution:")
print(df["price_category"].value_counts())

# ─── 3. VISUALIZATIONS ───────────────────────────────────────────────────────
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("House Prices – Exploratory Data Analysis", fontsize=16, fontweight='bold')

# 3.1 Price Distribution
axes[0, 0].hist(df["price_pkr_lakhs"], bins=25, color="#2196F3", edgecolor="white")
axes[0, 0].set_title("Price Distribution (Lakhs PKR)")
axes[0, 0].set_xlabel("Price")
axes[0, 0].set_ylabel("Frequency")

# 3.2 Category Counts
cat_counts = df["price_category"].value_counts()
axes[0, 1].bar(cat_counts.index, cat_counts.values, color=["#4CAF50", "#FF9800", "#F44336"])
axes[0, 1].set_title("Price Category Counts")
axes[0, 1].set_xlabel("Category")
axes[0, 1].set_ylabel("Count")

# 3.3 Area vs Price
axes[0, 2].scatter(df["area_sqft"], df["price_pkr_lakhs"],
                   c=df["price_category"].map({"Low": "#4CAF50", "Medium": "#FF9800", "High": "#F44336"}),
                   alpha=0.6, edgecolors="white", linewidths=0.4)
axes[0, 2].set_title("Area vs Price (colored by category)")
axes[0, 2].set_xlabel("Area (sqft)")
axes[0, 2].set_ylabel("Price (Lakhs PKR)")

# 3.4 Correlation Heatmap
numeric_df = df.select_dtypes(include=np.number).drop("house_id", axis=1)
corr = numeric_df.corr()
sns.heatmap(corr, ax=axes[1, 0], annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, annot_kws={"size": 7})
axes[1, 0].set_title("Correlation Heatmap")

# 3.5 Bedrooms vs Avg Price
bed_price = df.groupby("bedrooms")["price_pkr_lakhs"].mean()
axes[1, 1].bar(bed_price.index, bed_price.values, color="#9C27B0")
axes[1, 1].set_title("Avg Price by Bedrooms")
axes[1, 1].set_xlabel("Bedrooms")
axes[1, 1].set_ylabel("Avg Price (Lakhs PKR)")

# 3.6 Location Score vs Price
axes[1, 2].scatter(df["location_score"], df["price_pkr_lakhs"],
                   color="#00BCD4", alpha=0.6, edgecolors="white", linewidths=0.4)
axes[1, 2].set_title("Location Score vs Price")
axes[1, 2].set_xlabel("Location Score")
axes[1, 2].set_ylabel("Price (Lakhs PKR)")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n[Saved] eda_plots.png")

# ─── 4. PREPROCESSING ────────────────────────────────────────────────────────
features = ["area_sqft", "bedrooms", "bathrooms", "floors", "garage",
            "garden", "location_score", "age_years",
            "distance_to_city_km", "renovated"]

X = df[features]
y_reg = df["price_pkr_lakhs"]
le = LabelEncoder()
y_cls = le.fit_transform(df["price_category"])   # Low=0, Medium=1, High=2

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cls, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_r_sc = scaler.fit_transform(X_train_r)
X_test_r_sc  = scaler.transform(X_test_r)
X_train_c_sc = scaler.fit_transform(X_train_c)
X_test_c_sc  = scaler.transform(X_test_c)

# ─── 5. REGRESSION ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3 – REGRESSION")
print("=" * 60)

# 5.1 Linear Regression
lr = LinearRegression()
lr.fit(X_train_r_sc, y_train_r)
lr_pred = lr.predict(X_test_r_sc)

lr_mae  = mean_absolute_error(y_test_r, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test_r, lr_pred))
lr_r2   = r2_score(y_test_r, lr_pred)

print(f"\n[A] Linear Regression")
print(f"    MAE  : {lr_mae:.2f} Lakhs PKR")
print(f"    RMSE : {lr_rmse:.2f} Lakhs PKR")
print(f"    R²   : {lr_r2:.4f}")

# 5.2 Random Forest Regressor
rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(X_train_r, y_train_r)
rfr_pred = rfr.predict(X_test_r)

rfr_mae  = mean_absolute_error(y_test_r, rfr_pred)
rfr_rmse = np.sqrt(mean_squared_error(y_test_r, rfr_pred))
rfr_r2   = r2_score(y_test_r, rfr_pred)

print(f"\n[B] Random Forest Regressor")
print(f"    MAE  : {rfr_mae:.2f} Lakhs PKR")
print(f"    RMSE : {rfr_rmse:.2f} Lakhs PKR")
print(f"    R²   : {rfr_r2:.4f}")

# Regression Comparison Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Regression – Actual vs Predicted", fontsize=14, fontweight='bold')
for ax, preds, label, color in zip(
        axes,
        [lr_pred, rfr_pred],
        ["Linear Regression", "Random Forest Regressor"],
        ["#2196F3", "#4CAF50"]):
    ax.scatter(y_test_r, preds, alpha=0.6, color=color, edgecolors="white", linewidths=0.4)
    ax.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--', lw=2)
    ax.set_title(label)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.text(0.05, 0.92, f"R² = {r2_score(y_test_r, preds):.4f}", transform=ax.transAxes,
            fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
plt.tight_layout()
plt.savefig("regression_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("[Saved] regression_plots.png")

# ─── 6. CLASSIFICATION ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 4 – CLASSIFICATION")
print("=" * 60)

class_names = le.classes_   # ['High', 'Low', 'Medium']

# 6.1 Logistic Regression
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_c_sc, y_train_c)
logreg_pred = logreg.predict(X_test_c_sc)
logreg_acc  = accuracy_score(y_test_c, logreg_pred)

print(f"\n[A] Logistic Regression")
print(f"    Accuracy : {logreg_acc*100:.2f}%")
print(classification_report(y_test_c, logreg_pred, target_names=class_names))

# 6.2 Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train_c, y_train_c)
rfc_pred = rfc.predict(X_test_c)
rfc_acc  = accuracy_score(y_test_c, rfc_pred)

print(f"\n[B] Random Forest Classifier")
print(f"    Accuracy : {rfc_acc*100:.2f}%")
print(classification_report(y_test_c, rfc_pred, target_names=class_names))

# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Classification – Confusion Matrices", fontsize=14, fontweight='bold')
for ax, preds, label in zip(
        axes,
        [logreg_pred, rfc_pred],
        ["Logistic Regression", "Random Forest Classifier"]):
    cm = confusion_matrix(y_test_c, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(label)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
print("[Saved] confusion_matrices.png")

# ─── 7. RESULTS COMPARISON ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 5 – RESULTS COMPARISON & ANALYSIS")
print("=" * 60)

print("\n── REGRESSION ──────────────────────────────────────")
print(f"{'Metric':<10} {'Linear Reg':>15} {'Random Forest':>15}")
print("-" * 42)
print(f"{'MAE':<10} {lr_mae:>15.2f} {rfr_mae:>15.2f}")
print(f"{'RMSE':<10} {lr_rmse:>15.2f} {rfr_rmse:>15.2f}")
print(f"{'R²':<10} {lr_r2:>15.4f} {rfr_r2:>15.4f}")

print("\n── CLASSIFICATION ──────────────────────────────────")
print(f"{'Metric':<10} {'Logistic Reg':>15} {'Random Forest':>15}")
print("-" * 42)
print(f"{'Accuracy':<10} {logreg_acc*100:>14.2f}% {rfc_acc*100:>14.2f}%")

# Bar chart comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Algorithm Performance Comparison", fontsize=14, fontweight='bold')

# Regression bars
metrics_reg = ["MAE", "RMSE", "R²"]
lr_scores   = [lr_mae, lr_rmse, lr_r2]
rfr_scores  = [rfr_mae, rfr_rmse, rfr_r2]
x = np.arange(len(metrics_reg))
w = 0.35
axes[0].bar(x - w/2, lr_scores,  w, label="Linear Regression", color="#2196F3")
axes[0].bar(x + w/2, rfr_scores, w, label="Random Forest",     color="#4CAF50")
axes[0].set_xticks(x); axes[0].set_xticklabels(metrics_reg)
axes[0].set_title("Regression Metrics")
axes[0].legend()
for bar in axes[0].patches:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

# Classification bars
models_cls = ["Logistic\nRegression", "Random\nForest"]
accs = [logreg_acc * 100, rfc_acc * 100]
bars = axes[1].bar(models_cls, accs, color=["#FF9800", "#9C27B0"], width=0.4)
axes[1].set_ylim(0, 110)
axes[1].set_title("Classification Accuracy (%)")
axes[1].set_ylabel("Accuracy (%)")
for bar, acc in zip(bars, accs):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{acc:.1f}%", ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("comparison_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("[Saved] comparison_chart.png")

# Feature Importance (RF)
fig, ax = plt.subplots(figsize=(9, 5))
importances = pd.Series(rfr.feature_importances_, index=features).sort_values()
importances.plot(kind="barh", ax=ax, color="#2196F3")
ax.set_title("Feature Importances – Random Forest Regressor", fontweight='bold')
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("[Saved] feature_importance.png")

# ─── 8. PERFORMANCE EXPLANATION ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 6 – PERFORMANCE EXPLANATION")
print("=" * 60)
print("""
REGRESSION:
  • Linear Regression assumes a straight-line relationship between
    features and price. It performs reasonably but misses complex
    non-linear interactions (e.g., a premium location + large area
    producing a disproportionately high price).
  • Random Forest Regressor builds multiple decision trees and averages
    their predictions. It captures non-linear patterns and feature
    interactions automatically, yielding a higher R² and lower error.

CLASSIFICATION:
  • Logistic Regression is a linear boundary classifier. It works well
    when classes are roughly linearly separable but struggles with
    overlapping price categories (Medium ↔ High boundary).
  • Random Forest Classifier again outperforms by learning complex
    decision boundaries through ensemble voting, giving higher accuracy
    especially on the boundary classes.

KEY INSIGHT:
  Random Forest wins in both tasks because house pricing is driven by
  non-linear interactions among features (area × location × age).
  Linear models cannot capture these without manual feature engineering.
""")

print("=" * 60)
print("PROJECT COMPLETE – All plots saved.")
print("=" * 60)
