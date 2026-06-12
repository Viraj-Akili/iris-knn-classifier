# ============================================================
# PROJECT 2: Data Classification Using AI
# DecodeLabs Industrial Training Kit | Batch 2026
# Algorithm: K-Nearest Neighbors (KNN) on Iris Dataset
# ============================================================

# Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

print("=" * 55)
print("  PROJECT 2: Data Classification Using AI")
print("  KNN Classifier | Iris Benchmark Dataset")
print("=" * 55)


# Load and inspect dataset
print("\n[1] Loading the Iris Dataset...")
iris = load_iris()

# Tabular view for quick exploration and summary stats
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print(f"\n  Samples   : {df.shape[0]}")
print(f"  Features  : {df.shape[1] - 1}")
print(f"  Classes   : {list(iris.target_names)}")
print(f"\n  Class distribution:\n{df['species'].value_counts().to_string()}")
print(f"\n  First 5 rows:\n{df.head().to_string()}")
print(f"\n  Basic stats:\n{df.describe().round(2).to_string()}")


# Standardize features
print("\n[2] Applying Feature Scaling (StandardScaler)...")
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("  Raw data range  :", X.min().round(1), "to", X.max().round(1))
print("  Scaled mean     :", X_scaled.mean().round(4), "(should be ~0)")
print("  Scaled variance :", X_scaled.var().round(4), "(should be ~1)")


# Split into train and test sets
print("\n[3] Splitting data: 80% Train / 20% Test...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)
print(f"  Training samples : {X_train.shape[0]}")
print(f"  Testing samples  : {X_test.shape[0]}")


# Select K using the elbow method
print("\n[4] Finding Optimal K using the Elbow Method...")
k_range = range(1, 21)
error_rates = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    error_rates.append(1 - accuracy_score(y_test, preds))

optimal_k = k_range[error_rates.index(min(error_rates))]
print(f"  Optimal K found  : {optimal_k}")


# Train classifier
print(f"\n[5] Training KNN model with K={optimal_k}...")
model = KNeighborsClassifier(n_neighbors=optimal_k)
model.fit(X_train, y_train)
print("  Model trained successfully!")


# Evaluate on holdout set
print("\n[6] Evaluating model on test set...")
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="weighted")

print(f"\n  Accuracy  : {accuracy * 100:.2f}%")
print(f"  F1 Score  : {f1:.4f}")
print(f"\n  Classification Report:\n")
print(classification_report(y_test, predictions, target_names=iris.target_names))


# Generate visualizations
print("[7] Generating visualisation plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Project 2: KNN Iris Classification | DecodeLabs 2026",
             fontsize=14, fontweight="bold", y=0.98)

# Plot 1: Elbow curve
ax1 = axes[0, 0]
ax1.plot(k_range, error_rates, marker="o", color="#1a3a5c", linewidth=2)
ax1.axvline(x=optimal_k, color="#e05c2a", linestyle="--",
            label=f"Optimal K={optimal_k}")
ax1.scatter([optimal_k], [error_rates[optimal_k - 1]],
            color="#e05c2a", s=120, zorder=5)
ax1.set_title("Elbow Curve: Finding Optimal K", fontweight="bold")
ax1.set_xlabel("K Value")
ax1.set_ylabel("Error Rate")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Confusion matrix
ax2 = axes[0, 1]
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names, ax=ax2,
            linewidths=0.5, cbar=False)
ax2.set_title("Confusion Matrix", fontweight="bold")
ax2.set_xlabel("Predicted Label")
ax2.set_ylabel("True Label")

# Plot 3: Feature distribution
ax3 = axes[1, 0]
colors = ["#1a3a5c", "#e05c2a", "#2a7a4c"]
for i, (species, color) in enumerate(zip(iris.target_names, colors)):
    mask = y == i
    ax3.scatter(X[mask, 2], X[mask, 3],
                label=species, color=color, alpha=0.7, s=50)
ax3.set_title("Feature Space: Petal Length vs Width", fontweight="bold")
ax3.set_xlabel("Petal Length (cm)")
ax3.set_ylabel("Petal Width (cm)")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Performance summary
ax4 = axes[1, 1]
ax4.axis("off")
metrics_text = (
    f"  MODEL PERFORMANCE SUMMARY\n"
    f"  {'─' * 30}\n\n"
    f"  Algorithm     :  KNN (K={optimal_k})\n"
    f"  Dataset       :  Iris (150 samples)\n"
    f"  Train / Test  :  80% / 20%\n"
    f"  Scaler        :  StandardScaler\n\n"
    f"  Accuracy      :  {accuracy * 100:.2f}%\n"
    f"  F1 Score      :  {f1:.4f}\n\n"
    f"  Classes       :  Setosa | Versicolor | Virginica\n"
    f"  Test Samples  :  {len(y_test)}\n"
    f"  Correct       :  {int(accuracy * len(y_test))} / {len(y_test)}"
)
ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes,
         fontsize=11, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f4f8",
                   edgecolor="#1a3a5c", linewidth=1.5))

plt.tight_layout()
screenshots_dir = Path("screenshots")
screenshots_dir.mkdir(exist_ok=True)

plt.savefig(
    screenshots_dir / "project2_results.png",
    dpi=150,
    bbox_inches="tight"
)

print("Plot saved → screenshots/project2_results.png")

plt.show()
print("  Plot saved → screenshots/project2_results.png")


# Predict on unseen samples
print("\n[8] Testing on brand-new unseen flower measurements...")
new_flowers = np.array([
    [5.1, 3.5, 1.4, 0.2],   # Likely Setosa
    [6.0, 2.9, 4.5, 1.5],   # Likely Versicolor
    [6.7, 3.1, 5.6, 2.4],   # Likely Virginica
])
new_scaled = scaler.transform(new_flowers)
new_predictions = model.predict(new_scaled)
new_proba = model.predict_proba(new_scaled)

print(f"\n  {'Sepal_L':>8} {'Sepal_W':>8} {'Petal_L':>8} {'Petal_W':>8}  →  Prediction")
print("  " + "-" * 60)
for flower, pred in zip(new_flowers, new_predictions):
    label = iris.target_names[pred]
    print(f"  {flower[0]:>8} {flower[1]:>8} {flower[2]:>8} {flower[3]:>8}  →  {label.upper()}")

print("\n" + "=" * 55)
