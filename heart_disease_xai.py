import os
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

os.makedirs('images', exist_ok=True)


# ── 1. Data Loading ───────────────────────────────────────────────────────────
datasets = pd.read_csv('heart.csv')
print(datasets.isnull().sum())


# ── 2. Cholesterol Imputation ─────────────────────────────────────────────────
datasets['Cholesterol'] = datasets['Cholesterol'].replace(0, np.nan)
datasets['Cholesterol'] = datasets['Cholesterol'].fillna(datasets['Cholesterol'].median())


# ── 3. EDA ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle("Feature Distributions", fontsize=18)
sns.histplot(x=datasets["Age"],            ax=axes[0, 0]).set_title("Age")
sns.countplot(data=datasets, x="Sex",      ax=axes[0, 1]).set_title("Sex")
sns.countplot(x=datasets["ChestPainType"], ax=axes[0, 2]).set_title("ChestPainType")
sns.histplot(x=datasets["RestingBP"],      ax=axes[1, 0]).set_title("RestingBP")
sns.histplot(x=datasets["Cholesterol"],    ax=axes[1, 1]).set_title("Cholesterol")
sns.countplot(x=datasets["FastingBS"],     ax=axes[1, 2]).set_title("FastingBS")
sns.countplot(x=datasets["RestingECG"],    ax=axes[2, 0]).set_title("RestingECG")
sns.histplot(x=datasets["MaxHR"],          ax=axes[2, 1]).set_title("MaxHR")
sns.countplot(x=datasets["ST_Slope"],      ax=axes[2, 2]).set_title("ST_Slope")
plt.tight_layout()
plt.savefig('images/eda_distributions.png', dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.countplot(data=datasets, x="ST_Slope", hue="HeartDisease", ax=axes[0]).set_title("ST_Slope vs HeartDisease")
sns.pointplot(data=datasets, x="HeartDisease", y="Cholesterol", ax=axes[1]).set_title("Cholesterol vs HeartDisease")
plt.tight_layout()
plt.savefig('images/eda_heartdisease_correlation.png', dpi=150)
plt.close()

sns.pairplot(datasets, hue="HeartDisease")
plt.savefig('images/eda_pairplot.png', dpi=100)
plt.close()


# ── 4. Preprocessing ──────────────────────────────────────────────────────────
datasets.ExerciseAngina = datasets.ExerciseAngina == "Y"
datasets.Sex = datasets.Sex == "M"
datasets.rename(columns={"Sex": "Male"}, inplace=True)
datasets['Male'] = datasets['Male'].astype(int)
datasets['ExerciseAngina'] = datasets['ExerciseAngina'].astype(int)
datasets = pd.get_dummies(datasets)

print(datasets.corr()["HeartDisease"].sort_values(ascending=False))

datasets = datasets.drop(columns=[
    "RestingBP", "RestingECG_ST", "Cholesterol",
    "RestingECG_LVH", "ChestPainType_TA", "RestingECG_Normal"
])
print(f"최종 특성 수: {datasets.shape[1]}")


# ── 5. Train/Test Split & Baseline Model ─────────────────────────────────────
X = datasets.drop(columns="HeartDisease")
y = datasets["HeartDisease"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(random_state=7, eval_metric='logloss', use_label_encoder=False)
model.fit(x_train, y_train)

accuracy = accuracy_score(y_test, [round(v) for v in model.predict(x_test)])
print(f"Test Accuracy (SMOTE 적용 전): {accuracy * 100:.2f}%")


# ── 6. SMOTETomek Oversampling ────────────────────────────────────────────────
smote = SMOTETomek(random_state=42)
X_train_over, y_train_over = smote.fit_resample(x_train, y_train)

print(f"SMOTE 적용 전 훈련 데이터: {x_train.shape}")
print(f"SMOTE 적용 후 훈련 데이터: {X_train_over.shape}")
print(f"SMOTE 전  - Class 1: {sum(y_train==1)}, Class 0: {sum(y_train==0)}")
print(f"SMOTE 후  - Class 1: {sum(y_train_over==1)}, Class 0: {sum(y_train_over==0)}")

model.fit(X_train_over, y_train_over)
accuracy = accuracy_score(y_test, [round(v) for v in model.predict(x_test)])
print(f"Test Accuracy (SMOTE 적용 후): {accuracy * 100:.2f}%")


# ── 7. K-Fold Cross Validation ────────────────────────────────────────────────
def run_kfold(features, labels, title, base_model, resampler_cls=None):
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracy = []

    for fold, (train_idx, val_idx) in enumerate(skfold.split(features, labels), 1):
        X_tr, X_val = features.iloc[train_idx], features.iloc[val_idx]
        y_tr, y_val = labels.iloc[train_idx], labels.iloc[val_idx]

        if resampler_cls is not None:
            X_tr, y_tr = resampler_cls(random_state=42).fit_resample(X_tr, y_tr)

        fold_model = clone(base_model)
        fold_model.fit(X_tr, y_tr)

        acc = round(accuracy_score(y_val, fold_model.predict(X_val)), 4)
        print(f"  Fold {fold}: {acc:.4f}")
        cv_accuracy.append(acc)

    mean_acc = np.mean(cv_accuracy)
    std_acc  = np.std(cv_accuracy)
    print(f"  Mean: {mean_acc:.4f}")
    print(f"  Std : {std_acc:.4f}")

    values = [v * 100 for v in cv_accuracy] + [mean_acc * 100]
    colors = ["dodgerblue"] * 5 + ["C2"]

    plt.figure(figsize=(8, 5))
    plt.bar(range(6), values, color=colors)
    plt.xticks(range(6), ["1", "2", "3", "4", "5", "mean"])
    plt.ylabel("Accuracy (%)")
    plt.ylim(70, 100)
    plt.title(f"{title}\nMean: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    plt.tight_layout()

    return mean_acc, std_acc


print("\n[SMOTE: Before]")
before_smote_acc, before_smote_std = run_kfold(
    x_train, y_train, "K-Fold CV (Before SMOTE)", model
)
plt.savefig('images/kfold_before_smote.png', dpi=150)
plt.close()

print("\n[SMOTE: After]")
after_smote_acc, after_smote_std = run_kfold(
    x_train, y_train, "K-Fold CV (After SMOTE)", model, resampler_cls=SMOTE
)
plt.savefig('images/kfold_after_smote.png', dpi=150)
plt.close()

print(f"\n정확도 변화 (SMOTE): {before_smote_acc*100:.2f}% ± {before_smote_std*100:.2f}%")
print(f"                 → {after_smote_acc*100:.2f}% ± {after_smote_std*100:.2f}%")

print("\n[SMOTETomek: Before]")
before_acc, before_std = run_kfold(
    x_train, y_train, "K-Fold CV (Before SMOTETomek)", model
)
plt.savefig('images/kfold_before_smotetomek.png', dpi=150)
plt.close()

print("\n[SMOTETomek: After]")
after_acc, after_std = run_kfold(
    x_train, y_train, "K-Fold CV (After SMOTETomek)", model, resampler_cls=SMOTETomek
)
plt.savefig('images/kfold_after_smotetomek.png', dpi=150)
plt.close()

print(f"\n정확도 변화 (SMOTETomek): {before_acc*100:.2f}% ± {before_std*100:.2f}%")
print(f"                      → {after_acc*100:.2f}% ± {after_std*100:.2f}%")


# ── 8. Final Model (SMOTETomek) ───────────────────────────────────────────────
sm_final = SMOTETomek(random_state=42)
X_train_final, y_train_final = sm_final.fit_resample(x_train, y_train)
model.fit(X_train_final, y_train_final)


# ── 9. SHAP Analysis ──────────────────────────────────────────────────────────
x_test_r = x_test.reset_index(drop=True)
y_test_r = y_test.reset_index(drop=True)

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_test_r)
print(f"\nSHAP values shape: {shap_values.shape}")

y_pred_test = model.predict(x_test_r)
Person1 = int(np.where(y_pred_test == 1)[0][0])
Person2 = int(np.where(y_pred_test == 0)[0][0])

print(f"\nPerson1: index={Person1}, 실제={y_test_r[Person1]}, 예측={y_pred_test[Person1]}")
print(x_test_r.iloc[Person1])
print(f"\nPerson2: index={Person2}, 실제={y_test_r[Person2]}, 예측={y_pred_test[Person2]}")
print(x_test_r.iloc[Person2])

for idx, fname in [
    (Person1, 'images/shap_force_person1.png'),
    (Person2, 'images/shap_force_person2.png'),
]:
    shap.force_plot(
        explainer.expected_value,
        shap_values[idx, :],
        x_test_r.iloc[idx, :],
        matplotlib=True, show=False
    )
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()

shap.summary_plot(shap_values, x_test_r, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('images/shap_summary_bar.png', dpi=150)
plt.close()

shap.summary_plot(shap_values, x_test_r, show=False)
plt.tight_layout()
plt.savefig('images/shap_summary_beeswarm.png', dpi=150)
plt.close()

for feature in ['ST_Slope_Flat', 'ChestPainType_ASY', 'Oldpeak', 'ExerciseAngina', 'ST_Slope_Up']:
    shap.dependence_plot(feature, shap_values, x_test_r, interaction_index=None, show=False)
    plt.tight_layout()
    plt.savefig(f'images/shap_dependence_{feature}.png', dpi=150)
    plt.close()

print("\nDone. All figures saved to images/")
