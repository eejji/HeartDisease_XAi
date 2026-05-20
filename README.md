# 🫀 심장질환 예측 및 XAI 해석 (XGBoost + SHAP)

> XGBoost 모델로 심장질환을 예측하고, SHAP을 활용한 설명 가능한 AI(XAI)로 예측 근거를 시각화한 프로젝트입니다.   
> 기존 v1 버전과 v2 버전이 존재합니다.   
> v1(Heart_v1.ipynb) 버전을 재분석한 결과 (재현성, Data lakege)의 문제가 발견되어 수정하여 v2(Heart_v2.ipynb)가 탄생하였습니다.
---

## 📌 프로젝트 개요

1. 심혈관 질환은 자살로 이어질 수 있어 조기에 예방하는 것이 중요합니다.   
2. 머신러닝 모델은 높은 성능을 보이지만 예측 근거를 설명하기 어렵다는 한계가 있습니다.  
- 이 프로젝트는 **XGBoost** 모델을 통해 심장질환을 예측하고
- **SHAP(SHapley Additive exPlanations)** 을 통해 모델이 각 환자에 대해 왜 그런 예측을 했는지를 feature 수준에서 해석합니다.

### 핵심 질문
- 어떤 특성(feature)이 심장질환 예측에 가장 큰 영향을 미치는가?
- 개별 환자에 대해 예측 근거를 어떻게 설명할 수 있는가?

### 팀원 & 프로젝트 기간
- 기간 : 2023.01
- 총 4명
- 기여도 : 80 %

---

## 📊 데이터셋

| 항목 | 내용 |
|---|---|
| 출처 | [Kaggle - Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) |
| 샘플 수 | 918명 |
| 특성 수 | 12개 (전처리 후 사용) |
| 타겟 | `HeartDisease` (0: 정상, 1: 심장질환) |

### 주요 특성

| 특성 | 설명 |
|---|---|
| Age | 나이 |
| Sex | 성별 |
| ChestPainType | 흉통 유형 (TA / ATA / NAP / ASY) |
| Cholesterol | 콜레스테롤 수치 |
| MaxHR | 최대 심박수 |
| Oldpeak | 운동 유발 ST 강하 |
| ST_Slope | ST 분절 기울기 |
| ExerciseAngina | 운동 시 협심증 여부 |

---

## 🔧 방법론

```
데이터 로드 → EDA → 전처리 → 훈련/테스트 분리
       ↓
  SMOTETomek 데이터 증강 (클래스 불균형 해소)
       ↓
  XGBoost 분류 모델 학습
       ↓
  K-Fold 교차검증 (5-Fold)
       ↓
  SHAP (TreeExplainer) → 테스트 데이터 기준 해석
```

### SMOTETomek 적용 이유

원본 데이터에 클래스 불균형이 존재해 소수 클래스(정상)에 대한 예측 성능이 낮았습니다.  
SMOTETomek로 훈련 데이터를 증강하여 클래스 비율을 균등하게 맞췄습니다.

v1 문제 확인:
SMOTE 적용 후 정확도가 향상되지 않아 원인을 분석했습니다.   
SMOTE는 소수 클래스를 증강하지만 경계면 근처 노이즈 샘플이 함께 생성되어 결정 경계를 모호하게 만드는 한계가 있습니다.

v2 개선:
SMOTETomek을 적용해 오버샘플링과 동시에 Tomek Links 기반 노이즈 제거를 수행했습니다.   
실험 결과 정확도가 향상되어 증강 방식 변경의 효과를 확인했습니다.   


---

## 📈 실험 결과

| 구분 | 평균 정확도 (5-Fold) |
|---|---|
| SMOTE 적용 | 83.92%% |
| SMOTETomek 적용 | 84.88% |

| 구분 | 평균 정확도 (5-Fold) |
|---|---|
| SMOTETomek 적용 전 | ~83.92 ± 0.028% |
| SMOTETomek 적용 후 | ~84.88 ± 0.021% | (+0.96)


---

## 🔍 SHAP 분석 주요 결과

### Feature 중요도 (전체)

모델 예측에 영향을 미치는 상위 특성:

1. **ST_Slope_Flat** — ST 분절이 평탄할수록 심장질환 위험 증가
2. **ChestPainType_ASY** — 무증상 흉통이 가장 강한 양성 예측 인자
3. **ExerciseAngina** — 운동 시 협심증 발생이 위험 신호
4. **ST_Slope_Up** — ST 분절 상향이 심장질환 위험을 낮춤
5. **Oldpeak** — ST 강하 수치가 높을수록 위험

### 개별 환자 해석 (Force Plot)

- **Person1 (심장질환 예측)**: `ST_Slope_Flat`, `ChestPainType_ASY` 특성이 예측값을 기준값 이상으로 크게 끌어올림
- **Person2 (정상 예측)**: `ST_Slope_Up`, 낮은 `Oldpeak`이 예측값을 기준값 아래로 끌어내림

---

## 🗂️ 프로젝트 구조

```
heart-disease-xai/
├── Heart_v2.ipynb                  # 실험 노트북 (EDA, 분석, 결과 포함)
├── heart_disease_xai.py            # 전체 파이프라인 실행 스크립트
├── requirements.txt
├── .gitignore
├── README.md
├── heart.csv                       # 데이터셋 (별도 다운로드 필요 ↓)
└── images/                         # 실행 시 자동 생성
    ├── eda_distributions.png
    ├── eda_heartdisease_correlation.png
    ├── eda_pairplot.png
    ├── kfold_before_smote.png
    ├── kfold_after_smote.png
    ├── kfold_before_smotetomek.png
    ├── kfold_after_smotetomek.png
    ├── shap_force_person1.png
    ├── shap_force_person2.png
    ├── shap_summary_bar.png
    ├── shap_summary_beeswarm.png
    └── shap_dependence_*.png
```

---

## ⚙️ 설치 및 실행

### 1. 저장소 클론

```bash
git clone https://github.com/<your-username>/heart-disease-xai.git
cd heart-disease-xai
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 데이터셋 다운로드

[Kaggle - Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) 에서 `heart.csv`를 다운로드한 후 프로젝트 루트에 위치시킵니다.

```
heart-disease-xai/
├── heart.csv   ← 여기에 위치
└── ...
```

### 4. 실행

```bash
python heart_disease_xai.py
```

실행 후 분석 결과 이미지가 `images/` 폴더에 자동으로 저장됩니다.

> 노트북으로 단계별 결과를 확인하려면 `Heart_v2.ipynb`를 Jupyter에서 실행하세요.

---

## 🛠️ 사용 기술

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.41+-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-red)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.10+-purple)

---

## 📝 References

- Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions.* NeurIPS.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* KDD.
- Chawla, N. V. et al. (2002). *SMOTE: Synthetic minority over-sampling technique.* JAIR.
