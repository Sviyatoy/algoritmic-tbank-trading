import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# Загрузка данных
df = pd.read_csv("sber_h4_features.csv", index_col=0, parse_dates=True)
print(f"Загружено {len(df)} строк, {len(df.columns)} колонок")

# Определяем признаки (X) и цель (y) для горизонта 4 свечей
target_col = 'target_up_4'

exclude_cols = [
    'target_up_1', 'target_up_2', 'target_up_4',
    'target_down_1', 'target_down_2', 'target_down_4',
    'future_return_1', 'future_return_2', 'future_return_4',
    'open', 'high', 'low', 'close', 'volume'
]
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols].ffill().fillna(0)
y = df[target_col].fillna(0).astype(int)

# Временная кросс-валидация
tscv = TimeSeriesSplit(n_splits=5)

print(f"Признаков: {len(feature_cols)}")
print(f"Положительных классов: {y.sum()} ({y.mean() * 100:.1f}%)")

# Параметры LightGBM
params_base = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'class_weight': 'balanced', # ДОБАВЛЕНО ПОДУМАТЬ
    'num_leaves': 31,
    'max_depth': 5,
    'learning_rate': 0.05,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'min_child_samples': 20,
    'random_state': 42,
    'verbose': -1,
}

models = []
scores = []
thresholds = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Балансировка весов для текущего фолда
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    params = params_base.copy()
    params['scale_pos_weight'] = scale_pos

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='auc',
              callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])

    y_proba = model.predict_proba(X_val)[:, 1]

    # Оптимальный порог по индексу Юдена
    fpr, tpr, thr = roc_curve(y_val, y_proba)
    youden = tpr - fpr
    best_thr = thr[np.argmax(youden)] if len(thr) > 0 else 0.5
    thresholds.append(best_thr)

    y_pred = (y_proba > best_thr).astype(int)

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    scores.append({'fold': fold, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})
    models.append(model)
    print(
        f"Fold {fold + 1}: AUC={model.best_score_['valid_0']['auc']:.3f}, thr={best_thr:.3f}, Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")

print("\n=== Итоги кросс-валидации ===")
for metric in ['accuracy', 'precision', 'recall', 'f1']:
    values = [s[metric] for s in scores]
    print(f"{metric.capitalize()}: {np.mean(values):.3f} ± {np.std(values):.3f}")

# Сохраняем лучшую модель (последнюю) и средний порог
best_model = models[-1]
best_model.booster_.save_model('sber_lgb_model.txt')

avg_threshold = np.mean(thresholds)
print(f"\nСредний оптимальный порог: {avg_threshold:.3f}")
with open('threshold.txt', 'w') as f:
    f.write(str(avg_threshold))

print("💾 Модель сохранена в sber_lgb_model.txt, порог в threshold.txt")

# Анализ важности признаков
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Топ-10 важных признаков ===")
print(importance.head(10).to_string(index=False))