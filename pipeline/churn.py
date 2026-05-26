import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle, os
FEATURE_COLS = [
      'Recency', 'Frequency', 'LogMonetary', 'LogAvgOrderValue',
      'PurchaseSpread', 'UniqueProducts', 'IsOneTimeBuyer', 'AvgDaysBetweenPurchases',
      'Q1_Orders', 'Q2_Orders', 'Q3_Orders', 'PurchaseTrend', 'ActiveInQ3'
  ]

def run(data_dir='data', models_dir='models'):
      print('[3/6] Training churn model...')
      os.makedirs(models_dir, exist_ok=True)

      rfm = pd.read_csv(f'{data_dir}/rfm_with_churn.csv')
      X   = rfm[FEATURE_COLS]
      y   = rfm['Churn']

      X_train, X_test, y_train, y_test = train_test_split(
          X, y, test_size=0.2, random_state=42, stratify=y
      )
  
      model = XGBClassifier(
          max_depth=3, n_estimators=200, learning_rate=0.05,
          subsample=0.9, colsample_bytree=0.8,
          random_state=42, eval_metric='logloss', verbosity=0
      )
      model.fit(X_train, y_train)

      roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
      print(f'    ROC-AUC: {roc_auc:.4f}')

      rfm['ChurnProbability'] = model.predict_proba(X)[:, 1]
      rfm['ChurnPredicted']   = model.predict(X)
      rfm.to_csv(f'{data_dir}/rfm_with_churn.csv', index=False)

      with open(f'{models_dir}/churn_model.pkl', 'wb') as f:
          pickle.dump(model, f)
      print(f'    Saved churn_model.pkl and updated rfm_with_churn.csv')
      return True