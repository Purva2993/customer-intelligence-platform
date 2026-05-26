import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
CLUSTER_FEATURES  = ['Recency', 'Frequency', 'LogMonetary', 'PurchaseSpread']
SEGMENT_NAMES     = {2: 'Champions', 0: 'Loyal', 3: 'At-Risk', 1: 'Lost'}
def run(data_dir='data'):
      print('[4/6] Segmenting customers...')
      rfm      = pd.read_csv(f'{data_dir}/rfm_with_churn.csv')
      scaler   = StandardScaler()
      X_scaled = scaler.fit_transform(rfm[CLUSTER_FEATURES])

      km = KMeans(n_clusters=4, random_state=42, n_init=10)
      rfm['Segment']     = km.fit_predict(X_scaled)
      rfm['SegmentName'] = rfm['Segment'].map(SEGMENT_NAMES)

      rfm.to_csv(f'{data_dir}/rfm_segmented.csv', index=False)
      print(f'    Saved rfm_segmented.csv')
      print('    Distribution:', rfm['SegmentName'].value_counts().to_dict())
      return True
