import pandas as pd
import numpy as np

  
def run(data_dir='data'):
    print('[2/6] Engineering features...')
    df = pd.read_csv(f'{data_dir}/cleaned_retail.csv', parse_dates=['InvoiceDate'])
    feature_end    = pd.Timestamp('2011-09-30')
    observe_start  = pd.Timestamp('2011-10-01')
    feature_df     = df[df['InvoiceDate'] <= feature_end].copy()
    observe_df     = df[df['InvoiceDate'] >= observe_start].copy()
    reference_date = feature_end + pd.Timedelta(days=1)
    # Core RFM features
    rfm = feature_df.groupby('CustomerID').agg(
        Recency        = ('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency      = ('InvoiceNo',   'nunique'),
        Monetary       = ('TotalAmount', 'sum'),
        UniqueProducts = ('StockCode',   'nunique'),
        PurchaseSpread = ('InvoiceDate', lambda x: (x.max() - x.min()).days),
        AvgOrderValue  = ('TotalAmount', 'mean')
    ).reset_index()
    rfm['PurchaseSpread'] = rfm['PurchaseSpread'].fillna(0)
    # Log-transformed and derived features
    rfm['LogMonetary']    = np.log1p(rfm['Monetary'])
    rfm['LogAvgOrderValue'] = np.log1p(rfm['AvgOrderValue'])
    rfm['IsOneTimeBuyer'] = (rfm['Frequency'] == 1).astype(int)
    rfm['AvgDaysBetweenPurchases'] = rfm.apply(
        lambda row: row['PurchaseSpread'] / (row['Frequency'] - 1)
        if row['Frequency'] > 1 else 0,
        axis=1
    )

    # Quarterly order counts
    quarters = [
        ('2011-01-01', '2011-03-31', 'Q1_Orders'),
        ('2011-04-01', '2011-06-30', 'Q2_Orders'),
        ('2011-07-01', '2011-09-30', 'Q3_Orders'),
    ]
    for start, end, col in quarters:
        q = feature_df[                                                                                                                                                                                                              
            (feature_df['InvoiceDate'] >= start) &
            (feature_df['InvoiceDate'] <= end)
        ].groupby('CustomerID')['InvoiceNo'].nunique().rename(col)
        rfm = rfm.merge(q.reset_index(), on='CustomerID', how='left')
        rfm[col] = rfm[col].fillna(0)
    rfm['PurchaseTrend'] = rfm['Q3_Orders'] - rfm['Q1_Orders']
    rfm['ActiveInQ3']    = (rfm['Q3_Orders'] > 0).astype(int)
    # Churn label — time-based split
    returning    = set(observe_df['CustomerID'].unique())
    rfm['Churn'] = rfm['CustomerID'].apply(lambda x: 0 if x in returning else 1)
    rfm.to_csv(f'{data_dir}/rfm_with_churn.csv', index=False)
    print(f'    Saved rfm_with_churn.csv — {rfm.shape[0]:,} customers, {rfm.shape[1]} features')
    return True