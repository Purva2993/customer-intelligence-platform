import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def run(data_dir='data'):
    print('[5/6] Generating recommendations...')
    df  = pd.read_csv(f'{data_dir}/cleaned_retail.csv', parse_dates=['InvoiceDate'])
    rfm = pd.read_csv(f'{data_dir}/rfm_segmented.csv')
    valid   = set(rfm['CustomerID'].unique())
    df_f    = df[df['CustomerID'].isin(valid)]
    cp = df_f.pivot_table(
        index='CustomerID', columns='Description',
        values='Quantity', aggfunc='sum'
    ).fillna(0)
    cp = (cp > 0).astype(int)

    item_sim = pd.DataFrame(
        cosine_similarity(cp.T), index=cp.columns, columns=cp.columns
    )

    def get_recs(cid, n=5):
              if cid not in cp.index: return []
              bought     = cp.loc[cid]
              bought_items = bought[bought == 1].index.tolist()
              if not bought_items: return []
              scores = item_sim[bought_items].mean(axis=1).drop(labels=bought_items, errors='ignore')
              return scores.nlargest(n).index.tolist()

    repeat    = rfm[rfm['Frequency'] >= 2]['CustomerID'].tolist()
    recs_cf   = {cid: get_recs(cid) for cid in repeat}
    recs_cf   = {k: v for k, v in recs_cf.items() if v}
    seg_top = (
            df_f.merge(rfm[['CustomerID','SegmentName']], on='CustomerID', how='left')
            .groupby(['SegmentName','Description'])['CustomerID'].nunique()
            .reset_index().sort_values(['SegmentName','CustomerID'], ascending=[True,False])
            .groupby('SegmentName').head(5)
        )
    rows = []
    for _, c in rfm.iterrows():                                                                                                                                                                                                      
        cid, seg = c['CustomerID'], c['SegmentName']
        if cid in recs_cf:
            products, method = recs_cf[cid], 'Collaborative Filtering'
        else:
            products = seg_top[seg_top['SegmentName']==seg]['Description'].tolist()
            method   = 'Segment Fallback'
        for rank, prod in enumerate(products, 1):
            rows.append({'CustomerID': cid, 'SegmentName': seg,
                         'Rank': rank, 'Recommended': prod, 'Method': method})
    pd.DataFrame(rows).to_csv(f'{data_dir}/recommendations.csv', index=False)
    print(f'    Saved recommendations.csv — {len(rows):,} rows')
    return True