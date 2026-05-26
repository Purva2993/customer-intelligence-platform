import pandas as pd

def run(data_dir='data'):
      print('[1/6] Cleaning data...')
      df = pd.read_excel(f'{data_dir}/Online Retail.xlsx', dtype={'CustomerID': str})

      df.dropna(subset=['CustomerID', 'Description'], inplace=True)
      df = df[df['Quantity']   > 0]
      df = df[df['UnitPrice']  > 0]
      df['CustomerID']  = df['CustomerID'].astype(int)
      df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
      df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

      df.to_csv(f'{data_dir}/cleaned_retail.csv', index=False)
      print(f'    Saved cleaned_retail.csv — {df.shape[0]:,} rows, {df["CustomerID"].nunique():,} customers')
      return True
