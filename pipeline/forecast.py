import pandas as pd
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

def run(data_dir='data'):                                                                                                                                                                                                            
      print('[6/6] Forecasting revenue...')
      df = pd.read_csv(f'{data_dir}/cleaned_retail.csv', parse_dates=['InvoiceDate'])

      df['Month'] = df['InvoiceDate'].dt.to_period('M')
      monthly     = df.groupby('Month')['TotalAmount'].sum().reset_index()
      monthly['Month'] = monthly['Month'].dt.to_timestamp()
      monthly.columns  = ['ds', 'y']
      monthly_train    = monthly[monthly['ds'] < '2011-12-01']

      model = Prophet(
          yearly_seasonality=False, weekly_seasonality=False,
          daily_seasonality=False, interval_width=0.95,
          changepoint_prior_scale=0.05
      )
      model.fit(monthly_train)

      future   = model.make_future_dataframe(periods=4, freq='MS')
      forecast = model.predict(future)
      forecast['yhat']       = forecast['yhat'].clip(lower=0)
      forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
      forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

      out = forecast[['ds','yhat','yhat_lower','yhat_upper']].copy()
      out.columns = ['Month','Forecast_Revenue','Lower_Bound','Upper_Bound']
      out.to_csv(f'{data_dir}/revenue_forecast.csv', index=False)
      print('    Saved revenue_forecast.csv')
      print('    Forecast (next 3 months):')
      for _, r in out[out['Month'] > '2011-11-30'].iterrows():
          print(f"      {r['Month'].strftime('%b %Y')}: £{r['Forecast_Revenue']:,.0f}")
      return True