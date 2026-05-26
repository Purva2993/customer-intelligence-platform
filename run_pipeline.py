import sys, os
sys.path.insert(0, os.path.dirname(__file__))
  
from pipeline import clean, features, churn, segment, recommend, forecast

def run_full_pipeline(data_dir='data', models_dir='models'):
      print('=' * 50)
      print('Customer Intelligence Platform — Full Pipeline')
      print('=' * 50)

      steps = [
          ('Data Cleaning',        lambda: clean.run(data_dir)),
          ('Feature Engineering',  lambda: features.run(data_dir)),
          ('Churn Prediction',     lambda: churn.run(data_dir, models_dir)),
          ('Segmentation',         lambda: segment.run(data_dir)),
          ('Recommendations',      lambda: recommend.run(data_dir)),
          ('Revenue Forecasting',  lambda: forecast.run(data_dir)),
      ]

      for name, step in steps:
          try:
              step()
          except Exception as e:                                                                                                                                                                                                       
              print(f'\n[FAILED] {name}: {e}')
              sys.exit(1)

      print('=' * 50)
      print('Pipeline complete. All outputs saved to data/')
      print('=' * 50)

if __name__ == '__main__':
      run_full_pipeline()    


      
                                                                                                           