from airflow.decorators import dag, task
from datetime import datetime
import sys, os
sys.path.insert(0, '/Users/purvamugdiya/Desktop/customer-intelligence-platform')
from pipeline import clean, features, churn, segment, recommend, forecast

DATA_DIR   = '/Users/purvamugdiya/Desktop/customer-intelligence-platform/data'
MODELS_DIR = '/Users/purvamugdiya/Desktop/customer-intelligence-platform/models'

@dag(
      dag_id          = 'customer_intelligence_pipeline',
      description     = 'End-to-end customer intelligence platform pipeline',
      schedule        = '@weekly',
      start_date      = datetime(2024, 1, 1),
      catchup         = False,
      tags            = ['ml', 'customer-intelligence']
  )
def customer_intelligence_pipeline():

      @task()
      def task_clean():
          clean.run(DATA_DIR)
  
      @task()
      def task_features():
          features.run(DATA_DIR)

      @task()
      def task_churn():
          churn.run(DATA_DIR, MODELS_DIR)

      @task()
      def task_segment():
          segment.run(DATA_DIR)

      @task()
      def task_recommend():
          recommend.run(DATA_DIR)                                                                                                                                                                                                      

      @task()
      def task_forecast():
          forecast.run(DATA_DIR)

      # Define execution order
      t1 = task_clean()
      t2 = task_features()
      t3 = task_churn()
      t4 = task_segment()
      t5 = task_recommend()
      t6 = task_forecast()
  
      t1 >> t2 >> t3 >> t4 >> t5 >> t6

dag_instance = customer_intelligence_pipeline()