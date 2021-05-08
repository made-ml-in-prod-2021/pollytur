import pandas as pd
from pandas_profiling import ProfileReport
data_path = '../data/raw/heart.csv'
df = pd.read_csv(data_path)
profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)
profile.to_file("../reports/eda_report.html")