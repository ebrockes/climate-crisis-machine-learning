# https://towardsdatascience.com/time-series-analysis-and-climate-change-7bb4371021e

# Import the Pandas library
import pandas as pd
import numpy as np
# Import Matplotlib
import matplotlib.pyplot as plt
import io, requests

import calendar
from datetime import datetime

# Import Facebook's Prophet forecasting library
from fbprophet import Prophet

# Function definition
def populate_df_with_anomolies_from_row(row):
	year = row['Year']
	# Anomaly values (they seem to be a mixture of strings and floats)
    # data.iloc[<row_selection>,<column_selection>]
	monthly_anomolies = row.iloc[1:]
	# Abbreviated month names (index names)
	months = monthly_anomolies.index
	for month in monthly_anomolies.index:
		# Get the last day for each month 
		last_day = calendar.monthrange(year,datetime.strptime(month, '%b').month)[1]
		# construct the index with which we can reference our new DataFrame (to populate) 
		date_index = datetime.strptime(f'{year} {month} {last_day}', '%Y %b %d')
		# Populate / set value @ above index, to anomaly value
		t.loc[date_index] = monthly_anomolies[month]
		
def clean_anomaly_value(raw_value):
	try:
		return float(raw_value)
	except:
		return np.NaN
 
# Read in the raw temperature and emissions datasets (they are in CSV format)
url = 'https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv'
s = requests.get(url).content
raw_t = pd.read_csv(io.StringIO(s.decode('utf-8')), skiprows=1)
# raw_t = pd.read_csv('./data/GLB.Ts+dSST.csv')

# Create new dataframe with an index for each month
# Last day of each month from 31/01/1880 to 31/12/2018
date_rng = pd.date_range(start='1/1/1880', end='1/03/2019', freq='M')

# Next create the empty DataFrame, which we will populate using the actual data
# each line contains an element of date_rng
# index increment from 1
t = pd.DataFrame(date_rng, columns=['date'])

# Create a column for the anomoly values
# index increment from 1
t['Avg_Anomaly_deg_C'] = None

# Set the index of the DataFrame to the date column (DateTime index)
t.set_index('date', inplace=True)

# data.iloc[<row_selection>,<column_selection>]
#,:13 - da primeira a 13a coluna
# https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/
raw_t = raw_t.iloc[:,:13]

# Apply function to each row of raw data 
_ = raw_t.apply(lambda row: populate_df_with_anomolies_from_row(row), axis=1)

# Apply above function to all anomaly values in DataFrame
t['Avg_Anomaly_deg_C'] = t['Avg_Anomaly_deg_C'].apply(lambda raw_value: clean_anomaly_value(raw_value))

# 'Forward fill' to take care of NaN values
t.fillna(method='ffill', inplace=True)

# Show the first few elements
print(t.head())

# Allow for rendering within notebook
#%matplotlib inline

# Create a new DataFrame with which we will create/train our Prophet model 
t_prophet = pd.DataFrame()
t_prophet['ds'] = t.index
t_prophet['y'] = t['Avg_Anomaly_deg_C'].values

# Instantiate model and fit to data (just like with sklearn model API)
m = Prophet()
m.fit(t_prophet)

# Generate future dataframe containing predictions (we are doing this for 100 years into the future)
future = m.make_future_dataframe(freq='m', periods=100*12)
forecast = m.predict(future)

# Plot the resulting forecast
m.plot(forecast)