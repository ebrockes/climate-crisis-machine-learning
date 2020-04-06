import pandas as pd 
import calendar
from datetime import datetime

# Import Numpy, a library meant for large arrays - we will use it for its NaN representation 
import numpy as np

# Import Matplotlib
import matplotlib.pyplot as plt


# Function definition
def populate_df_with_anomolies_from_row(row):
    year = row['Year']
    
    # Anomaly values (they seem to be a mixture of strings and floats)
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
        
# Define function to convert values to floats, and return a 'NaN = Not a Number' if this is not possible
def clean_anomaly_value(raw_value):
    try:
        return float(raw_value)
    except:
        return np.NaN

# Read in the raw temperature and emissions datasets (they are in CSV format) 
raw_t = pd.read_csv('./data/GLB.Ts+dSST.csv', skiprows=1)
#print(raw_t.head())

# Create new dataframe with an index for each month
# First create the date range
date_rng = pd.date_range(start='1/1/1880', end='1/03/2019', freq='M')
#print(date_rng)

# Next create the empty DataFrame, which we will populate using the actual data
t = pd.DataFrame(date_rng, columns=['date'])

# Create a column for the anomoly values
t['Avg_Anomaly_deg_C'] = None

# Set the index of the DataFrame to the date column (DateTime index)
t.set_index('date', inplace=True)

# Show the first few elements
#print(t.head())

raw_t = raw_t.iloc[:,:13]

# Apply function to each row of raw data 
_ = raw_t.apply(lambda row: populate_df_with_anomolies_from_row(row), axis=1)

# Show the first few elements of our newly populated DataFrame
#print(t.head())

# Apply above function to all anomaly values in DataFrame
t['Avg_Anomaly_deg_C'] = t['Avg_Anomaly_deg_C'].apply(lambda raw_value: clean_anomaly_value(raw_value))

# 'Forward fill' to take care of NaN values
t.fillna(method='ffill', inplace=True)

# Show the first few elements of our newly cleaned DataFrame
print(t.head())

# Create figure, title and plot data
plt.figure(figsize=(10,8))
plt.xlabel('Time')
plt.ylabel('Temperature Anomaly (°Celsius)')
plt.plot(t, color='#1C7C54', linewidth=1.0)
plt.show()

#Let’s downsample our temperature data into years, the string ‘A’ represents ‘calendar year-end’.
t.resample('A').mean().head()

# Create figure, title and plot resampled data
plt.figure(figsize=(10,8))
plt.xlabel('Time')
plt.ylabel('Temperature Anomaly (°Celsius)')
plt.plot(t.resample('A').mean(), color='#1C7C54', linewidth=1.0)
plt.show()