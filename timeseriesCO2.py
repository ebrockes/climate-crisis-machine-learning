# Import the Pandas library
import pandas as pd 
# import Matplotlib
import matplotlib.pyplot as plt
# Allow for graphs to be displayed in Jupyter notebook
#%matplotlib inline

# Read in the raw temperature and emissions datasets (they are in CSV format) 
raw_e = pd.read_csv('./data/API_EN.ATM.CO2E.PC_DS2_en_csv_v2_821708.csv', skiprows=3)

# Define function to pull value from raw data, using DateIndex from new DataFrame row
def populate_df(row):
    index = str(row['date'].year)
    value = raw_e_world.loc[index]
    return value
    
# Select just the co2 emissions for the 'world', and the columns for the years 1960-2018 
raw_e_world = raw_e[raw_e['Country Name']=='World'].loc[:,'1960':'2018']


# 'Traspose' the resulting slice, making the columns become rows and vice versa
raw_e_world = raw_e_world.T
raw_e_world.columns = ['value']

# Create a new DataFrame with a daterange the same the range for.. 
# the Temperature data (after resampling to years)
date_rng = pd.date_range(start='31/12/1960', end='31/12/2018', freq='y')
e = pd.DataFrame(date_rng, columns=['date'])

# Populate the new DataFrame using the values from the raw data slice
v = e.apply(lambda row: populate_df(row), axis=1)
e['Global CO2 Emissions per Capita'] = v
e.set_index('date', inplace=True)

e.fillna(method='ffill', inplace=True)

plt.figure(figsize=(10,8))
plt.xlabel('Time (Years)')
plt.ylabel('Emissions (Metric Tons per Capita)')
plt.plot(e, color='#1C7C54', linewidth=1.0)
plt.show()
