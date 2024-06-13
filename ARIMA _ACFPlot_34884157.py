#!/usr/bin/env python
# coding: utf-8

# In[1]:


from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt


# ### Visual Inspection: The original data was visually inspected using line plots to understand its general trends, patterns, and seasonality.

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'K54Ddata_34884157.xlsx'
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Plot the original data using a line plot
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Earning'], label='Earning')
plt.title('Original Earning Data')
plt.xlabel('Date')
plt.ylabel('Earning')
plt.legend()
plt.grid(True)
plt.show()


# ### Statistical Summary: Descriptive statistics such as mean, median, standard deviation, etc., were computed to gain insights into the central tendency and variability of the data.

# In[3]:


# Compute descriptive statistics
statistics = data['Earning'].describe()

# Print the statistical summary
print("Statistical Summary for Earning Data:")
print(statistics)


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm

# Load the dataset
file_path = 'K54Ddata_34884157.xlsx'
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Plot ACF and PACF plots to identify model parameters
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(data['Earning'], ax=ax[0], lags=40)
plot_pacf(data['Earning'], ax=ax[1], lags=40)
plt.show()


# ## Model

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the dataset
file_path = 'K54Ddata_34884157.xlsx'
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Find the best parameters using auto_arima
auto_model = auto_arima(data['Earning'], seasonal=True, m=12, trace=True)

# Get the best model parameters
order = auto_model.order
seasonal_order = auto_model.seasonal_order

print("Best model parameters (p, d, q):", order)
print("Best seasonal parameters (P, D, Q, S):", seasonal_order)

# Fit SARIMAX model with the best parameters
model = SARIMAX(data['Earning'], order=order, seasonal_order=seasonal_order)
fit_model = model.fit()

# Generate forecasts until December 2024 (48 months)
forecast_index = pd.date_range(start=data.index[-1], periods=13, freq='M')[1:]  # Forecast index for the next 48 months
forecast = fit_model.forecast(steps=len(forecast_index))  # Forecast for the next 48 months

# Plot original data and forecasts
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Earning'], label='Actual')
plt.plot(forecast_index, forecast, label='Forecast', linestyle='--')
plt.title('SARIMA Forecasting until December 2024')
plt.xlabel('Date')
plt.ylabel('Earning')
plt.legend()
plt.grid(True)
plt.show()



# In[6]:


# Extract forecasted values from December 2023 to December 2024
forecast_dec_2023_to_dec_2024 = forecast[(forecast_index >= '2023-12-01') & (forecast_index <= '2025-01-01')]

# Display the forecasted values
print(forecast_dec_2023_to_dec_2024)


# In[7]:


from sklearn.metrics import mean_squared_error
import numpy as np

# Define the actual values for the forecast period
actual_values = data['Earning'][-12:]  # Assuming the last 12 months are the test period

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_values, forecast))

# Print RMSE
print("Root Mean Squared Error (RMSE):", rmse)


# In[ ]:





# In[ ]:




