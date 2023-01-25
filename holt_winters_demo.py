#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Last uptade: 25/01/2023

@author: Miguel Burgos
"""

# ------------------------------- LIBRARIES -------------------------------

import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error as mse


# --------------------------- DATA EXPLORATION ---------------------------

df = pd.read_csv('datasets/airline-passengers.csv',
                  header=0, index_col=0, parse_dates=['Month'])

print(df.head())
print(df.describe())

# plotting
plt.figure(1)

plt.plot(df)
plt.xlabel("Months")
plt.ylabel("Passengers")
plt.title("Total passengers over time")

plt.show()

plt.figure(2)

plot_acf(df, fft=1)
plt.xlabel("Periods (months)")
plt.ylabel("Passengers")
plt.title("Autocorrelation over periods")

plt.show()


# -------------------------------- MODEL --------------------------------
model = ExponentialSmoothing(endog=df['Passengers'], seasonal_periods=12,
                             trend="mul", seasonal="mul").fit(optimized=1)
df['Holtwinters'] = model.fittedvalues
prediction = model.forecast(30)

# Error
print("MSE Holt-Winters: ", mse(df['Passengers'], df['Holtwinters']))

# plotting
plt.figure(3)

plt.plot(df.index, df['Passengers'], color='blue', label='data')
plt.plot(prediction, color='pink', label='Holt-Winters')
plt.xlabel("Months")
plt.ylabel("Passengers")
plt.title("Total passengers over time")

plt.show()