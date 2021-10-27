import pandas as pd
from prophet import Prophet

# pre-processing dataset
df = pd.read_csv('../../../datasets/example_wp_log_peyton_manning.csv')

#downsample dataset

df.head()

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)

