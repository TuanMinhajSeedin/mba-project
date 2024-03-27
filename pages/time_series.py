import numpy as np 
import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly.express  as px
import datetime
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title='Recomendation',page_icon='',layout='wide')
custom_css = """
<style>
body {
    background-color: #0E1117; 
    secondary-background {
    background-color: #262730; 
    padding: 10px; 
}
</style>
"""
st.write(custom_css, unsafe_allow_html=True)
st.markdown(custom_css, unsafe_allow_html=True)
st.title('Time Series Analysis')

df = pd.read_csv('dataset/Assignment-1_Data.csv',sep=';')
df['Date'] =  pd.to_datetime(df['Date'],format="%d.%m.%Y %H:%M")
df.set_index(df['Date'],inplace=True)
df['Price'] = df['Price'].str.replace(',', '.')
df['Total'] = df['Quantity'].astype(float) * df['Price'].astype(float)
df=df[['Total']]

# df.to_csv('Outputs/time_series.csv')

# df1 = pd.read_csv('Outputs/time_series.csv').set_index('Date')

daily_sales = df.groupby(df.index.date)['Total'].sum().reset_index()
daily_sales.columns = ['Date', 'Sales']
# Add a sidebar for user input
st.sidebar.title("Select Time Range")
time_range = st.sidebar.selectbox("Select Time Range", ["Daily","Week", "Month", "Quarter", "Custom Range"])

# Process data based on selected time range
if time_range == "Daily":
    daily_sales=daily_sales
    seasonal_order=(1,1,1,30)

elif time_range == "Week":
    daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
    daily_sales = daily_sales.resample('W-Mon', on='Date').sum().reset_index()
    seasonal_order=(1,1,1,4)

elif time_range == "Month":
    daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
    daily_sales = daily_sales.resample('M', on='Date').sum().reset_index()
    seasonal_order=(1,1,1,12)

elif time_range == "Quarter":
    daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
    daily_sales = daily_sales.resample('Q', on='Date').sum().reset_index()
    seasonal_order=(1,1,1,4)

else:
    # Custom range can be implemented using date input widgets
    start_date = st.sidebar.date_input("Start Date",value=datetime.date(2011, 1, 1))
    end_date = st.sidebar.date_input("End Date",value=datetime.date(2011, 12, 1))
    daily_sales = daily_sales[(daily_sales['Date'] >= start_date) & (daily_sales['Date'] <= end_date)]
    seasonal_order=(1,1,1,30)


# st.write(daily_sales)
# Create the line plot
fig = px.line(daily_sales, x='Date', y='Sales', title="Sales Over Time")

# Update layout
fig.update_layout(
    width=1200,
    height=600,
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False)
)

# Display the Plotly Express figure in Streamlit
st.plotly_chart(fig)



def check_stationarity(df, param):
    stationarity = False
    adfTest = adfuller(df[param], autolag='AIC')
    stats = pd.DataFrame(adfTest[0:4], index=['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used'])
    stats.columns = ['Value']
    st.write(stats)
    for key, value in adfTest[4].items():
        st.write("Criticality ", key, " : ", value)
        if float(value) > float(adfTest[0]):
            stationarity = True
            break
    if stationarity == False and adfTest[1] <= 0.05:
        stationarity = True
    df['rollMean'] = df[param].rolling(window=7).mean()
    df['rollStd'] = df[param].rolling(window=7).std()
    
    # Create Plotly Express plot
    fig = px.line(df, x=df.index, y=[param, 'rollMean', 'rollStd'], 
                  title="Stationarity Applications")
    fig.update_layout(showlegend=True, width=1000, height=600, 
                      xaxis=dict(title='Time Period', showgrid=False),
                      yaxis=dict(title=param, showgrid=False))
    
    st.write("Stationarity: ", stationarity)
    st.plotly_chart(fig)

    return stationarity

daily_sales['log_sales']=daily_sales['Sales'].apply(lambda x: np.log(x))
daily_sales['log_sqrt']=daily_sales['log_sales'].apply(lambda x: np.sqrt(x))
# check_stationarity(daily_sales,'log_sqrt')
daily_sales['log_sqrt_shift_diff'] = daily_sales['log_sqrt'] - daily_sales['log_sqrt'].shift()
daily_sales['log_sqrt_shift_diff'].dropna(inplace=True)

daily_sales['log_sqrt_shift_diff'].fillna(0,inplace=True)
with st.expander("Check for Stationary"):
    check_stationarity(daily_sales, 'log_sqrt_shift_diff')

##-----------------------------------------------------------------------------------------------------------------------
train = daily_sales[:int(0.7*(len(daily_sales)))]
test = daily_sales[int(0.7*(len(daily_sales))):]
summary=SARIMAX(train['Sales'], order=(1, 1, 1), seasonal_order=(1,1,1,30)).fit().summary()
predictions = SARIMAX(train['Sales'], order=(1, 1, 1), seasonal_order=seasonal_order).fit().predict(start=len(train), end=len(train)+len(test)-1, typ='levels')
daily_sales['sarimax_prediction'] = predictions
daily_sales['sarimax_prediction'].dropna()
train_size = int(0.7 * len(daily_sales))
train_data = daily_sales.iloc[:train_size]
test_data = daily_sales.iloc[train_size:]


fig = go.Figure()
fig.add_trace(go.Scatter(x=train_data['Date'], y=train_data['Sales'], mode='lines', name='Train Data', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Sales'], mode='lines', name='Test Data', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=test_data['Date'], y=test_data['sarimax_prediction'], mode='lines', name='SARIMAX Prediction', line=dict(color='green')))
fig.update_layout(title='Model Evaluation',showlegend=True, width=1000, height=600, 
                  xaxis=dict(title='Time Period', showgrid=False),
                  yaxis=dict(title='Sales', showgrid=False))

col1,col2=st.columns((0.6,0.2))
with col1:
    st.plotly_chart(fig)

daily_sales_backup=daily_sales.copy()
daily_sales_backup['log_sales1']=daily_sales_backup['sarimax_prediction'].apply(lambda x: np.log(x))
daily_sales_backup['log_sqrt1']=daily_sales_backup['log_sales1'].apply(lambda x: np.sqrt(x))
# check_stationarity(daily_sales,'log_sqrt')
daily_sales_backup['log_sqrt_shift_diff1'] = daily_sales_backup['log_sqrt1'] - daily_sales_backup['log_sqrt1'].shift()
daily_sales_backup['log_sqrt_shift_diff1'].dropna(inplace=True)

daily_sales_backup['log_sqrt_shift_diff1'].fillna(0,inplace=True)
MSE_Daily=mean_squared_error(daily_sales['log_sqrt_shift_diff'], daily_sales_backup['log_sqrt_shift_diff1'])

# prediction = ARIMA(train['log_sqrt_shift_diff'], order=(1, 1, 3)).fit().predict(start=test.index[0], end=test.index[-1], typ='levels')
# MSE_Daily = mean_squared_error(daily_sales['log_sqrt_shift_diff'][int(0.7*(len(daily_sales))):], prediction)
with col2:
    with st.expander("View Sales Predictions"):
        st.write(daily_sales[['Date','Sales','sarimax_prediction']].set_index('Date').tail(17))
    st.write("Mean squared error  -: ",MSE_Daily)

y = daily_sales.set_index('Date')['Sales']
train_size = int(0.7 * len(y))
train = y.iloc[:train_size]
test = y.iloc[train_size:]

st.write('-'*1000)
col3,col4=st.columns((0.6,0.2))
future_pred = SARIMAX(train, order=(1, 1, 1), seasonal_order=seasonal_order).fit().predict(start=len(y), end=len(y)+12, typ='levels')
# future_pred.index=date_list
if time_range == "Daily":
    max_date = y.index.max()

    future_end_date = max_date + pd.Timedelta(days=13)
    start_date = max_date + pd.Timedelta(days=1)  
    end_date = future_end_date
    date_range = pd.date_range(start=start_date, end=end_date)
    date_list = date_range.tolist()
    future_pred = SARIMAX(train, order=(1, 1, 1), seasonal_order=seasonal_order).fit().predict(start=len(y), end=len(y)+12, typ='levels')

    future_pred.index=date_list
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=y.iloc[train_size:].index, y=y.iloc[train_size:], mode='lines', name='Current Sale', line=dict(color='blue')))
fig1.add_trace(go.Scatter(x=future_pred.index, y=future_pred, mode='lines', name='Future Data', line=dict(color='orange')))
fig1.update_layout(title='Sales Forecast',showlegend=True, width=1000, height=600, 
                  xaxis=dict(title='Time Period', showgrid=False),
                  yaxis=dict(title='Sales', showgrid=False))
with col3:
    st.plotly_chart(fig1)
with col4:
    with st.expander("View Sales Forecating"):
        st.write(future_pred)