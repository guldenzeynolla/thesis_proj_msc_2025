import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob
import plotly.express as px

from sklearn.ensemble import IsolationForest

from openai import OpenAI
import time

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

from itertools import combinations

# Log File
log_file_path = "strisciate_vc/log_veronaCard.csv"
log_df = pd.read_csv(log_file_path, encoding='utf-8')

print("Log file head:")
print(log_df.head(), "\n")

print("Log file columns:")
print(log_df.columns.tolist(), "\n")

print("Log file info:")
print(log_df.info(), "\n")

print("Log file summary statistics:")
print(log_df.describe(include='all'))

# Inspect raw values before conversion
print("\nRaw values in 'attivazione' before conversion:")
print(log_df['attivazione'].head(20))
print("\nRaw values in 'istante' before conversion:")
print(log_df['istante'].head(20))

# Convert datetime columns in log_df
# For 'attivazione', the raw format is 'YYYY-MM-DD'
if 'attivazione' in log_df.columns:
    log_df['attivazione'] = pd.to_datetime(log_df['attivazione'], format='%Y-%m-%d', errors='coerce')

if 'istante' in log_df.columns:
    log_df['istante'] = log_df['istante'].str.strip() 
    log_df['istante'] = pd.to_datetime(log_df['istante'], format='%d/%m/%Y %H:%M', errors='coerce')

print("\nLog file after datetime conversion:")
print(log_df.head(), "\n")
print(log_df.info())

# Combine Yearly Files (2014-2020)
years = list(range(2014, 2021))
file_paths = [f'strisciate_vc/veronacard_opendata/veronacard_{year}_opendata.csv' for year in years]
dfs_dict = {}

print("\nReading yearly files and displaying first few rows:")
for file_path in file_paths:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        dfs_dict[file_path] = df
        print(f"{file_path}:")
        print(df.head(), "\n")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

print("Checking column structure for each file:")
columns_dict = {}
for file_path, df in dfs_dict.items():
    columns_dict[file_path] = list(df.columns)
    print(f"{file_path}:")
    print(f"  Number of columns: {len(df.columns)}")
    print(f"  Columns: {list(df.columns)}\n")
all_columns = list(columns_dict.values())
same_structure = all(cols == all_columns[0] for cols in all_columns)
print("All files have the same column structure:", same_structure)

if same_structure:
    combined_df = pd.concat(list(dfs_dict.values()), ignore_index=True)
    print("\nCombined DataFrame shape:", combined_df.shape)
    print("\nInfo about the combined DataFrame:")
    print(combined_df.info())

    # Convert datetime columns in combined_df
    # Combine 'data_visita' and 'ora_visita' into 'visit_datetime'
    if 'data_visita' in combined_df.columns and 'ora_visita' in combined_df.columns:
        combined_df['visit_datetime'] = pd.to_datetime(
            combined_df['data_visita'].astype(str) + ' ' + combined_df['ora_visita'].astype(str),
            format='%d/%m/%Y %H:%M', errors='coerce'
        )
    # Convert 'data_attivazione' into 'activation_datetime'
    if 'data_attivazione' in combined_df.columns:
        combined_df['activation_datetime'] = pd.to_datetime(
            combined_df['data_attivazione'], format='%d/%m/%Y', errors='coerce'
        )
    
    print("\nCombined DataFrame after datetime conversion:")
    print(combined_df.head(), "\n")
    print(combined_df.info())
else:
    print("File structures differ; further analysis is required before combining the files.")

# Data Cleaning
# Combined Yearly DataFrame 
print("Combined DataFrame Missing Values:")
print(combined_df.isnull().sum())

if 'visit_datetime' in combined_df.columns:
    missing_visit_dt = combined_df['visit_datetime'].isnull().sum()
    print(f"Missing visit_datetime values: {missing_visit_dt}")

if 'activation_datetime' in combined_df.columns:
    missing_activation_dt = combined_df['activation_datetime'].isnull().sum()
    print(f"Missing activation_datetime values: {missing_activation_dt}")

# Check duplicates in combined_df
duplicates_combined = combined_df.duplicated().sum()
print(f"Number of duplicate rows in combined_df: {duplicates_combined}")
combined_df_clean = combined_df.drop_duplicates()
print(f"Combined DataFrame shape after dropping duplicates: {combined_df_clean.shape}")

# Log DataFrame 
print("\nLog DataFrame Missing Values:")
print(log_df.isnull().sum())

if 'attivazione' in log_df.columns:
    missing_attivazione = log_df['attivazione'].isnull().sum()
    print(f"Missing attivazione values: {missing_attivazione}")

if 'istante' in log_df.columns:
    missing_istante = log_df['istante'].isnull().sum()
    print(f"Missing istante values: {missing_istante}")

# Check duplicates in log_df
duplicates_log = log_df.duplicated().sum()
print(f"Number of duplicate rows in log_df: {duplicates_log}")
log_df_clean = log_df.drop_duplicates()
print(f"Log DataFrame shape after dropping duplicates: {log_df_clean.shape}")


#EDA tiiiime

# Агрегируем количество посещений по месяцам и график с усреднением (скользящее среднее за 3 месяца)
combined_df_clean.set_index('visit_datetime', inplace=True)
monthly_visits = combined_df_clean.resample('M').size()
monthly_visits_smoothed = monthly_visits.rolling(window=3).mean()
plt.figure(figsize=(15, 6))
plt.plot(monthly_visits, color='skyblue', alpha=0.5, label='Monthly visits')
plt.plot(monthly_visits_smoothed, color='blue', label='3-month rolling mean')
plt.title('Monthly Visits with Rolling Average (2014–2020)')
plt.xlabel('Year')
plt.ylabel('Number of Visits')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# Среднее количество посещений каждого месяца за все годы
combined_df_clean['month'] = combined_df_clean.index.month
avg_monthly_visits = combined_df_clean.groupby('month').size() / combined_df_clean.index.year.nunique()
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_monthly_visits.index, y=avg_monthly_visits.values, palette='Blues_r')
plt.title('Average Monthly Visits (2014–2020)')
plt.xlabel('Month')
plt.ylabel('Average Number of Visits')
plt.grid(alpha=0.3)
plt.show()


#Топ-10 самых популярных точек интереса (POI)
top_pois = combined_df_clean['sito_nome'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.barplot(y=top_pois.index, x=top_pois.values, palette='viridis')
plt.title('Top 10 Most Visited POIs')
plt.xlabel('Number of Visits')
plt.ylabel('POI')
plt.grid(alpha=0.3, axis='x')
plt.show()

#по часам
combined_df_clean['hour_of_day'] = combined_df_clean.index.hour
plt.figure(figsize=(14, 6))
sns.countplot(x='hour_of_day', data=combined_df_clean, palette='coolwarm')
plt.title('Visits by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Visits')
plt.grid(alpha=0.3)
plt.show()

#по дням недель
combined_df_clean['day_of_week'] = combined_df_clean.index.day_name()
plt.figure(figsize=(12, 6))
sns.countplot(x='day_of_week', data=combined_df_clean,
              order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
              palette='Spectral')
plt.title('Visits by Day of Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Visits')
plt.grid(alpha=0.3)
plt.show()

# maybe we can play with cordinates(why we have them....) 
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=combined_df_clean,
    x='sito_longitudine',
    y='sito_latitudine',
    alpha=0.3,
)
plt.title('Geographic distribution of visits')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(alpha=0.3)
plt.show()


# i cant understand anithyyyyyyng so lets try plotly, берем unique
df_unique = combined_df_clean[['sito_nome', 'sito_latitudine', 'sito_longitudine']].drop_duplicates()
fig = px.scatter_mapbox(
    df_unique,
    lat="sito_latitudine",
    lon="sito_longitudine",
    hover_name="sito_nome",
    zoom=13,
    height=600,
    mapbox_style="open-street-map",
    color_discrete_sequence=["darkorange"],  
    size_max=15
)
fig.update_traces(marker=dict(size=10))
fig.update_layout(title="Interactive Map of Visits in Verona")
fig.show()

#lets add weather
daily_visits = combined_df_clean.resample('D').size().to_frame('visits')
df_wth = pd.read_csv('UTF-8weather.csv', encoding='utf-8', parse_dates=['data'])
df_wth = df_wth.set_index('data')[['temp','rain']]

# объединение по дате (inner, чтобы отобрать совпадающие дни)
df_weather_vis = daily_visits.join(df_wth, how='inner')

plt.figure(figsize=(8,4))
plt.scatter(df_weather_vis['temp'], df_weather_vis['visits'], alpha=0.6)
plt.title('Daily Visits vs Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Visits per Day')
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(8,4))
plt.scatter(df_weather_vis['rain'], df_weather_vis['visits'], alpha=0.6)
plt.title('Daily Visits vs Rainfall')
plt.xlabel('Rainfall (mm)')
plt.ylabel('Visits per Day')
plt.grid(alpha=0.3)
plt.show()

# Корреляции
corr_t = df_weather_vis['visits'].corr(df_weather_vis['temp'])
corr_r = df_weather_vis['visits'].corr(df_weather_vis['rain'])
print(f"Correlation Visits–Temp: {corr_t:.2f}")
print(f"Correlation Visits–Rain: {corr_r:.2f}")


# LOESS-линию на temp vs visits
df_clean = df_weather_vis[['temp', 'visits']].dropna().sort_values(by='temp')
x = df_clean['temp'].values
y = df_clean['visits'].values

lowess = sm.nonparametric.lowess
smoothed = lowess(endog=y, exog=x, frac=0.2)

plt.figure(figsize=(10, 5))
plt.scatter(x, y, alpha=0.4, label='Data')
plt.plot(smoothed[:, 0], smoothed[:, 1], color='red', label='LOESS curve')
plt.title('LOESS Smoothing: Daily Visits vs Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Number of Visits')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("loess_temp_visits_manual.pdf")
plt.show()


# LOESS для rain
df_rain = df_weather_vis[['rain', 'visits']].dropna()
loess_rain = lowess(endog=df_rain['visits'], exog=df_rain['rain'], frac=0.2)

plt.figure(figsize=(10, 5))
plt.scatter(df_rain['rain'], df_rain['visits'], alpha=0.4, label='Data')
plt.plot(loess_rain[:, 0], loess_rain[:, 1], color='red', label='LOESS curve')
plt.title('LOESS Smoothing: Daily Visits vs Rainfall')
plt.xlabel('Rainfall (mm)')
plt.ylabel('Number of Visits')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()



# Barplot средних визитов по бинам температуры
df_weather_vis['temp_bin'] = pd.cut(df_weather_vis['temp'], 
                                    bins=range(0, 36, 5), right=False)
temp_means = df_weather_vis.groupby('temp_bin')['visits'].mean().reset_index()

plt.figure(figsize=(8,4))
sns.barplot(x='temp_bin', y='visits', data=temp_means, palette='Blues')
plt.xticks(rotation=45)
plt.title('Average Visits by Temperature Bin')
plt.xlabel('Temperature Bin (°C)')
plt.ylabel('Average Visits per Day')
plt.grid(alpha=0.3, axis='y')
plt.show()

# Barplot средних визитов по бинам осадков
df_weather_vis['rain_bin'] = pd.cut(df_weather_vis['rain'], 
                                    bins=[0,1,5,15, df_weather_vis['rain'].max()],
                                    labels=['0–1','1–5','5–15','15+'], right=False)
rain_means = df_weather_vis.groupby('rain_bin')['visits'].mean().reset_index()

plt.figure(figsize=(8,4))
sns.barplot(x='rain_bin', y='visits', data=rain_means, palette='Blues_r')
plt.title('Average Visits by Rainfall Bin')
plt.xlabel('Rainfall Bin (mm)')
plt.ylabel('Average Visits per Day')
plt.grid(alpha=0.3, axis='y')
plt.show()


#ANOMALY DETECTIOOOOOON START!!!!


high_thr = daily_visits['visits'].quantile(0.95)
low_thr  = daily_visits['visits'].quantile(0.05)
daily_visits['high_volume'] = (daily_visits['visits'] >  high_thr).astype(int)
daily_visits['low_volume']  = (daily_visits['visits'] <  low_thr ).astype(int)

model_if = IsolationForest(contamination=0.02, random_state=42)
daily_visits['anomaly_if'] = model_if.fit_predict(daily_visits[['visits']])

# with high/low
model_if2 = IsolationForest(contamination=0.02, random_state=42)
X2 = daily_visits[['visits','high_volume','low_volume']]
daily_visits['anomaly_if2'] = model_if2.fit_predict(X2)

# to see difference
plt.figure(figsize=(15,6))
plt.plot(daily_visits.index, daily_visits['visits'],
         color='gray', alpha=0.5, label='Visits')

# original IF
mask1 = daily_visits['anomaly_if'] == -1
plt.scatter(daily_visits.index[mask1],
            daily_visits['visits'][mask1],
            color='red', marker='x', s=50, label='IF (original)')

# IF with high/low
mask2 = daily_visits['anomaly_if2'] == -1
plt.scatter(daily_visits.index[mask2],
            daily_visits['visits'][mask2],
            color='blue', marker='o', s=50, label='IF (with high/low)')

plt.title("Isolation Forest: original vs с with high_volume/low_volume")
plt.xlabel("Date")
plt.ylabel("Number of Visits")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# ОРИГИНАЛЬНЫЙ LSTM-БЛОК (двухступенчатый автоэнкодер на visits) 
n_steps = 14     # длина окна в днях
n_features = 1   # у нас один признак — visits

# Подготовка скользящих окон 
values = daily_visits['visits'].values.reshape(-1, 1)
scaler = MinMaxScaler()
values_scaled = scaler.fit_transform(values)

windows = []
for i in range(len(values_scaled) - n_steps + 1):
    windows.append(values_scaled[i : i + n_steps])
windows = np.array(windows)  # (num_windows, n_steps, 1)
X = windows.reshape((windows.shape[0], n_steps, n_features))

# Автоэнкодер (оригинальный)
inputs = Input(shape=(n_steps, n_features))
encoded = LSTM(64, activation='relu')(inputs)
decoded = RepeatVector(n_steps)(encoded)
decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
outputs = TimeDistributed(Dense(n_features))(decoded)
autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(
    X, X,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    shuffle=True
)

X_pred = autoencoder.predict(X)
mse = np.mean((X_pred - X)**2, axis=(1,2))
threshold = np.quantile(mse, 0.98)       # 2% самых больших ошибок
lstm_flags = (mse > threshold).astype(int)

lstm_dates = daily_visits.index[n_steps - 1:].to_list()
lstm_df = pd.DataFrame({
    'date': lstm_dates,
    'anomaly_lstm_old': lstm_flags
}).set_index('date')

daily_visits = daily_visits.join(lstm_df)
daily_visits['anomaly_lstm_old'] = daily_visits['anomaly_lstm_old'].fillna(0).astype(int)

plt.figure(figsize=(15,5))
plt.plot(daily_visits.index, daily_visits['visits'], label='Visits', alpha=0.5)
plt.scatter(
    daily_visits.index[daily_visits['anomaly_lstm_old']==1],
    daily_visits['visits'][daily_visits['anomaly_lstm_old']==1],
    c='green', label='LSTM Anomaly (old)', s=50
)
plt.title("LSTM-AE (original): anomalies")
plt.legend()
plt.show()


# НОВЫЙ MF-LSTM-БЛОК (3-feature автоэнкодер) 
daily_visits['high_volume'] = (
    daily_visits['visits'] > daily_visits['visits'].quantile(0.95)
).astype(int)
daily_visits['low_volume'] = (
    daily_visits['visits'] < daily_visits['visits'].quantile(0.05)
).astype(int)

# Формируем windows из трёх фич
fv = daily_visits[['visits','high_volume','low_volume']].values
fv_scaled = MinMaxScaler().fit_transform(fv)
n_steps = 14
X_mf = np.array([fv_scaled[i:i+n_steps] for i in range(len(fv_scaled)-n_steps+1)])
n_features = X_mf.shape[2]

# Строим и обучаем MF-LSTM-AE
inputs = Input(shape=(n_steps, n_features))
encoded = LSTM(64, activation='relu')(inputs)
decoded = RepeatVector(n_steps)(encoded)
decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
outputs = TimeDistributed(Dense(n_features))(decoded)
ae_mf = Model(inputs, outputs)
ae_mf.compile(optimizer='adam', loss='mse')

history_mf = ae_mf.fit(
    X_mf, X_mf,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    shuffle=True
)

# Делаем детекцию по MSE
X_pred_mf = ae_mf.predict(X_mf)
mse_mf = np.mean((X_pred_mf - X_mf)**2, axis=(1,2))
threshold_mf = np.quantile(mse_mf, 0.98)  # настраиваемый квантиль

dates_mf = daily_visits.index[n_steps-1:]
daily_visits['anomaly_lstm_new'] = 0
daily_visits.loc[dates_mf[mse_mf>threshold_mf], 'anomaly_lstm_new'] = 1

# Визуализируем новый блок
plt.figure(figsize=(15,5))
plt.plot(daily_visits.index, daily_visits['visits'], alpha=0.4, label='Visits')
plt.scatter(
    daily_visits.index[daily_visits['anomaly_lstm_new']==1],
    daily_visits['visits'][daily_visits['anomaly_lstm_new']==1],
    c='purple', label='LSTM Anomaly (new)', s=50
)
plt.title("LSTM-AE (3 features): anomalies")
plt.legend()
plt.show()

daily_visits.to_csv(
    'anomalies_report.csv',   
    index=False,               
    encoding='utf-8-sig'     
)

# Сравнение старого и нового LSTM 
old_count = daily_visits['anomaly_lstm_old'].sum()
new_count = daily_visits['anomaly_lstm_new'].sum()
print(f"Old LSTM flagged {old_count} anomalies")
print(f"New LSTM flagged {new_count} anomalies")

both    = ((daily_visits['anomaly_lstm_old']==1) & (daily_visits['anomaly_lstm_new']==1)).sum()
union   = ((daily_visits['anomaly_lstm_old']==1) | (daily_visits['anomaly_lstm_new']==1)).sum()
jaccard = both / union if union else 0
print(f"Both flagged {both}, union {union}, Jaccard={jaccard:.2f}")

# confusion matrix
conf_mat = pd.crosstab(daily_visits['anomaly_lstm_old'], daily_visits['anomaly_lstm_new'],
                       rownames=['Old'], colnames=['New'])
print("\nConfusion matrix:\n", conf_mat)

# тепловая карта
plt.figure(figsize=(5,4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Old vs New LSTM Flags')
plt.show()

# timeline
plt.figure(figsize=(12,3))
plt.plot(daily_visits.index, daily_visits['anomaly_lstm_old'], drawstyle='steps-mid',
         label='Old LSTM', alpha=0.7)
plt.plot(daily_visits.index, daily_visits['anomaly_lstm_new'], drawstyle='steps-mid',
         label='New LSTM', alpha=0.7)
plt.ylim(-0.1,1.1)
plt.title('Timeline of LSTM Anomaly Flags')
plt.legend()
plt.show()

# даты расхождений
diffs = daily_visits[daily_visits['anomaly_lstm_old'] != daily_visits['anomaly_lstm_new']]
print(f"\nDates with differing flags ({len(diffs)}):")
print(diffs[['anomaly_lstm_old','anomaly_lstm_new']])

# Merge daily weather data with anomaly flags
df_weather_vis = df_weather_vis.join(
    daily_visits[['anomaly_if', 'anomaly_lstm_old', 'anomaly_lstm_new']]
)

# Temperature vs IF anomalies
plt.figure(figsize=(8, 4))
sns.boxplot(x='anomaly_if', y='temp', data=df_weather_vis)
plt.title('Temperature: Normal vs IF-Anomalous Days')
plt.xlabel('Anomaly by IF (-1 = normal, 1 = anomaly)')
plt.ylabel('Temperature (°C)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Temperature vs LSTM anomalies
plt.figure(figsize=(8, 4))
sns.boxplot(x='anomaly_lstm_new', y='temp', data=df_weather_vis)
plt.title('Temperature: Normal vs LSTM-Anomalous Days')
plt.xlabel('Anomaly by LSTM (0 = normal, 1 = anomaly)')
plt.ylabel('Temperature (°C)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Rainfall vs IF anomalies
plt.figure(figsize=(8, 4))
sns.boxplot(x='anomaly_if', y='rain', data=df_weather_vis)
plt.title('Rainfall: Normal vs IF-Anomalous Days')
plt.xlabel('Anomaly by IF (-1 = normal, 1 = anomaly)')
plt.ylabel('Rainfall (mm)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Rainfall vs LSTM anomalies
plt.figure(figsize=(8, 4))
sns.boxplot(x='anomaly_lstm_new', y='rain', data=df_weather_vis)
plt.title('Rainfall: Normal vs LSTM-Anomalous Days')
plt.xlabel('Anomaly by LSTM (0 = normal, 1 = anomaly)')
plt.ylabel('Rainfall (mm)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Mean weather values in normal vs anomalous days
print("Average weather conditions on normal vs IF-anomalous days:")
print(df_weather_vis.groupby('anomaly_if')[['temp', 'rain']].mean())

print("\nAverage weather conditions on normal vs LSTM-anomalous days:")
print(df_weather_vis.groupby('anomaly_lstm_new')[['temp', 'rain']].mean())


# Объединяем visits с погодой
df_for_if = daily_visits.join(df_wth, how='inner')
df_for_if = df_for_if.dropna(subset=['visits', 'temp', 'rain'])

# Модель Isolation Forest на 3 признаках
features = ['visits', 'temp', 'rain']
model_if_weather = IsolationForest(contamination=0.02, random_state=42)
df_for_if['anomaly_if_weather'] = model_if_weather.fit_predict(df_for_if[features])

# Обновляем основной датафрейм
daily_visits['anomaly_if_weather'] = df_for_if['anomaly_if_weather']

#LSTM with weather

# Признаки: visits, high/low volume, temp, rain
df_lstm = daily_visits.join(df_wth, how='inner').copy()
df_lstm = df_lstm.dropna(subset=['visits', 'temp', 'rain'])

df_lstm['high_volume'] = (df_lstm['visits'] > df_lstm['visits'].quantile(0.95)).astype(int)
df_lstm['low_volume']  = (df_lstm['visits'] < df_lstm['visits'].quantile(0.05)).astype(int)

features = ['visits', 'high_volume', 'low_volume', 'temp', 'rain']
X_vals = df_lstm[features].values
X_scaled = MinMaxScaler().fit_transform(X_vals)

# sliding window
n_steps = 14
X_seq = np.array([X_scaled[i:i+n_steps] for i in range(len(X_scaled)-n_steps+1)])
n_features = X_seq.shape[2]

# Model
inputs = Input(shape=(n_steps, n_features))
encoded = LSTM(64, activation='relu')(inputs)
decoded = RepeatVector(n_steps)(encoded)
decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
outputs = TimeDistributed(Dense(n_features))(decoded)

ae_weather = Model(inputs, outputs)
ae_weather.compile(optimizer='adam', loss='mse')

ae_weather.fit(X_seq, X_seq, epochs=20, batch_size=32, validation_split=0.1, shuffle=True)

# Predictions
X_pred_weather = ae_weather.predict(X_seq)
mse_weather = np.mean((X_pred_weather - X_seq)**2, axis=(1,2))
threshold_weather = np.quantile(mse_weather, 0.98)

# Аномалии
dates_lstm = df_lstm.index[n_steps - 1:]
df_lstm['anomaly_lstm_weather'] = 0
df_lstm.loc[dates_lstm[mse_weather > threshold_weather], 'anomaly_lstm_weather'] = 1

# Объединяем обратно
daily_visits['anomaly_lstm_weather'] = df_lstm['anomaly_lstm_weather']


def compare_flags(df, columns, label):
    print(f"\n🔍 Comparison: {label}")
    for a, b in combinations(columns, 2):
        a_flags = df[a].replace({-1:1})  # IF uses -1 for anomaly
        b_flags = df[b].replace({-1:1})
        both = ((a_flags == 1) & (b_flags == 1)).sum()
        union = ((a_flags == 1) | (b_flags == 1)).sum()
        jaccard = both / union if union else 0
        print(f"{a} vs {b}:")
        print(f"  {a} anomalies: {a_flags.sum()}")
        print(f"  {b} anomalies: {b_flags.sum()}")
        print(f"  Both: {both}, Union: {union}, Jaccard: {jaccard:.2f}\n")

# --- Сравнение IF версий ---
compare_flags(
    daily_visits,
    ['anomaly_if', 'anomaly_if2', 'anomaly_if_weather'],
    label='Isolation Forest Versions'
)

# --- Сравнение LSTM версий ---
compare_flags(
    daily_visits,
    ['anomaly_lstm_old', 'anomaly_lstm_new', 'anomaly_lstm_weather'],
    label='LSTM Versions'
)

#Пример для IF:
plt.figure(figsize=(12,4))
plt.plot(daily_visits.index, (daily_visits['anomaly_if']==-1).astype(int), drawstyle='steps-mid', label='IF (visits)', alpha=0.5)
plt.plot(daily_visits.index, (daily_visits['anomaly_if2']==-1).astype(int), drawstyle='steps-mid', label='IF (visits + vol)', alpha=0.7)
plt.plot(daily_visits.index, (daily_visits['anomaly_if_weather']==-1).astype(int), drawstyle='steps-mid', label='IF (with weather)', alpha=0.9)
plt.ylim(-0.1,1.1)
plt.title('Comparison of Isolation Forest Anomaly Flags')
plt.legend()
plt.tight_layout()
plt.show()

#LSTM
plt.figure(figsize=(12,4))
plt.plot(daily_visits.index, daily_visits['anomaly_lstm_old'], drawstyle='steps-mid', label='LSTM (visits)', alpha=0.5)
plt.plot(daily_visits.index, daily_visits['anomaly_lstm_new'], drawstyle='steps-mid', label='LSTM (+vol)', alpha=0.7)
plt.plot(daily_visits.index, daily_visits['anomaly_lstm_weather'], drawstyle='steps-mid', label='LSTM (with weather)', alpha=0.9)
plt.ylim(-0.1,1.1)
plt.title('Comparison of LSTM Anomaly Flags')
plt.legend()
plt.tight_layout()
plt.show()


# Приводим флаги к 0/1
daily_visits['IF_visits'] = (daily_visits['anomaly_if'] == -1).astype(int)
daily_visits['IF_vol'] = (daily_visits['anomaly_if2'] == -1).astype(int)
daily_visits['IF_weather'] = (daily_visits['anomaly_if_weather'] == -1).astype(int)

daily_visits['LSTM_visits'] = daily_visits['anomaly_lstm_old']
daily_visits['LSTM_vol'] = daily_visits['anomaly_lstm_new']
daily_visits['LSTM_weather'] = daily_visits['anomaly_lstm_weather']

def compare_three_versions(df, version_list, model_name):
    rows = []
    for a, b in combinations(version_list, 2):
        A = df[a]
        B = df[b]
        both = ((A == 1) & (B == 1)).sum()
        only_a = ((A == 1) & (B == 0)).sum()
        only_b = ((A == 0) & (B == 1)).sum()
        union = ((A == 1) | (B == 1)).sum()
        jaccard = both / union if union else 0
        overlap_a = both / A.sum() * 100 if A.sum() else 0
        overlap_b = both / B.sum() * 100 if B.sum() else 0
        disagreements = (A != B).sum()
        rows.append({
            'Model': model_name,
            'Version A': a,
            'Version B': b,
            'Anomalies A': A.sum(),
            'Anomalies B': B.sum(),
            'Both Anomalies': both,
            'Only in A': only_a,
            'Only in B': only_b,
            'Disagreements': disagreements,
            'Jaccard Similarity': round(jaccard, 2),
            'Overlap % A→B': round(overlap_a, 1),
            'Overlap % B→A': round(overlap_b, 1),
        })
    return pd.DataFrame(rows)

# Сравнение IF
if_versions = ['IF_visits', 'IF_vol', 'IF_weather']
df_if_compare = compare_three_versions(daily_visits, if_versions, model_name='Isolation Forest')

# Сравнение LSTM
lstm_versions = ['LSTM_visits', 'LSTM_vol', 'LSTM_weather']
df_lstm_compare = compare_three_versions(daily_visits, lstm_versions, model_name='LSTM')

# Вывод
print("\n=== Isolation Forest (3 versions) ===")
print(df_if_compare)

print("\n=== LSTM Autoencoder (3 versions) ===")
print(df_lstm_compare)

#add AI

API_KEY = os.getenv("REMOVED_OPENAI_KEY")
MODEL_NAME = "gpt-4"           # или "gpt-4o", "gpt-3.5-turbo-16k" 
OUTPUT_FILE = "verona_contextual_events.csv"

client = OpenAI(api_key=API_KEY)

anomaly_dates = anomalies.index.strftime('%Y-%m-%d').tolist()

if os.path.exists(OUTPUT_FILE):
    print(f"\n Events file found: {OUTPUT_FILE}")
    events_df = pd.read_csv(OUTPUT_FILE)
    existing_dates = events_df['date'].tolist()
else:
    print(f"\n No events file found. Creating new...")
    events_df = pd.DataFrame(columns=['date', 'event_summary'])
    existing_dates = []

new_dates = [date for date in anomaly_dates if date not in existing_dates]
print(f"\n Found {len(new_dates)} new anomaly dates")

events_list = []

def get_event_summary(date):
    prompt = (f"What important events, festivals, incidents, or special conditions occurred "
              f"in Verona, Italy around the date {date}? Provide a short, clear summary.")
    chat_completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300
    )
    return chat_completion.choices[0].message.content.strip()

for date in new_dates:
    print(f"Processing date: {date}")
    try:
        summary = get_event_summary(date)
        print(f"Event: {summary}\n")
        events_list.append({'date': date, 'event_summary': summary})
        time.sleep(1.5)
    except Exception as e:
        print(f"Error on {date}: {e}")

if events_list:
    new_events_df = pd.DataFrame(events_list)
    events_df = pd.concat([events_df, new_events_df], ignore_index=True)
    events_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n Events file updated: '{OUTPUT_FILE}'")
else:
    print("\n No new events")

print(events_df.head())

# Добавим колонку "context_event" в DataFrame с аномалиями.
#Если событие есть в verona_contextual_events.csv, оно подтянется.
#Если нет → будет "No event found".
events_df = pd.read_csv('verona_contextual_events.csv')
anomalies_with_context = anomalies.copy()
anomalies_with_context['date'] = anomalies_with_context.index.strftime('%Y-%m-%d')
anomalies_with_context = anomalies_with_context.merge(events_df, how='left', on='date')

print(anomalies_with_context[['visits', 'event_summary']].head())

#Анализ и интерпретация результатов. Посчитать процент аномалий, совпавших с событиями
total_anomalies = len(anomalies_with_context)
matched_events = anomalies_with_context['event_summary'].notnull().sum()
print(f"Events found for {matched_events} out of {total_anomalies} anomalies ({matched_events/total_anomalies:.2%})")
plt.figure(figsize=(12,6))
sns.countplot(y=anomalies_with_context['event_summary'].notnull(), palette='Set2')
plt.title('Anomalies with/without Contextual Event')
plt.xlabel('Count')
plt.ylabel('Has Event')
plt.grid(alpha=0.3)
plt.show()



# LSTM-периоды 
# Группировка LSTM-аномалий в периоды
df = daily_visits.copy()
df['flag'] = df['anomaly_lstm']
df['grp'] = (df['flag'] != df['flag'].shift()).cumsum()

periods_lstm = (
    df[df['flag']==1]
      .groupby('grp')
      .agg(
          start_date=('flag','idxmin'),
          end_date  =('flag','idxmax')
      )
      .reset_index(drop=True)
)

print("LSTM intervals:\n", periods_lstm)

# Новая функция для AI-запросов
def get_event_summary_lstm(start, end):
    prompt = (
        f"You are a factual assistant. Using only verified sources (e.g. Wikipedia), "
        f"list major public events, festivals or incidents that actually took place in Verona, Italy "
        f"between {start} and {end}. "
        "If there are no records, answer “No known events in this period.”"
    )

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=200
    )
    return resp.choices[0].message.content.strip()

# Сбор объяснений по каждому интервалу
OUTPUT_LSTM = "verona_lstm_contextual_events.csv"
if os.path.exists(OUTPUT_LSTM):
    events_lstm_df = pd.read_csv(OUTPUT_LSTM)
    existing_intervals = set(
        zip(events_lstm_df['start_date'], events_lstm_df['end_date'])
    )
else:
    events_lstm_df = pd.DataFrame(columns=['start_date','end_date','explanation'])
    existing_intervals = set()

new_rows = []
for _, row in periods_lstm.iterrows():
    sd = row['start_date'].date().isoformat()
    ed = row['end_date'].date().isoformat()
    if (sd, ed) in existing_intervals:
        continue
    print(f"Processing interval: {sd} → {ed}")
    summary = get_event_summary_lstm(sd, ed)
    new_rows.append({'start_date': sd, 'end_date': ed, 'explanation': summary})
    time.sleep(1.5)

if new_rows:
    events_lstm_df = pd.concat([events_lstm_df, pd.DataFrame(new_rows)], ignore_index=True)
    events_lstm_df.to_csv(OUTPUT_LSTM, index=False)
    print(f"\nLSTM events file updated: '{OUTPUT_LSTM}'")
else:
    print("\nNo new LSTM intervals to process")

print(events_lstm_df.head())


