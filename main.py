# %% main code
import os
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from itertools import combinations
from sklearn.metrics import jaccard_score
import calendar
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import ttest_ind

from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline





def read_csv(path):
    try:
        df = pd.read_csv(path, encoding='utf-8')
        print(f"Loaded {path}: {len(df)} rows")
        return df
    except FileNotFoundError:
        print(f"File not found: {path}")
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return pd.DataFrame()

#  Обработка логов 
log_path = "strisciate_vc/log_veronaCard.csv"
log_df = read_csv(log_path)

# Конвертируем даты
if "attivazione" in log_df:
    log_df["attivazione"] = pd.to_datetime(
        log_df["attivazione"], format="%Y-%m-%d", errors="coerce"
    )
if "istante" in log_df:
    log_df["istante"] = pd.to_datetime(
        log_df["istante"].str.strip(), format="%d/%m/%Y %H:%M", errors="coerce"
    )

print("\nLog DataFrame info after datetime conversion:")
print(log_df.info())

# Убираем дубликаты
before = len(log_df)
log_df = log_df.drop_duplicates().reset_index(drop=True)
print(f"Removed {before - len(log_df)} duplicate rows in log_df\n")

# --- Обработка годовых файлов ---
years = range(2014, 2021)
folder = "strisciate_vc/veronacard_opendata"
list_dfs = []

for year in years:
    path = os.path.join(folder, f"veronacard_{year}_opendata.csv")
    df = read_csv(path)
    if not df.empty:
        list_dfs.append(df)

# Проверяем, что есть файлы для объединения
if not list_dfs:
    raise RuntimeError("No yearly files loaded. Проверьте пути и наличие файлов.")

# Проверяем одинаковость структуры
cols0 = set(list_dfs[0].columns)
if not all(set(df.columns) == cols0 for df in list_dfs):
    raise RuntimeError("Column structures differ between yearly files.")

# Объединяем
combined_df = pd.concat(list_dfs, ignore_index=True)
print(f"\nCombined DataFrame shape: {combined_df.shape}")

# Конвертируем даты посещений и активаций
if "data_visita" in combined_df and "ora_visita" in combined_df:
    combined_df["visit_datetime"] = pd.to_datetime(
        combined_df["data_visita"].astype(str).str.strip()
        + " "
        + combined_df["ora_visita"].astype(str).str.strip(),
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )
if "data_attivazione" in combined_df:
    combined_df["activation_datetime"] = pd.to_datetime(
        combined_df["data_attivazione"], format="%d/%m/%Y", errors="coerce"
    )

print("\nCombined DataFrame info after datetime conversion:")
print(combined_df.info())

# Убираем дубликаты
before = len(combined_df)
combined_df_clean = combined_df.drop_duplicates().reset_index(drop=True)
print(f"Removed {before - len(combined_df_clean)} duplicate rows in combined_df\n")

# Итоговая проверка пропусков
print("Missing values in log_df:")
print(log_df.isnull().sum()[lambda x: x > 0], "\n")

print("Missing values in combined_df_clean:")
print(combined_df_clean.isnull().sum()[lambda x: x > 0])

# %% EDA tiiiime

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
df_wth = pd.read_csv('strisciate_vc/weather_db.csv', encoding='utf-8', parse_dates=['data'])
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


# %% ANOMALY DETECTIOOOOOON START!!!!

# Создаёт и обучает модель LSTM Autoencoder, которая учится 
# восстанавливать входные последовательности. Ошибки восстановления (MSE) 
# позже используются для поиска аномалий.
def fit_lstm_autoencoder(X, n_steps, n_features, epochs=20):
    inputs = Input(shape=(n_steps, n_features))
    encoded = LSTM(64, activation='relu')(inputs)
    decoded = RepeatVector(n_steps)(encoded)
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    outputs = TimeDistributed(Dense(n_features))(decoded)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, X, epochs=epochs, batch_size=32, validation_split=0.1, shuffle=True, verbose=0)
    return model


# Вычисляет MSE между предсказанием модели и реальными данными, 
# и помечает точки с высокой ошибкой как аномалии (выше заданного квантиля).
 
def get_lstm_anomalies(X, model, threshold_std=3):
    preds = model.predict(X)
    mse = np.mean((preds - X)**2, axis=(1, 2))

    # Вычисляем порог как среднее + несколько стандартных отклонений
    threshold = np.mean(mse) + threshold_std * np.std(mse)
    return mse, (mse > threshold).astype(int)


# Строит график посещений с выделенными аномальными точками определённого типа.
def plot_anomalies(df, flag_col, color, label, title):
    plt.figure(figsize=(15, 5))
    plt.plot(df.index, df['visits'], alpha=0.5, label='Visits')
    plt.scatter(df.index[df[flag_col] == 1], df['visits'][df[flag_col] == 1],
                color=color, s=50, label=label)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# Сравнивает погодные условия (температура, дождь) в нормальные и аномальные дни. 
# Используется t-тест для оценки значимости различий.
def compare_weather_stats(df, col_flag, feature):
    normal = df[df[col_flag] == 0][feature]
    anomaly = df[df[col_flag] == 1][feature]
    t_stat, p_val = ttest_ind(normal, anomaly, equal_var=False)
    print(f"{feature}: normal={normal.mean():.2f}, anomaly={anomaly.mean():.2f}, p={p_val:.4f}")


# Сравнивает попарно различные модели по количеству совпадений аномалий, расхождений, 
# Jaccard-индексу и проценту перекрытия.
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


def print_metrics(y_true, y_pred):
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1:", f1_score(y_true, y_pred))


# Загрузка и предобработка данных предполагается заранее

# Признаки объема
high_thr = daily_visits['visits'].quantile(0.95)
low_thr = daily_visits['visits'].quantile(0.05)
daily_visits['high_volume'] = (daily_visits['visits'] > high_thr).astype(int)
daily_visits['low_volume'] = (daily_visits['visits'] < low_thr).astype(int)

# --- Isolation Forest ---
model_if = IsolationForest(random_state=42)
daily_visits['anomaly_if'] = model_if.fit_predict(daily_visits[['visits']])

model_if2 = IsolationForest(random_state=42)
daily_visits['anomaly_if2'] = model_if2.fit_predict(
    daily_visits[['visits', 'high_volume', 'low_volume']]
)

model_if_weather = IsolationForest(random_state=42)
df_for_if = daily_visits.join(df_wth).dropna()
df_for_if['anomaly_if_weather'] = model_if_weather.fit_predict(
    df_for_if[['visits', 'temp', 'rain']]
)
daily_visits['anomaly_if_weather'] = df_for_if['anomaly_if_weather']

# --- LSTM AE (1 фича) ---
n_steps = 14
values = daily_visits['visits'].values.reshape(-1, 1)
values_scaled = MinMaxScaler().fit_transform(values)
X = np.array([values_scaled[i:i+n_steps] for i in range(len(values_scaled)-n_steps+1)])
lstm_model = fit_lstm_autoencoder(X, n_steps, 1)
mse, flags = get_lstm_anomalies(X, lstm_model)
daily_visits['anomaly_lstm_old'] = 0
daily_visits.iloc[n_steps-1:, daily_visits.columns.get_loc('anomaly_lstm_old')] = flags

# --- LSTM AE (+volume) ---
fv = daily_visits[['visits', 'high_volume', 'low_volume']].values
fv_scaled = MinMaxScaler().fit_transform(fv)
X_mf = np.array([fv_scaled[i:i+n_steps] for i in range(len(fv_scaled)-n_steps+1)])
lstm_model2 = fit_lstm_autoencoder(X_mf, n_steps, 3)
mse2, flags2 = get_lstm_anomalies(X_mf, lstm_model2)
daily_visits['anomaly_lstm_new'] = 0
daily_visits.iloc[n_steps-1:, daily_visits.columns.get_loc('anomaly_lstm_new')] = flags2

# --- LSTM AE (+weather) ---
lstm_df = daily_visits.join(df_wth).dropna().copy()
lstm_df['high_volume'] = (lstm_df['visits'] > high_thr).astype(int)
lstm_df['low_volume'] = (lstm_df['visits'] < low_thr).astype(int)
features = ['visits', 'high_volume', 'low_volume', 'temp', 'rain']
X_vals = lstm_df[features].values
X_scaled = MinMaxScaler().fit_transform(X_vals)
X_seq = np.array([X_scaled[i:i+n_steps] for i in range(len(X_scaled)-n_steps+1)])
lstm_model3 = fit_lstm_autoencoder(X_seq, n_steps, X_seq.shape[2])
mse3, flags3 = get_lstm_anomalies(X_seq, lstm_model3)
lstm_df['anomaly_lstm_weather'] = 0
lstm_df.iloc[n_steps-1:, lstm_df.columns.get_loc('anomaly_lstm_weather')] = flags3
daily_visits['anomaly_lstm_weather'] = lstm_df['anomaly_lstm_weather']

# --- Экспорт и сравнение ---
daily_visits['IF_visits'] = (daily_visits['anomaly_if'] == -1).astype(int)
daily_visits['IF_vol'] = (daily_visits['anomaly_if2'] == -1).astype(int)
daily_visits['IF_weather'] = (daily_visits['anomaly_if_weather'] == -1).astype(int)
daily_visits['LSTM_visits'] = daily_visits['anomaly_lstm_old']
daily_visits['LSTM_vol'] = daily_visits['anomaly_lstm_new']
daily_visits['LSTM_weather'] = daily_visits['anomaly_lstm_weather']

anomaly_flags = ['IF_visits','IF_vol','IF_weather','LSTM_visits','LSTM_vol','LSTM_weather']
anomalies = daily_visits[daily_visits[anomaly_flags].sum(axis=1) > 0].copy()
anomalies = anomalies.reset_index().rename(columns={'index': 'date'})
anomalies.to_csv("anomalies_with_context.csv", index=False)
print("Saved: anomalies_with_context.csv")

# Сравнение версий моделей
print("\n=== Isolation Forest (3 versions) ===")
df_if_compare = compare_three_versions(daily_visits, ['IF_visits', 'IF_vol', 'IF_weather'], 'Isolation Forest')
print(df_if_compare)

print("\n=== LSTM Autoencoder (3 versions) ===")
df_lstm_compare = compare_three_versions(daily_visits, ['LSTM_visits', 'LSTM_vol', 'LSTM_weather'], 'LSTM')
print(df_lstm_compare)


# %% Визуализация различий 

# Isolation Forest: original vs with high/low volume
plt.figure(figsize=(15,6))
plt.plot(daily_visits.index, daily_visits['visits'], color='gray', alpha=0.5, label='Visits')
plt.scatter(daily_visits.index[daily_visits['anomaly_if'] == -1], daily_visits['visits'][daily_visits['anomaly_if'] == -1], color='red', marker='x', s=50, label='IF (original)')
plt.scatter(daily_visits.index[daily_visits['anomaly_if2'] == -1], daily_visits['visits'][daily_visits['anomaly_if2'] == -1], color='blue', marker='o', s=50, label='IF (with high/low)')
plt.title("Isolation Forest: original vs with high_volume/low_volume")
plt.xlabel("Date")
plt.ylabel("Number of Visits")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# LSTM-AE (original)
plt.figure(figsize=(15,5))
plt.plot(daily_visits.index, daily_visits['visits'], label='Visits', alpha=0.5)
plt.scatter(daily_visits.index[daily_visits['anomaly_lstm_old']==1], daily_visits['visits'][daily_visits['anomaly_lstm_old']==1], c='green', label='LSTM Anomaly (old)', s=50)
plt.title("LSTM-AE (original): anomalies")
plt.legend()
plt.show()

# LSTM-AE (3 features)
plt.figure(figsize=(15,5))
plt.plot(daily_visits.index, daily_visits['visits'], alpha=0.4, label='Visits')
plt.scatter(daily_visits.index[daily_visits['anomaly_lstm_new']==1], daily_visits['visits'][daily_visits['anomaly_lstm_new']==1], c='purple', label='LSTM Anomaly (new)', s=50)
plt.title("LSTM-AE (3 features): anomalies")
plt.legend()
plt.show()

# Timeline of old vs new LSTM flags
plt.figure(figsize=(12,3))
plt.plot(daily_visits.index, daily_visits['anomaly_lstm_old'], drawstyle='steps-mid', label='Old LSTM', alpha=0.7)
plt.plot(daily_visits.index, daily_visits['anomaly_lstm_new'], drawstyle='steps-mid', label='New LSTM', alpha=0.7)
plt.ylim(-0.1,1.1)
plt.title('Timeline of LSTM Anomaly Flags')
plt.legend()
plt.show()

# Temperature vs IF anomalies
plt.figure(figsize=(8, 4))
sns.boxplot(x='anomaly_if', y='temp', data=df_for_if)
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
sns.boxplot(x='anomaly_if', y='rain', data=df_for_if)
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

# Comparison: IF flags timeline
plt.figure(figsize=(15,3))
plt.plot(daily_visits.index, (daily_visits['anomaly_if']==-1).astype(int), drawstyle='steps-mid', label='IF (visits)', alpha=0.5)
plt.plot(daily_visits.index, (daily_visits['anomaly_if2']==-1).astype(int), drawstyle='steps-mid', label='IF (+vol)', alpha=0.7)
plt.plot(daily_visits.index, (daily_visits['anomaly_if_weather']==-1).astype(int), drawstyle='steps-mid', label='IF (weather)', alpha=0.9)
plt.ylim(-0.1,1.1)
plt.title('Comparison of Isolation Forest Anomaly Flags')
plt.legend()
plt.tight_layout()
plt.show()

# Comparison: LSTM flags timeline
plt.figure(figsize=(15,3))
plt.plot(daily_visits.index, daily_visits['anomaly_lstm_old'], drawstyle='steps-mid', label='LSTM (visits)', alpha=0.5)
plt.plot(daily_visits.index, daily_visits['anomaly_lstm_new'], drawstyle='steps-mid', label='LSTM (+vol)', alpha=0.7)
plt.plot(daily_visits.index, daily_visits['anomaly_lstm_weather'], drawstyle='steps-mid', label='LSTM (weather)', alpha=0.9)
plt.ylim(-0.1,1.1)
plt.title('Comparison of LSTM Anomaly Flags')
plt.legend()
plt.tight_layout()
plt.show()

# Heatmap: Сравнение моделей по Jaccard Similarity
models = ['IF_visits', 'IF_vol', 'IF_weather', 'LSTM_visits', 'LSTM_vol', 'LSTM_weather']
flags = daily_visits[models].copy()
jaccard_matrix = pd.DataFrame(index=models, columns=models)

for a in models:
    for b in models:
        score = jaccard_score(flags[a], flags[b]) if a != b else 1.0
        jaccard_matrix.loc[a, b] = round(score, 2)

plt.figure(figsize=(8,6))
sns.heatmap(jaccard_matrix.astype(float), annot=True, cmap='YlGnBu')
plt.title("Jaccard Similarity between Models")
plt.show()


# Group anomalies by month and weekday
calendar_df = daily_visits.copy()
calendar_df['month'] = calendar_df.index.month
calendar_df['weekday'] = calendar_df.index.weekday
monthly_counts = calendar_df.groupby('month')[models].sum()
weekday_counts = calendar_df.groupby('weekday')[models].sum()
# Month 
plt.figure(figsize=(10,4))
sns.heatmap(monthly_counts.T, cmap='viridis', annot=True)
plt.title("Monthly Anomaly Counts per Model")
plt.xlabel("Month")
plt.ylabel("Model")
plt.show()
# Weekday heatmap
plt.figure(figsize=(10,4))
sns.heatmap(weekday_counts.T, cmap='plasma', annot=True)
plt.title("Weekday Anomaly Counts per Model (0=Mon)")
plt.xlabel("Weekday")
plt.ylabel("Model")
plt.show()

#table with all 
anomaly_cols = ['IF_visits', 'IF_vol', 'IF_weather', 'LSTM_visits', 'LSTM_vol', 'LSTM_weather']
for anomaly in anomaly_cols:
    count = daily_visits[anomaly].sum()
    print(f"{anomaly}: {count} аномалий")


## После детекции аномалий (IF/LSTM) мы не всегда знаем причину — это может быть COVID, фестиваль, климат и т.д.
# Чтобы интерпретировать найденные аномалии, мы подключаем LLM — GPT-4 и Mixtral.
# Они получают одинаковые промпты и объясняют: ПОЧЕМУ день или период мог быть аномальным.
# %% AI Initialization
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1) Загрузить ключ из файла aikey.env
load_dotenv(dotenv_path="aikey.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Не найден OPENAI_API_KEY в файле aikey.env")

client = OpenAI(api_key=OPENAI_API_KEY)

def ask_openai(prompt: str, model: str = "gpt-4") -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300
    )
    return resp.choices[0].message.content.strip()

# 3) Ленивый HF-пайплайн для бесплатной модели
FREE_MODEL_NAME = "bigscience/bloom-560m"
_hf_pipe = None

def ask_free_model(prompt: str, max_tokens: int = 150) -> str:
    global _hf_pipe
    if _hf_pipe is None:
        print("Loading HF model…")
        tok = AutoTokenizer.from_pretrained(FREE_MODEL_NAME)
        mdl = AutoModelForCausalLM.from_pretrained(FREE_MODEL_NAME, device_map="auto")
        _hf_pipe = pipeline("text-generation", model=mdl, tokenizer=tok, device_map="auto")
        print("HF model ready.")
    out = _hf_pipe(prompt, max_new_tokens=max_tokens, do_sample=False)[0]["generated_text"]
    return out.replace(prompt, "").strip()

def clean_response(resp: str) -> str:
    """
    Normalize AI response: trim whitespace, catch various 'no events' patterns.
    """
    if not resp or not resp.strip():
        return "No response"
    low = resp.lower()
    neg_patterns = [
        "no events found", "no known events", "no response",
        "nothing special", "нет событий", "нет особых"
    ]
    for pat in neg_patterns:
        if pat in low:
            return "No events found"
    return resp.strip()

print("✅ AI Initialization complete.")



# %%
