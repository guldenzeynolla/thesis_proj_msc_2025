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
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import ttest_ind
from sklearn.metrics import jaccard_score, roc_curve, precision_recall_curve, auc

import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.gofplots import qqplot
from scipy import stats



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

# %% lets add weather

daily_visits = combined_df_clean.resample('D').size().to_frame('visits')
# Добавим погодные данные к daily_visits
df_wth = pd.read_csv('strisciate_vc/weather_db.csv', encoding='utf-8', parse_dates=['data'])
df_wth = df_wth.set_index('data')[['temp', 'rain']]
daily_visits = daily_visits.join(df_wth, how='inner') # объединение по дате (inner, чтобы отобрать совпадающие дни)

plt.figure(figsize=(8,4))
plt.scatter(daily_visits['temp'], daily_visits['visits'], alpha=0.6)
plt.title('Daily Visits vs Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Visits per Day')
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(8,4))
plt.scatter(daily_visits['rain'], daily_visits['visits'], alpha=0.6)
plt.title('Daily Visits vs Rainfall')
plt.xlabel('Rainfall (mm)')
plt.ylabel('Visits per Day')
plt.grid(alpha=0.3)
plt.show()

# Корреляции
corr_t = daily_visits['visits'].corr(daily_visits['temp'])
corr_r = daily_visits['visits'].corr(daily_visits['rain'])
print(f"Correlation Visits–Temp: {corr_t:.2f}")
print(f"Correlation Visits–Rain: {corr_r:.2f}")

# LOESS-линию на temp vs visits
df_clean = daily_visits[['temp', 'visits']].dropna()
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
df_rain = daily_visits[['rain', 'visits']].dropna()
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
daily_visits['temp_bin'] = pd.cut(daily_visits['temp'], 
                                    bins=range(0, 36, 5), right=False)
temp_means = daily_visits.groupby('temp_bin')['visits'].mean().reset_index()
plt.figure(figsize=(8,4))
sns.barplot(x='temp_bin', y='visits', data=temp_means, palette='Blues')
plt.xticks(rotation=45)
plt.title('Average Visits by Temperature Bin')
plt.xlabel('Temperature Bin (°C)')
plt.ylabel('Average Visits per Day')
plt.grid(alpha=0.3, axis='y')
plt.show()

# Barplot средних визитов по бинам осадков
daily_visits['rain_bin'] = pd.cut(daily_visits['rain'], 
                                    bins=[0,1,5,15, daily_visits['rain'].max()],
                                    labels=['0–1','1–5','5–15','15+'], right=False)
rain_means = daily_visits.groupby('rain_bin')['visits'].mean().reset_index()
plt.figure(figsize=(8,4))
sns.barplot(x='rain_bin', y='visits', data=rain_means, palette='Blues_r')
plt.title('Average Visits by Rainfall Bin')
plt.xlabel('Rainfall Bin (mm)')
plt.ylabel('Average Visits per Day')
plt.grid(alpha=0.3, axis='y')
plt.show()


# %% ANOMALY DETECTIOOOOOON START!!!!

print(daily_visits.columns)
print(daily_visits.head())

# Динамическое обнаружение аномалий объёма визитов относительно текущего тренда
# Перед Isolation Forest и LSTM:
WINDOW = 30
K = 2
rolling = daily_visits['visits'].rolling(window=WINDOW, min_periods=WINDOW//2)
daily_visits['roll_mean'] = rolling.mean()
daily_visits['roll_std']  = rolling.std()

daily_visits['zscore'] = (daily_visits['visits'] - daily_visits['roll_mean']) / daily_visits['roll_std']
daily_visits['high_volume'] = (daily_visits['zscore'] > K).astype(int)
daily_visits['low_volume']  = (daily_visits['zscore'] < -K).astype(int)
daily_visits.drop(columns=['roll_mean', 'roll_std', 'zscore'], inplace=True)


# Константы
N_STEPS       = 14
LSTM_EPOCHS   = 20
THRESHOLD_STD = 3
RF_STATE      = 42

def fit_lstm_autoencoder(X: np.ndarray, epochs: int = LSTM_EPOCHS) -> Model:
    """
    Строит и обучает LSTM Autoencoder на входных последовательностях X.
    X.shape == (n_samples, N_STEPS, n_features)
    """
    n_steps, n_features = X.shape[1], X.shape[2]
    inp = Input(shape=(n_steps, n_features))
    enc = LSTM(64, activation='relu')(inp)
    dec = RepeatVector(n_steps)(enc)
    dec = LSTM(64, activation='relu', return_sequences=True)(dec)
    out = TimeDistributed(Dense(n_features))(dec)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, X, epochs=epochs, batch_size=32,
              validation_split=0.1, shuffle=True, verbose=0)
    return model

def get_lstm_anomalies(X: np.ndarray, model: Model) -> np.ndarray:
    """
    Возвращает бинарный флаг аномалии для каждой последовательности в X.
    Порог: mean(MSE) + THRESHOLD_STD * std(MSE).
    """
    preds = model.predict(X)
    mse   = np.mean((X - preds)**2, axis=(1,2))
    thresh = mse.mean() + THRESHOLD_STD * mse.std()
    flags  = (mse > thresh).astype(int)
    # Вернуть массив длины original, с учётом сдвига N_STEPS-1
    return np.concatenate([np.zeros(N_STEPS-1, dtype=int), flags])

def extract_anomaly_periods(series: pd.Series) -> list:
    """Преобразует серию 0/1 в список периодов с подряд идущими единицами."""
    series = series[series == 1]
    if series.empty:
        return []

    dates = series.index.to_list()
    periods = []
    start = dates[0]

    for i in range(1, len(dates)):
        if (dates[i] - dates[i-1]).days > 1:
            end = dates[i-1]
            periods.append((start, end))
            start = dates[i]
    periods.append((start, dates[-1]))
    return periods

# === Функция для создания бинарной маски из периодов ===
def mark_periods(series: pd.Series, col_name: str) -> pd.Series:
    """Из 0/1 флагов формирует бинарный вектор, размечающий все дни в аномальных интервалах."""
    periods = extract_anomaly_periods(series)
    bin_col = pd.Series(0, index=series.index)
    for start, end in periods:
        bin_col.loc[start:end] = 1
    return bin_col


def plot_anomalies(df, flag_col: str, title: str):
    """
    Рисует временной ряд визитов и отмечает аномальные точки.
    """
    plt.figure(figsize=(12,4))
    plt.plot(df.index, df['visits'], alpha=0.5, label='Visits')
    anoms = df[df[flag_col] == 1]
    plt.scatter(anoms.index, anoms['visits'], color='red', s=40, label='Anomaly')
    plt.title(title)
    plt.legend(); plt.grid(alpha=0.3); plt.show()

def compare_weather_stats(df, flag_col: str, feature: str):
    """
    T-тест между нормальными и аномальными значениями feature.
    """
    norm = df[df[flag_col]==0][feature]
    ano  = df[df[flag_col]==1][feature]
    t, p = ttest_ind(norm, ano, equal_var=False)
    print(f"{feature}: mean_norm={norm.mean():.1f}, mean_ano={ano.mean():.1f}, p={p:.4f}")

def compare_three_versions(df, cols: list, model_name: str):
    """
    Считает Jaccard, пересечения и различия между парами флагов.
    """
    from itertools import combinations
    import pandas as pd

    rows = []
    for a, b in combinations(cols, 2):
        A, B = df[a], df[b]
        both = int(((A==1)&(B==1)).sum())
        union= int(((A==1)|(B==1)).sum())
        jacc = both/union if union else 0
        rows.append({
            'Model': model_name,
            'A': a, 'B': b,
            'Both': both,
            'Union': union,
            'Jaccard': round(jacc,2),
            'Disagreements': int((A!=B).sum())
        })
    return pd.DataFrame(rows)

# --- Пример применения ---
# Предполагается, что daily_visits уже содержит колонки ['visits','temp','rain','high_volume','low_volume']

# Isolation Forest сразу в 0/1
for name, feats in [
    ('IF_visits',  ['visits']),
    ('IF_vol',     ['visits','high_volume','low_volume']),
    ('IF_weather', ['visits','temp','rain'])
]:
    model = IsolationForest(random_state=RF_STATE)
    daily_visits[name] = (model.fit_predict(daily_visits[feats]) == -1).astype(int)

# LSTM Autoencoder
    
# -- только visits
vals = daily_visits['visits'].values.reshape(-1,1)
scaled = MinMaxScaler().fit_transform(vals)
X_seq  = np.array([scaled[i:i+N_STEPS] for i in range(len(scaled)-N_STEPS+1)])
model  = fit_lstm_autoencoder(X_seq)
daily_visits['LSTM_visits'] = get_lstm_anomalies(X_seq, model)

# -- +volume
fv     = daily_visits[['visits','high_volume','low_volume']].values
scaled = MinMaxScaler().fit_transform(fv)
X_seq2 = np.array([scaled[i:i+N_STEPS] for i in range(len(scaled)-N_STEPS+1)])
model2 = fit_lstm_autoencoder(X_seq2)
daily_visits['LSTM_vol'] = get_lstm_anomalies(X_seq2, model2)

# -- +weather
features = ['visits', 'high_volume', 'low_volume', 'temp', 'rain']
df_lstm = daily_visits.dropna(subset=features).copy()
data_vals = df_lstm[features].values
scaled   = MinMaxScaler().fit_transform(data_vals)
seq      = np.array([scaled[i:i+N_STEPS] for i in range(len(scaled)-N_STEPS+1)])
model3 = fit_lstm_autoencoder(seq, epochs=LSTM_EPOCHS)
flags3 = get_lstm_anomalies(seq, model3)  # возвращает уже с паддингом N_STEPS-1
df_lstm['LSTM_weather'] = flags3
daily_visits['LSTM_weather'] = 0
daily_visits.loc[df_lstm.index, 'LSTM_weather'] = df_lstm['LSTM_weather']

lstm_periods = extract_anomaly_periods(daily_visits['LSTM_vol'])

# Создаём бинарные колонки
daily_visits['LSTM_visits_bin']   = mark_periods(daily_visits['LSTM_visits'], 'LSTM_visits')
daily_visits['LSTM_vol_bin']      = mark_periods(daily_visits['LSTM_vol'], 'LSTM_vol')
daily_visits['LSTM_weather_bin']  = mark_periods(daily_visits['LSTM_weather'], 'LSTM_weather')

jacc_df = compare_three_versions(
    daily_visits,
    ['IF_visits', 'LSTM_visits_bin', 'LSTM_vol_bin', 'LSTM_weather_bin'],
    model_name="Final Models with Periods"
)
print(jacc_df)


# %% Визуализация различий 
from upsetplot import from_memberships, UpSet


# --- Настройка ---
anomaly_models = [
    'IF_visits','IF_vol','IF_weather',
    'LSTM_visits_bin','LSTM_vol_bin','LSTM_weather_bin'
]

# Если у вас есть колонки с raw‐скорингами:
if_score_col = 'IF_score'           # e.g. output decision_function
lstm_error_col = 'LSTM_recon_error' # e.g. MSE по окну

# Названия признаков для IF (в порядке, как подавали в модель)
if_features = ['visits','vol','weather','temp','rain']

# === Функции визуализации ===

def plot_upset_from_anomalies(df, models):
    sets = []
    for idx, row in df[models].iterrows():
        active = [m for m in models if row[m] == 1]
        if active:
            sets.append(tuple(sorted(active)))
    data = from_memberships(sets)
    plt.figure(figsize=(10,5))
    UpSet(data, subset_size='count', show_percentages=True).plot()
    plt.suptitle("Anomaly Overlaps Between Models", fontsize=16)
    plt.show()

def get_period_lengths(series):
    periods = extract_anomaly_periods(series)
    return [ (end - start).days + 1 for start, end in periods ]

def plot_anomaly_period_lengths(df):
    data = {
        'LSTM_visits_bin': get_period_lengths(df['LSTM_visits']),
        'LSTM_vol_bin': get_period_lengths(df['LSTM_vol']),
        'LSTM_weather_bin': get_period_lengths(df['LSTM_weather']),
        'IF_visits': get_period_lengths(df['IF_visits']),
        'IF_vol': get_period_lengths(df['IF_vol']),
        'IF_weather': get_period_lengths(df['IF_weather'])

    }
    df_lengths = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in data.items()]))
    df_lengths = df_lengths.melt(var_name='Model', value_name='Duration (days)')

    plt.figure(figsize=(10,5))
    sns.boxplot(data=df_lengths, x='Model', y='Duration (days)', palette='Set3')
    plt.title("Distribution of Anomaly Period Lengths")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

def plot_anomaly_timeline(df, models):
    import matplotlib.dates as mdates
    fig, ax = plt.subplots(figsize=(15, len(models) * 0.6))
    for i, model in enumerate(models):
        periods = extract_anomaly_periods(df[model])
        for start, end in periods:
            ax.barh(model, (end - start).days + 1, left=start, height=0.4)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.title("Anomaly Periods by Model")
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout(); plt.show()

def plot_time_series_with_anomalies(df, models):
    plt.figure(figsize=(15,5))
    plt.plot(df.index, df['visits'], alpha=0.6, label='Visits', color='black')
    colors = ['red', 'blue', 'green', 'orange']
    for i, model in enumerate(models):
        mask = df[model] == 1
        plt.fill_between(df.index, 0, df['visits'], where=mask, alpha=0.2, color=colors[i % len(colors)], label=model)
    plt.title('Daily Visits with Anomaly Periods')
    plt.xlabel('Date'); plt.ylabel('Visits')
    plt.legend(ncol=2); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

def plot_anomaly_counts(df, models):
    counts = df[models].sum()
    plt.figure(figsize=(10,5))
    sns.barplot(x=counts.index, y=counts.values, palette='Set2')
    plt.title('Anomaly Count by Model')
    plt.ylabel('Number of Anomalies')
    plt.xticks(rotation=45); plt.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.show()

def plot_visit_distribution(df, model):
    plt.figure(figsize=(10,5))
    sns.histplot(df[df[model]==0]['visits'], stat='density', kde=True, label='Normal')
    sns.histplot(df[df[model]==1]['visits'], stat='density', kde=True, label='Anomaly')
    plt.title(f'Visits: Normal vs {model}')
    plt.xlabel('Visits'); plt.ylabel('Density')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

def plot_monthly_heatmap(df, model):
    data = (df
        .assign(year=df.index.year, month=df.index.month)
        .groupby(['year','month'])[model]
        .sum()
        .unstack(fill_value=0)
    )
    plt.figure(figsize=(12,6))
    sns.heatmap(data, cmap='Reds', annot=True, fmt='d')
    plt.title(f'Monthly Anomaly Counts ({model})')
    plt.xlabel('Month'); plt.ylabel('Year')
    plt.tight_layout(); plt.show()

def plot_jaccard_heatmap(df, models):
    mat = pd.DataFrame(index=models, columns=models, dtype=float)
    for a in models:
        for b in models:
            mat.loc[a,b] = jaccard_score(df[a], df[b])
    plt.figure(figsize=(8,6))
    sns.heatmap(mat, annot=True, cmap='YlGnBu', vmin=0, vmax=1)
    plt.title('Jaccard Similarity between Models')
    plt.tight_layout(); plt.show()

def plot_rolling_rate(df, models, window=30):
    plt.figure(figsize=(15,5))
    for m in models:
        rate = df[m].rolling(window).mean()
        plt.plot(rate.index, rate, label=m)
    plt.title(f'{window}-Day Rolling Anomaly Rate')
    plt.xlabel('Date'); plt.ylabel('Proportion')
    plt.legend(ncol=2, fontsize='small'); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

def plot_feature_correlation(df, features):
    corr = df[features].corr()
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix'); plt.tight_layout(); plt.show()

def plot_seasonal_decomposition(df, column, period=365):
    decomp = seasonal_decompose(df[column], model='additive', period=period)
    fig = decomp.plot()
    fig.set_size_inches(12,9)
    plt.suptitle(f'Seasonal Decomposition ({column})', fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95]); plt.show()
    resid = decomp.resid.dropna()
    # Гистограмма остатков
    plt.figure(figsize=(10,4))
    sns.histplot(resid, stat='density', kde=True)
    plt.title('Histogram of Residuals'); plt.tight_layout(); plt.show()
    # QQ-plot
    plt.figure(figsize=(6,6))
    qqplot(resid, line='s', ax=plt.gca())
    plt.title('QQ-Plot of Residuals'); plt.tight_layout(); plt.show()

def plot_score_distribution(df, score_col, threshold=None):
    plt.figure(figsize=(10,5))
    sns.histplot(df[score_col], stat='density', kde=True)
    if threshold is not None:
        plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
    plt.title(f'Distribution of {score_col}')
    plt.legend(); plt.tight_layout(); plt.show()

def plot_if_feature_importance(if_model, feature_names):
    importances = if_model.feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(8,5))
    sns.barplot(x=fi.values, y=fi.index, palette='Blues_d')
    plt.title('Isolation Forest Feature Importances')
    plt.tight_layout(); plt.show()

def plot_roc_pr(df, score_col, true_col):
    fpr, tpr, _ = roc_curve(df[true_col], df[score_col])
    prec, rec, _ = precision_recall_curve(df[true_col], df[score_col])
    plt.figure(figsize=(12,5))
    # ROC
    plt.subplot(1,2,1)
    plt.plot(fpr, tpr, label=f'AUC={auc(fpr,tpr):.2f}')
    plt.title('ROC Curve'); plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.legend(); plt.grid(alpha=0.3)
    # PR
    plt.subplot(1,2,2)
    plt.plot(rec, prec, label=f'AP={auc(rec,prec):.2f}')
    plt.title('Precision-Recall Curve'); plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

# === Вызовы ===

plot_time_series_with_anomalies(daily_visits, anomaly_models)
plot_anomaly_counts(daily_visits, anomaly_models)

for m in anomaly_models:
    plot_visit_distribution(daily_visits, m)
    plot_monthly_heatmap(daily_visits, m)

plot_jaccard_heatmap(daily_visits, anomaly_models)
plot_rolling_rate(daily_visits, anomaly_models)
plot_feature_correlation(daily_visits, ['visits','temp','rain','high_volume','low_volume'])
plot_seasonal_decomposition(daily_visits, 'visits')

plot_upset_from_anomalies(daily_visits, anomaly_models)
plot_anomaly_period_lengths(daily_visits)
plot_anomaly_timeline(daily_visits, anomaly_models)


## После детекции аномалий (IF/LSTM) мы не всегда знаем причину — это может быть COVID, фестиваль, климат и т.д.
# Чтобы интерпретировать найденные аномалии, мы подключаем LLM — GPT-4 и Mixtral.
# Они получают одинаковые промпты и объясняют: ПОЧЕМУ день или период мог быть аномальным.


 # %% AI Initialization

load_dotenv(dotenv_path="aikey.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY не найден в файле aikey.env")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

def ask_paid_ai(prompt: str, model: str = "gpt-4") -> str:
    """Запрос к платному OpenAI API."""
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300
    )
    return resp.choices[0].message.content.strip()

# Ленивый загрузчик HF-пайплайна для бесплатной модели
FREE_MODEL_NAME = "bigscience/bloom-560m"
_hf_pipeline = None

def ask_free_ai(prompt: str, max_tokens: int = 80) -> str:
    """Запрос к локальной бесплатной модели HF."""
    global _hf_pipeline
    if _hf_pipeline is None:
        print("Загрузка HF-модели…")
        tokenizer = AutoTokenizer.from_pretrained(FREE_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(FREE_MODEL_NAME, device_map="auto")
        _hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
        print("HF-модель готова.")
    
    output = _hf_pipeline(prompt, max_new_tokens=max_tokens, do_sample=False)[0]["generated_text"]
    return output.replace(prompt, "").strip()

def normalize_response(resp: str) -> str:
    """
    Нормализует ответы AI:
    - Пустые или содержащие маркеры «нет данных» приводятся к 'no data'
    """
    if not resp or not resp.strip():
        return "no data"
    lower = resp.lower()
    negative_markers = ["no events found", "no known events", "nothing special",
                        "нет данных", "нет событий", "no data"]
    for marker in negative_markers:
        if marker in lower:
            return "no data"
    return resp.strip()

print("✅ Клиенты AI инициализированы.")



# %% начало получения причин

from tqdm import tqdm

# Генерация промптов для IF и LSTM 
def generate_prompts(df):
    if_dates = df[df['IF_vol'] == 1].index
    lstm_periods = extract_anomaly_periods(df['LSTM_vol'])
    prompts = []
    for date in if_dates:
        text = f"On {date.strftime('%B %d, %Y')}, a spike or drop in tourism activity was detected in Verona. Were there any events (weather, cultural, political) that could explain this?"
        prompts.append({'model': 'IF_vol', 'date': str(date.date()), 'prompt': text})

    for start, end in lstm_periods:
        text = f"Between {start.strftime('%B %d, %Y')} and {end.strftime('%B %d, %Y')}, a pattern change in tourism activity occurred in Verona. Are there any known events that could explain it?"
        prompts.append({'model': 'LSTM_vol_bin', 'date': f"{start.date()} – {end.date()}", 'prompt': text})
    return prompts

# def run_paid_llm_queries(prompts, autosave_every=5, output_file="llm_progress.csv"):
#     try:
#         df_existing = pd.read_csv(output_file, encoding='utf-8', on_bad_lines='skip')
#         done_prompts = set(df_existing['prompt'])
#         results = df_existing.to_dict('records')
#         print(f"🔁 Продолжение: найдено {len(done_prompts)} уже сохранённых запросов.")
#     except FileNotFoundError:
#         done_prompts = set()
#         results = []

#     for i, p in enumerate(tqdm(prompts, desc="Запросы к GPT (платный)")):
#         if p['prompt'] in done_prompts:
#             continue

#         paid = normalize_response(ask_paid_ai(p['prompt']))

#         results.append({
#             'model': p['model'],
#             'date': p['date'],
#             'prompt': p['prompt'],
#             'paid_response': paid,
#             'free_response': None,
#             'verdict': None
#         })

#         if (len(results) % autosave_every) == 0:
#             pd.DataFrame(results).to_csv(output_file, index=False, encoding='utf-8')
#             print(f"💾 Автосохранено {len(results)} строк...")

#     final_df = pd.DataFrame(results)
#     final_df.to_csv(output_file, index=False, encoding='utf-8')
#     print(f"✅ Финальное сохранение в {output_file}")
#     return final_df




# def fill_free_llm_responses(input_csv="llm_progress.csv", output_csv="llm_completed.csv", autosave_every=10):
#     try:
#         df = pd.read_csv(input_csv)
#     except Exception as e:
#         raise RuntimeError(f"❌ Не удалось загрузить файл {input_csv}: {e}")
    
#     required_columns = {'prompt', 'paid_response', 'free_response'}
#     if not required_columns.issubset(df.columns):
#         raise ValueError(f"❌ Входной CSV должен содержать колонки: {required_columns}")

#     if (df['free_response'].fillna("filled") != "filled").all():
#         print("✅ Все free_response уже заполнены.")
#         df.to_csv(output_csv, index=False, encoding='utf-8')
#         return df

#     updated_rows = []
#     for i, row in tqdm(df.iterrows(), total=len(df), desc="Запросы к BLOOM (бесплатный)"):
#         if pd.notnull(row['free_response']) and str(row['free_response']).lower().strip() != "skipped":
#             updated_rows.append(row)
#             continue

#         prompt = row['prompt']
#         paid = str(row['paid_response']).strip().lower() if pd.notnull(row['paid_response']) else "no data"

#         try:
#             t0 = time.time()
#             free = normalize_response(ask_free_ai(prompt))
#             response_time = round(time.time() - t0, 2)
#         except Exception as e:
#             print(f"⚠️ Ошибка на строке {i}: {e}")
#             free = "no data"
#             response_time = -1

#         if paid == free:
#             verdict = "equal"
#         elif free == "no data":
#             verdict = "paid_only"
#         elif paid == "no data":
#             verdict = "free_only"
#         else:
#             verdict = "different"

#         row['free_response'] = free
#         row['free_response_time'] = response_time
#         row['verdict'] = verdict
#         updated_rows.append(row)

#         if (i + 1) % autosave_every == 0:
#             pd.DataFrame(updated_rows).to_csv(output_csv, index=False, encoding='utf-8')
#             print(f"💾 Автосохранено {i + 1} строк...")

#     df_updated = pd.DataFrame(updated_rows)
#     df_updated.to_csv(output_csv, index=False, encoding='utf-8')
#     print(f"✅ Обновлённый файл сохранён: {output_csv}")
#     return df_updated


def run_dual_llm_queries_safe(prompts, autosave_every=10, output_file="llm_completed.csv"):
    # Пытаемся загрузить существующий файл
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file, encoding='utf-8')
        existing_keys = {(row['model'], row['date'], row['prompt']) for _, row in df_existing.iterrows()}
        results = df_existing.to_dict('records')
        print(f"🔁 Найдено {len(existing_keys)} уже выполненных запросов — продолжаем с места остановки.")
    else:
        existing_keys = set()
        results = []
        print("📄 Файл не найден. Создаём новый с нуля.")

    # Проходим по новым промптам
    for i, p in enumerate(tqdm(prompts, desc="GPT + BLOOM (safe)")):
        key = (p['model'], p['date'], p['prompt'])
        if key in existing_keys:
            continue  # уже было

        # Запрос к GPT (платный)
        try:
            paid_start = time.time()
            paid = normalize_response(ask_paid_ai(p['prompt']))
            paid_time = round(time.time() - paid_start, 2)
        except Exception as e:
            print(f"❌ GPT fail: {e}")
            paid = "no data"
            paid_time = -1

        # Запрос к BLOOM (бесплатный)
        try:
            free_start = time.time()
            free = normalize_response(ask_free_ai(p['prompt']))
            free_time = round(time.time() - free_start, 2)
        except Exception as e:
            print(f"❌ BLOOM fail: {e}")
            free = "no data"
            free_time = -1

        # Вердикт
        if paid == free:
            verdict = "equal"
        elif free == "no data":
            verdict = "paid_only"
        elif paid == "no data":
            verdict = "free_only"
        else:
            verdict = "different"

        # Запись результата
        results.append({
            'model': p['model'],
            'date': p['date'],
            'prompt': p['prompt'],
            'paid_response': paid,
            'free_response': free,
            'paid_response_time': paid_time,
            'free_response_time': free_time,
            'verdict': verdict
        })

        # Автосейв
        if (len(results) % autosave_every == 0):
            pd.DataFrame(results).to_csv(output_file, index=False, encoding='utf-8')
            print(f"💾 Автосохранено {len(results)} строк...")

    # Финальное сохранение
    df_final = pd.DataFrame(results)
    df_final.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✅ Финальный результат сохранён в {output_file}")
    return df_final


prompts = generate_prompts(daily_visits)
df_done = run_dual_llm_queries_safe(prompts, output_file="llm_completed.csv")





# %% визуализация и сравнение ИИ

from wordcloud import WordCloud


# Группировка по вердиктам
verdict_counts = df_done['verdict'].value_counts()

plt.figure(figsize=(8, 5))
sns.barplot(x=verdict_counts.index, y=verdict_counts.values)
plt.title("Comparison Verdicts: Paid vs Free LLM")
plt.ylabel("Number of Cases")
plt.xlabel("Verdict Type")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Группировка по модели и вердикту
model_verdict = df_done.groupby(['model', 'verdict']).size().unstack().fillna(0)

model_verdict.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title("Verdict Breakdown by Anomaly Detection Model")
plt.ylabel("Number of Cases")
plt.xlabel("Model")
plt.legend(title="Verdict", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Boxplot времени ответа (если есть данные)
if 'paid_response_time' in df_done.columns and 'free_response_time' in df_done.columns:
    df_melted = df_done.melt(value_vars=['paid_response_time', 'free_response_time'],
                            var_name='Model', value_name='Response Time')
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_melted, x='Model', y='Response Time')
    plt.title("Response Time Comparison")
    plt.ylabel("Time (seconds)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


#WordCloud
diffs = df_done[df_done['verdict'] == 'different']
text_paid = ' '.join(diffs['paid_response'].dropna())
text_free = ' '.join(diffs['free_response'].dropna())
wc_paid = WordCloud(width=800, height=400, background_color='white').generate(text_paid)
wc_free = WordCloud(width=800, height=400, background_color='white').generate(text_free)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.imshow(wc_paid, interpolation='bilinear')
plt.title("Words in GPT (Paid) — Different Verdicts")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(wc_free, interpolation='bilinear')
plt.title("Words in BLOOM (Free) — Different Verdicts")
plt.axis('off')
plt.tight_layout()
plt.show()

summary = pd.crosstab(
    df_done['paid_response'] == "no data",
    df_done['free_response'] == "no data",
    rownames=['GPT no data'], colnames=['BLOOM no data']
)

print(summary)


# %%
