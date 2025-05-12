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
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import ttest_ind

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

# 1) Isolation Forest сразу в 0/1
for name, feats in [
    ('IF_visits',  ['visits']),
    ('IF_vol',     ['visits','high_volume','low_volume']),
    ('IF_weather', ['visits','temp','rain'])
]:
    model = IsolationForest(random_state=RF_STATE)
    daily_visits[name] = (model.fit_predict(daily_visits[feats]) == -1).astype(int)

# 2) LSTM Autoencoder
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


# 3) Визуализация и сравнение
plot_anomalies(daily_visits, 'LSTM_weather', title='LSTM+Weather Anomalies')
compare_weather_stats(daily_visits, 'IF_weather', 'temp')
df_cmp = compare_three_versions(daily_visits,
    ['IF_visits','IF_vol','IF_weather','LSTM_visits','LSTM_vol','LSTM_weather'],
    model_name='AnomalyModels')
print(df_cmp)


# %% Визуализация различий 
print(daily_visits.columns)
print(daily_visits.head())

# Константы
ANOMALY_COLS = [
    'IF_visits','IF_vol','IF_weather',
    'LSTM_visits','LSTM_vol','LSTM_weather'
]

def plot_time_series(df, window=7):
    plt.figure(figsize=(15, 5))
    df['visits'].plot(alpha=0.5, label='Visits')
    df['visits'].rolling(window).mean().plot(
        linewidth=2, label=f'{window}-day MA'
    )
    plt.title(f'Daily Visits with {window}-day Rolling Mean')
    plt.xlabel('Date'); plt.ylabel('Visits')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.show()

def plot_anomaly_counts(df, cols):
    counts = [df[c].sum() for c in cols]
    plt.figure(figsize=(10,5))
    sns.barplot(x=cols, y=counts, palette='Set2')
    plt.title('Anomaly Count by Model')
    plt.ylabel('Number of Anomalies')
    plt.xticks(rotation=45); plt.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.show()

def plot_visit_distribution(df, flag_col):
    plt.figure(figsize=(10,5))
    sns.histplot(df[df[flag_col]==0]['visits'], label='Normal', kde=True)
    sns.histplot(df[df[flag_col]==1]['visits'], label='Anomaly', kde=True)
    plt.title(f'Visit Distribution: Normal vs {flag_col}')
    plt.xlabel('Visits'); plt.ylabel('Frequency')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

def plot_boxplot_feature(df, flag_col, feature, labels=('Normal','Anomaly')):
    tmp = df.dropna(subset=[feature, flag_col])
    tmp[flag_col] = tmp[flag_col].map({0:labels[0],1:labels[1]})
    plt.figure(figsize=(8,5))
    sns.boxplot(x=flag_col, y=feature, data=tmp)
    plt.title(f'{feature} on Normal vs Anomaly Days ({flag_col})')
    plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

def plot_monthly_heatmap(df, flag_col):
    # рассчитываем год и месяц на лету
    idx = df.index
    data = df.assign(
        year=idx.year,
        month=idx.month
    ).groupby(['year','month'])[flag_col]\
     .sum().unstack(fill_value=0)
    plt.figure(figsize=(12,6))
    sns.heatmap(data, cmap='Reds', annot=True, fmt='d')
    plt.title(f'Monthly {flag_col} Anomalies (Heatmap)')
    plt.xlabel('Month'); plt.ylabel('Year')
    plt.tight_layout(); plt.show()

def plot_calendar_heatmap(df, flag_col):
    tmp = df.assign(
        month=df.index.month,
        weekday=df.index.weekday
    ).pivot_table(
        values=flag_col, index='month',
        columns='weekday', aggfunc='sum', fill_value=0
    )
    plt.figure(figsize=(10,6))
    sns.heatmap(tmp, cmap='Reds', annot=True, fmt='d')
    plt.title(f'{flag_col} by Weekday and Month')
    plt.xlabel('Weekday (0=Mon)')
    plt.ylabel('Month')
    plt.tight_layout(); plt.show()

def plot_scatter(df, x, y, hue_flag):
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x=x, y=y, hue=df[hue_flag].astype(int), alpha=0.7)
    plt.title(f'{y} vs {x} (colored by {hue_flag})')
    plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

def plot_jaccard_heatmap(df, cols):
    # строим матрицу Jaccard
    mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            if a==b:
                mat.loc[a,b] = 1.0
            else:
                mat.loc[a,b] = jaccard_score(df[a], df[b])
    plt.figure(figsize=(8,6))
    sns.heatmap(mat, annot=True, cmap='YlGnBu', vmin=0, vmax=1)
    plt.title('Jaccard Similarity between Models')
    plt.tight_layout(); plt.show()

# --- Вызовы функций ---

plot_time_series(daily_visits, window=7)
plot_anomaly_counts(daily_visits, ANOMALY_COLS)
plot_visit_distribution(daily_visits, 'LSTM_visits')
plot_boxplot_feature(daily_visits, 'LSTM_weather', 'temp')
plot_monthly_heatmap(daily_visits, 'LSTM_weather')

# пример для IF_weather vs rain
plot_boxplot_feature(daily_visits, 'anomaly_if_weather', 'rain')

# сравнение моделей
plot_jaccard_heatmap(daily_visits, ANOMALY_COLS)
plot_calendar_heatmap(daily_visits, 'LSTM_weather')
plot_scatter(daily_visits, 'temp', 'visits', 'anomaly_if_weather')


# Параметр period=365, т.к. данные – ежедневные за несколько лет
decomp = seasonal_decompose(daily_visits['visits'], model='additive', period=365)
fig = decomp.plot()
fig.set_size_inches(12, 9)
plt.suptitle('Seasonal Decomposition of Daily Visits', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


plt.figure(figsize=(12, 5))
for col in ['IF_visits','IF_vol','IF_weather','LSTM_visits','LSTM_vol','LSTM_weather']:
    rolling_rate = daily_visits[col].rolling(window=30).mean()
    plt.plot(rolling_rate.index, rolling_rate, label=col)
plt.title('30-Day Rolling Anomaly Rate by Model')
plt.xlabel('Date')
plt.ylabel('Proportion of Anomalies')
plt.legend(loc='upper right', fontsize='small')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


features = ['visits', 'temp', 'rain', 'high_volume', 'low_volume']
corr = daily_visits[features].corr()
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(corr, cmap='coolwarm')
fig.colorbar(cax)
ax.set_xticks(range(len(features)))
ax.set_yticks(range(len(features)))
ax.set_xticklabels(features, rotation=45, ha='left')
ax.set_yticklabels(features)
for (i, j), val in np.ndenumerate(corr.values):
    ax.text(j, i, f"{val:.2f}", ha='center', va='center',
            color='white' if abs(val) > 0.5 else 'black')

plt.title('Feature Correlation Matrix', pad=20)
plt.tight_layout()
plt.show()


# 1. Violin-plot сезонного компонента по месяцам
decomp = seasonal_decompose(daily_visits['visits'], model='additive', period=365)
seasonal = decomp.seasonal.to_frame(name='seasonal')
seasonal['month'] = seasonal.index.month

plt.figure(figsize=(12, 6))
sns.violinplot(x='month', y='seasonal', data=seasonal, palette='muted')
plt.title('Distribution of Seasonal Component by Month')
plt.xlabel('Month')
plt.ylabel('Seasonal Value')
plt.tight_layout()
plt.show()

# 2. Heatmap аномалий IF_models по месяцу и дню недели
for col in ['IF_visits','IF_weather']:
    tmp = daily_visits.copy()
    tmp['month']   = tmp.index.month
    tmp['weekday'] = tmp.index.weekday
    heat = tmp.pivot_table(values=col, index='month', columns='weekday', aggfunc='sum', fill_value=0)
    plt.figure(figsize=(8,5))
    sns.heatmap(heat, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Heatmap of {col} by Month & Weekday')
    plt.xlabel('Weekday (0=Mon)'); plt.ylabel('Month')
    plt.tight_layout()
    plt.show()

# 3. Гистограмма и QQ-plot остатков decomposition
resid = decomp.resid.dropna()

plt.figure(figsize=(10,4))
sns.histplot(resid, bins=30, kde=True, color='gray')
plt.title('Histogram of Decomposition Residuals')
plt.xlabel('Residual'); plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))
qqplot(resid, line='s', ax=plt.gca())
plt.title('QQ-Plot of Decomposition Residuals')
plt.tight_layout()
plt.show()

# 4. Кластеризация соседних аномалий IF_weather → «события» и scatter duration vs avg temp
df_evt = daily_visits[['IF_weather','temp']].copy()
df_evt['grp'] = (df_evt['IF_weather'] != df_evt['IF_weather'].shift()).cumsum()
events = (
    df_evt[df_evt['IF_weather']==1]
    .groupby('grp')
    .agg(duration=('IF_weather','size'),
         avg_temp=('temp','mean'))
    .reset_index(drop=True)
)

plt.figure(figsize=(8,5))
plt.scatter(events['duration'], events['avg_temp'], s=50, alpha=0.7)
plt.title('Anomaly Event Duration vs Avg Temperature (IF_weather)')
plt.xlabel('Duration (days)')
plt.ylabel('Average Temperature (°C)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


## После детекции аномалий (IF/LSTM) мы не всегда знаем причину — это может быть COVID, фестиваль, климат и т.д.
# Чтобы интерпретировать найденные аномалии, мы подключаем LLM — GPT-4 и Mixtral.
# Они получают одинаковые промпты и объясняют: ПОЧЕМУ день или период мог быть аномальным.
 # %% AI Initialization

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
