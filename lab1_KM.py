print('Пункт 1')
import pandas as pd
import numpy as np
from math import isnan

try:
    from scipy import stats
except Exception as e:
    stats = None
    print("Внимание: scipy недоступен — критическое значение вычислить нельзя.")

input_path = "RC_F01_09_2024_T01_09_2025.csv"
output_path = "RC_F01_09_2024_T01_09_2025_grubbs_imputed.csv"

df = pd.read_csv(input_path)
col = df.select_dtypes(include=[np.number]).columns[0]
series = df[col].astype(float).copy()

alpha = 0.05
outlier_entries = []

def grubbs_critical_value(N, alpha):
    if stats is None:
        raise RuntimeError("scipy.stats недоступен — критическое значение вычислить нельзя.")
    t = stats.t.ppf(1 - alpha/(2*N), N-2)
    Gcrit = (N-1)/np.sqrt(N) * np.sqrt(t*t / (N-2 + t*t))
    return Gcrit

series_work = series.copy()
while True:
    non_na = series_work.dropna()
    N = non_na.shape[0]
    if N < 3:
        break
    mean = non_na.mean()
    std = non_na.std(ddof=1)
    deviations = (non_na - mean).abs()
    max_idx = deviations.idxmax()
    max_val = series_work.loc[max_idx]
    G = deviations.loc[max_idx] / std if std > 0 else np.inf
    try:
        Gcrit = grubbs_critical_value(N, alpha)
    except Exception as e:
        print("Не удалось вычислить Gcrit:", e)
        break
    if G > Gcrit:
        outlier_entries.append({"index": int(max_idx), "value": float(max_val), "G": float(G), "Gcrit": float(Gcrit), "N": int(N)})
        series_work.loc[max_idx] = np.nan
    else:
        break

# Impute with median
median = series_work.dropna().median() if series_work.dropna().shape[0] > 0 else 0.0
series_imputed = series_work.fillna(median)
df[col] = series_imputed
df.to_csv(output_path, index=False)

print(f"Столбец: {col}")
print("Количество обнаруженных выбросов:", len(outlier_entries))
if len(outlier_entries) > 0:
    display(pd.DataFrame(outlier_entries))
else:
    print("Выбросы не найдены по критерию Граббса (alpha=0.05).")

print("\nМедиана, использованная для восстановления:", float(median))
print("\nСсылка на очищенный файл:", output_path)

print('Пункт 2')
import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox

# Пример: случайный белый шум
np.random.seed(42)
data = np.random.normal(0, 1, 200)  # ряд длиной 200

# Тест Бокса–Пирса (lag=10)
result = acorr_ljungbox(data, lags=[10], boxpierce=True)

print("Статистика Бокса–Пирса:", result['bp_stat'].values[0])
print("p-value:", result['bp_pvalue'].values[0])

if result['bp_pvalue'].values[0] < 0.05:
    print("Ряд не является белым шумом (есть автокорреляция).")
else:
    print("Ряд можно считать белым шумом.")

print('Пункт 3')
# Повторный запуск: применю фильтр Бесселя (или запасной метод, если scipy недоступен).
import pandas as pd
import numpy as np
import os

input_path = "RC_F01_09_2024_T01_09_2025.csv"
output_path = "RC_F01_09_2024_T01_09_2025_bessel_filtered.csv"

df = pd.read_csv(input_path)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("Файл:", input_path)
print("Форма:", df.shape)
print("Найденные числовые столбцы:", numeric_cols)

# Попробуем импортировать scipy.signal (для фильтра Бесселя)
use_scipy = True
try:
    from scipy import signal
except Exception as e:
    use_scipy = False
    print("Внимание: scipy.signal недоступен — фильтр Бесселя не может быть применён через scipy:", e)

df_clean = df.copy()

if len(numeric_cols) == 0:
    print("Нет числовых столбцов для фильтрации. Ничего не меняю.")
else:
    if not use_scipy:
        # Если scipy недоступен — применим простую медианную фильтрацию как запасной вариант
        print("Использую запасной метод: скользящая медиана (окно=5).")
        window = 5
        for col in numeric_cols:
            series = df[col].astype(float).copy()
            filtered = series.rolling(window=window, center=True, min_periods=1).median()
            df_clean[col] = filtered
    else:
        order = 4
        cutoff = 0.1  # нормированная частота (0..1), где 1 соответствует Nyquist = fs/2
        for col in numeric_cols:
            series = df[col].astype(float).copy().to_numpy()
            N = series.size
            if N < (3 * (order + 1)):
                print(f"Столбец {col}: слишком короткий ряд (N={N}), применяю скользящую среднюю (window=3).")
                filt = pd.Series(series).rolling(window=3, center=True, min_periods=1).mean().to_numpy()
                df_clean[col] = filt
            else:
                b, a = signal.bessel(order, cutoff, btype='low', analog=False, output='ba')
                try:
                    filtered = signal.filtfilt(b, a, series, method="pad")
                except Exception as e:
                    print(f"filtfilt не сработал для столбца {col}: {e}. Использую lfilter (фазовый сдвиг).")
                    filtered = signal.lfilter(b, a, series)
                df_clean[col] = filtered

# Сохраним результат
df_clean.to_csv(output_path, index=False)
print("\nОчищённый файл сохранён как:", output_path)

# Показать первые 5 строк до и после для числовых столбцов
if len(numeric_cols) > 0:
    compare_df = pd.concat([df[numeric_cols].head().add_prefix("orig_"),
                            df_clean[numeric_cols].head().add_prefix("filt_")], axis=1)

    # Построим график сравнения для первого числового столбца
    import matplotlib.pyplot as plt
    col0 = numeric_cols[0]
    plt.figure(figsize=(8,4))
    plt.plot(df[col0].astype(float).values, label="original")
    plt.plot(df_clean[col0].astype(float).values, label="bessel_filtered")
    plt.title(f"Сравнение: {col0}")
    plt.xlabel("индекс")
    plt.ylabel("значение")
    plt.legend()
    plt.tight_layout()
    plt.show()

print('Пункт 4')
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

def check_stationarity(series, name="ряд"):
    print(f"\n===== Проверка стационарности для {name} =====")
    
    # ADF
    adf_result = adfuller(series, autolag="AIC")
    print("ADF-тест:")
    print(f"  Статистика: {adf_result[0]:.4f}")
    print(f"  p-value: {adf_result[1]:.4f}")
    print("  Критические значения:", adf_result[4])
    
    # KPSS
    try:
        kpss_result = kpss(series, regression="c", nlags="auto")
        print("\nKPSS-тест:")
        print(f"  Статистика: {kpss_result[0]:.4f}")
        print(f"  p-value: {kpss_result[1]:.4f}")
        print("  Критические значения:", kpss_result[3])
    except Exception as e:
        print("\nОшибка KPSS:", e)

# Пути к файлам
orig_path = "RC_F01_09_2024_T01_09_2025.csv"
grubbs_path = "RC_F01_09_2024_T01_09_2025_grubbs_imputed.csv"
bessel_path = "RC_F01_09_2024_T01_09_2025_bessel_filtered.csv"

# Загружаем исходный ряд
orig = pd.read_csv(orig_path)
col = orig.select_dtypes(include="number").columns[0]
orig_series = orig[col].dropna().astype(float)

# После удаления выбросов (Grubbs)
grubbs = pd.read_csv(grubbs_path)
grubbs_series = grubbs[col].dropna().astype(float)

# После фильтрации (Bessel)
bessel = pd.read_csv(bessel_path)
bessel_series = bessel[col].dropna().astype(float)

# Проверим стационарность для трёх случаев
check_stationarity(orig_series, "Исходный ряд")
check_stationarity(grubbs_series, "После удаления выбросов (Grubbs)")
check_stationarity(bessel_series, "После фильтрации шума (Bessel)")

print('Пункт 5')
import pandas as pd
import numpy as np
from scipy.stats import binomtest

def cox_stuart_test(series):
    series = np.array(series)
    n = len(series)
    half = n // 2
    
    first_half = series[:half]
    second_half = series[-half:]
    
    diffs = np.sign(second_half - first_half)
    diffs = diffs[diffs != 0]  # убираем равные
    
    n_pos = np.sum(diffs > 0)
    n_neg = np.sum(diffs < 0)
    
    n_total = n_pos + n_neg
    if n_total == 0:
        return {"n_pos": n_pos, "n_neg": n_neg, "p-value": 1.0, "trend": "нет различий (равные значения)"}
    
    test_result = binomtest(min(n_pos, n_neg), n=n_total, p=0.5, alternative="greater")
    
    return {
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "p-value": test_result.pvalue,
        "trend": "есть тренд" if test_result.pvalue < 0.05 else "тренд не выявлен"
    }

# Пути к файлам
orig_path = "RC_F01_09_2024_T01_09_2025.csv"
grubbs_path = "RC_F01_09_2024_T01_09_2025_grubbs_imputed.csv"
bessel_path = "RC_F01_09_2024_T01_09_2025_bessel_filtered.csv"

# Загружаем ряды
orig = pd.read_csv(orig_path)
col = orig.select_dtypes(include="number").columns[0]
orig_series = orig[col].dropna().astype(float)

grubbs = pd.read_csv(grubbs_path)
grubbs_series = grubbs[col].dropna().astype(float)

bessel = pd.read_csv(bessel_path)
bessel_series = bessel[col].dropna().astype(float)

# Проверка методом Кокса–Стюарта
results = {
    "Исходный ряд": cox_stuart_test(orig_series),
    "После удаления выбросов (Grubbs)": cox_stuart_test(grubbs_series),
    "После фильтрации шума (Bessel)": cox_stuart_test(bessel_series)
}

print(results)