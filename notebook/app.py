# Ensure this is run in a cell in Colab to create/overwrite the app.py file

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np

# --- Configuration and Global Variables ---
DATA_PATH = 'data' # Make sure this path is correct
DEFAULT_SEGMENT = 'Global'
FORECAST_HORIZON = 7  # Горизонт прогнозирования (важно для отображения таблицы)
MODEL_PRED_COLUMNS = {
    'Baseline': 'baseline_pred',
    'Prophet': 'prophet_pred', # Имя модели, как оно используется в mape_results и для отображения
    'SARIMA': 'sarima_pred',
    'LightGBM': 'lgbm_pred',
    'XGBoost': 'xgb_pred'
}
# Отображаемые имена для легенды графика и таблицы MAPE (могут отличаться от ключей MODEL_PRED_COLUMNS, если нужно)
# В данном случае они совпадают с ключами MODEL_PRED_COLUMNS
MODEL_DISPLAY_NAMES = {
    'Baseline': 'Baseline',
    'Prophet': 'Prophet (Optuna)', # Если в mape_results.csv модель называется так
    'SARIMA': 'SARIMA',
    'LightGBM': 'LightGBM',
    'XGBoost': 'XGBoost'
}

MODEL_COLORS = {
    'Actual': 'black',
    'Baseline': 'grey',
    'Prophet (Optuna)': 'blue', # Используйте отображаемое имя, если оно другое
    'SARIMA': 'green',
    'LightGBM': 'orange',
    'XGBoost': 'purple',
    # Confidence/Prediction Intervals Colors
    'SARIMA Lower': 'rgba(0,128,0,0.2)', # Цвет для заливки интервала SARIMA
    'SARIMA Upper': 'rgba(0,128,0,0.2)'  # Цвет для заливки интервала SARIMA
}
MODEL_INTERVAL_COLUMNS = {
    'SARIMA': {'lower': 'sarima_lower', 'upper': 'sarima_upper'},
    # Добавьте сюда другие модели с интервалами, если они есть
    # 'Prophet': {'lower': 'prophet_lower', 'upper': 'prophet_upper'}, # Пример
}


# --- Data Loading Functions (with caching) ---
@st.cache_data
def load_mape_results():
    """Loads MAPE results from CSV."""
    mape_file = os.path.join(DATA_PATH, "mape_results.csv") # Имя файла с результатами MAPE
    if os.path.exists(mape_file):
        try:
            df = pd.read_csv(mape_file)
            if 'segment_id' in df.columns:
                df['segment_id'] = df['segment_id'].astype(str)
            # Проверяем и переименовываем колонку с MAPE, если нужно
            if 'mape' in df.columns and 'mape_fraction' not in df.columns:
                 df.rename(columns={'mape': 'mape_fraction'}, inplace=True)
            elif 'MAPE' in df.columns and 'mape_fraction' not in df.columns: # Учитываем возможный регистр
                 df.rename(columns={'MAPE': 'mape_fraction'}, inplace=True)
            return df
        except Exception as e:
            st.error(f"Error loading mape_results.csv: {e}")
            return pd.DataFrame(columns=['model_type', 'segment_id', 'mape_fraction']) # Возвращаем пустой DF с ожидаемыми колонками
    else:
        st.warning("File mape_results.csv not found. MAPE data will not be displayed.")
        return pd.DataFrame(columns=['model_type', 'segment_id', 'mape_fraction'])

@st.cache_data
def load_forecast_data(segment_id_param):
    """Loads forecast data for the specified segment."""
    # Создаем безопасное имя файла из segment_id
    safe_segment_id_str = str(segment_id_param).replace('.', '_').replace(' ', '_').replace('/', '_')
    forecast_file = os.path.join(DATA_PATH, f"forecast_data_segment_{safe_segment_id_str}.csv")
    if os.path.exists(forecast_file):
        try:
            df = pd.read_csv(forecast_file)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            else:
                st.error(f"'date' column not found in {forecast_file}.")
                return pd.DataFrame()
            return df
        except Exception as e:
            st.error(f"Error loading {forecast_file}: {e}")
            return pd.DataFrame()
    else:
        st.warning(f"File {forecast_file} not found for segment '{segment_id_param}'.")
        return pd.DataFrame()

def get_available_segments(mape_df_param):
    """Gets a list of available segments from MAPE DataFrame or by scanning files."""
    segments_list = []
    # Сначала пытаемся получить сегменты из MAPE данных, если они есть
    if mape_df_param is not None and not mape_df_param.empty and 'segment_id' in mape_df_param.columns:
        segments_list = sorted(mape_df_param['segment_id'].unique().tolist())

    # Если из MAPE не получили или хотим дополнить, сканируем файлы
    # (можно сделать опциональным или как fallback)
    try:
        scanned_segments = []
        for filename_scan in os.listdir(DATA_PATH):
            if filename_scan.startswith("forecast_data_segment_") and filename_scan.endswith(".csv"):
                segment_name_scan = filename_scan.replace("forecast_data_segment_", "").replace(".csv", "")
                scanned_segments.append(segment_name_scan)

        # Объединяем и убираем дубликаты
        combined_segments = sorted(list(set(segments_list + scanned_segments)))
        segments_list = combined_segments

    except FileNotFoundError:
        st.error(f"Directory {DATA_PATH} not found. Cannot determine available segments by scanning files.")

    # Перемещаем 'Global' в конец списка, если он там есть
    if 'Global' in segments_list:
        segments_list.remove('Global')
        segments_list.append('Global')
    return segments_list


# --- Main Application ---
st.set_page_config(layout="wide", page_title="Demand Analysis and Forecasting")
st.title("Demand Analysis and Forecasting by Cluster")

mape_data_df = load_mape_results()

st.sidebar.header("Visualization Settings")
available_segments_list = get_available_segments(mape_data_df)

if not available_segments_list:
    st.error("Could not find data for any segments. Please check the DATA_PATH and ensure CSV files (mape_results.csv and forecast_data_segment_*.csv) are available and correctly named.")
    st.stop()

# Установка индекса по умолчанию для selectbox
default_index = 0
if DEFAULT_SEGMENT in available_segments_list:
    default_index = available_segments_list.index(DEFAULT_SEGMENT)
elif available_segments_list: # Если DEFAULT_SEGMENT нет, но список не пуст, берем первый
    pass # default_index уже 0
else: # Если список пуст (хотя выше есть st.stop(), но на всякий случай)
    st.error("No segments available for selection.")
    st.stop()

selected_segment_id_str = st.sidebar.selectbox(
    "Select segment (Cluster or Global):",
    options=available_segments_list,
    index=default_index
)

forecast_data_df = load_forecast_data(selected_segment_id_str)

st.sidebar.subheader(f"MAPE for segment: {selected_segment_id_str}")
if not mape_data_df.empty and 'segment_id' in mape_data_df.columns and 'mape_fraction' in mape_data_df.columns and 'model_type' in mape_data_df.columns:
    segment_mape_data = mape_data_df[mape_data_df['segment_id'] == str(selected_segment_id_str)]
    if not segment_mape_data.empty:
        # Сортируем модели для отображения MAPE в сайдбаре
        # Используем порядок из MODEL_DISPLAY_NAMES для консистентности
        sorted_model_names_for_sidebar = [name for name in MODEL_DISPLAY_NAMES.values() if name in segment_mape_data['model_type'].unique()]
        sorted_model_names_for_sidebar += [name for name in segment_mape_data['model_type'].unique() if name not in sorted_model_names_for_sidebar]


        for model_name_mape in sorted_model_names_for_sidebar:
            mape_row = segment_mape_data[segment_mape_data['model_type'] == model_name_mape]
            if not mape_row.empty:
                mape_frac_val = mape_row['mape_fraction'].iloc[0]
                if pd.notna(mape_frac_val):
                    st.sidebar.metric(label=f"{model_name_mape}", value=f"{mape_frac_val:.2%}")
                else:
                    st.sidebar.text(f"{model_name_mape}: N/A")
            else:
                 st.sidebar.text(f"{model_name_mape}: (No MAPE data)") # Если для какой-то модели из списка нет MAPE
    else:
        st.sidebar.write("MAPE data for this segment not found in mape_results.csv.")
else:
    st.sidebar.write("MAPE data (mape_results.csv) not loaded or missing required columns (segment_id, model_type, mape_fraction).")


st.header(f"Actual Demand and Forecasts for Segment: {selected_segment_id_str}")

if not forecast_data_df.empty and 'date' in forecast_data_df.columns:
    fig = go.Figure()

    # Добавляем фактические данные
    if 'actual' in forecast_data_df.columns:
        actual_trace_plot_data = forecast_data_df[forecast_data_df['actual'].notna()]
        if not actual_trace_plot_data.empty:
            fig.add_trace(go.Scatter(
                x=actual_trace_plot_data['date'],
                y=actual_trace_plot_data['actual'],
                mode='lines+markers', name='Actual',
                line=dict(color=MODEL_COLORS.get('Actual', 'black'), width=2),
                marker=dict(size=4)
            ))

    # Добавляем прогнозы моделей
    # Итерируемся по MODEL_DISPLAY_NAMES, чтобы сохранить порядок и использовать корректные имена
    for model_key, model_display_name_plot in MODEL_DISPLAY_NAMES.items():
        pred_col_name_plot = MODEL_PRED_COLUMNS.get(model_key) # Получаем имя колонки из MODEL_PRED_COLUMNS
        if pred_col_name_plot and pred_col_name_plot in forecast_data_df.columns:
            valid_preds_plot = forecast_data_df[forecast_data_df[pred_col_name_plot].notna()]
            if not valid_preds_plot.empty:
                fig.add_trace(go.Scatter(
                    x=valid_preds_plot['date'], y=valid_preds_plot[pred_col_name_plot],
                    mode='lines', name=f"{model_display_name_plot} Pred", # Используем отображаемое имя
                    line=dict(color=MODEL_COLORS.get(model_display_name_plot, 'grey'), dash='dash') # Цвет по отображаемому имени
                ))

                # Добавляем доверительные/предсказательные интервалы, если есть
                interval_cols_plot = MODEL_INTERVAL_COLUMNS.get(model_key) # Ищем интервалы по ключу модели
                if interval_cols_plot and \
                   interval_cols_plot['lower'] in forecast_data_df.columns and \
                   interval_cols_plot['upper'] in forecast_data_df.columns:

                    lower_bound_col_plot = interval_cols_plot['lower']
                    upper_bound_col_plot = interval_cols_plot['upper']

                    # Данные для интервала (где есть и прогноз, и границы)
                    interval_plot_data = forecast_data_df[
                        forecast_data_df[pred_col_name_plot].notna() &
                        forecast_data_df[lower_bound_col_plot].notna() &
                        forecast_data_df[upper_bound_col_plot].notna()
                    ].sort_values(by='date') # Важно отсортировать для корректной заливки

                    if not interval_plot_data.empty:
                        # Верхняя граница (невидимая линия, нужна для fill='tonexty')
                        fig.add_trace(go.Scatter(
                            x=interval_plot_data['date'], y=interval_plot_data[upper_bound_col_plot],
                            mode='lines', line=dict(width=0), hoverinfo='skip',
                            showlegend=False, name=f"{model_display_name_plot} Upper Bound" # Имя для отладки, не для легенды
                        ))
                        # Нижняя граница с заливкой до верхней
                        fig.add_trace(go.Scatter(
                            x=interval_plot_data['date'], y=interval_plot_data[lower_bound_col_plot],
                            mode='lines', line=dict(width=0),
                            fillcolor=MODEL_COLORS.get(f"{model_display_name_plot} Lower", MODEL_COLORS.get(model_key + " Lower", 'rgba(128,128,128,0.2)')), # Цвет заливки
                            fill='tonexty', hoverinfo='skip', # Заливка до предыдущего (верхней границы)
                            showlegend=False, name=f"{model_display_name_plot} Lower Bound" # Имя для отладки
                        ))
    fig.update_layout(
        title=f'Demand for Segment: {selected_segment_id_str}',
        xaxis_title='Date', yaxis_title='Demand Volume', # Убрал (total_count) для общности
        legend_title_text='Data Series', hovermode="x unified"
    )
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

    # Отображение таблицы с данными
    if st.checkbox("Show recent data in table", value=False): # По умолчанию таблица скрыта
        last_n_days_for_table = 30 # Количество дней до конца исторических данных

        # Определяем последнюю дату с фактическими данными
        last_actual_date = forecast_data_df[forecast_data_df['actual'].notna()]['date'].max()

        if pd.isna(last_actual_date): # Если фактических данных нет, берем просто последние N+H записей
            display_df_for_table_filtered = forecast_data_df.tail(last_n_days_for_table + FORECAST_HORIZON).copy()
        else:
            # Даты для отображения: N дней до last_actual_date + FORECAST_HORIZON дней после
            start_display_date = last_actual_date - pd.Timedelta(days=last_n_days_for_table -1) # -1 т.к. включаем last_actual_date
            end_display_date = last_actual_date + pd.Timedelta(days=FORECAST_HORIZON)

            display_df_for_table_filtered = forecast_data_df[
                (forecast_data_df['date'] >= start_display_date) &
                (forecast_data_df['date'] <= end_display_date)
            ].copy()

        # Формируем список колонок для таблицы
        cols_for_table_display = ['date', 'actual']
        for model_key in MODEL_DISPLAY_NAMES.keys(): # Идем по ключам, чтобы сохранить порядок
            pred_col = MODEL_PRED_COLUMNS.get(model_key)
            if pred_col and pred_col in display_df_for_table_filtered.columns:
                cols_for_table_display.append(pred_col)

            interval_cols = MODEL_INTERVAL_COLUMNS.get(model_key)
            if interval_cols:
                if interval_cols['lower'] in display_df_for_table_filtered.columns:
                    cols_for_table_display.append(interval_cols['lower'])
                if interval_cols['upper'] in display_df_for_table_filtered.columns:
                    cols_for_table_display.append(interval_cols['upper'])

        # Оставляем только существующие колонки
        cols_present_in_df_for_table = [col_tab for col_tab in cols_for_table_display if col_tab in display_df_for_table_filtered.columns]
        display_df_for_table_final = display_df_for_table_filtered[cols_present_in_df_for_table].copy()

        # Форматирование числовых колонок
        for col_fmt_table in cols_present_in_df_for_table:
            if col_fmt_table == 'date':
                display_df_for_table_final[col_fmt_table] = display_df_for_table_final[col_fmt_table].dt.strftime('%Y-%m-%d')
            elif col_fmt_table == 'actual':
                 display_df_for_table_final[col_fmt_table] = display_df_for_table_final[col_fmt_table].apply(lambda x_val: f"{int(x_val)}" if pd.notna(x_val) else "N/A")
            elif display_df_for_table_final[col_fmt_table].dtype in [np.float64, np.int64]: # Форматируем только числовые
                display_df_for_table_final[col_fmt_table] = display_df_for_table_final[col_fmt_table].apply(lambda x_val: f"{x_val:.2f}" if pd.notna(x_val) else "N/A")

        st.dataframe(display_df_for_table_final.sort_values(by='date', ascending=False).reset_index(drop=True))

else:
    st.warning(f"No data to plot for segment '{selected_segment_id_str}'. Check if the CSV file exists, is not empty, and contains a 'date' column.")

# Сравнение с глобальной моделью
if selected_segment_id_str != 'Global' and 'Global' in available_segments_list:
    st.header("Comparison with Global Model Performance (MAPE)")
    if not mape_data_df.empty and 'mape_fraction' in mape_data_df.columns and 'model_type' in mape_data_df.columns:
        global_mape_comp_data = mape_data_df[mape_data_df['segment_id'] == 'Global']
        cluster_mape_comp_data = mape_data_df[mape_data_df['segment_id'] == str(selected_segment_id_str)]

        if not global_mape_comp_data.empty and not cluster_mape_comp_data.empty:
            comparison_data_list = []
            # Используем MODEL_DISPLAY_NAMES для порядка и имен моделей
            for model_key, model_display_name_comp in MODEL_DISPLAY_NAMES.items():
                # Находим MAPE для текущей модели в данных глобального сегмента
                gm_mape_row = global_mape_comp_data[global_mape_comp_data['model_type'] == model_display_name_comp]
                gm_frac_comp = gm_mape_row['mape_fraction'].iloc[0] if not gm_mape_row.empty and pd.notna(gm_mape_row['mape_fraction'].iloc[0]) else np.nan

                # Находим MAPE для текущей модели в данных кластера
                cm_mape_row = cluster_mape_comp_data[cluster_mape_comp_data['model_type'] == model_display_name_comp]
                cm_frac_comp = cm_mape_row['mape_fraction'].iloc[0] if not cm_mape_row.empty and pd.notna(cm_mape_row['mape_fraction'].iloc[0]) else np.nan

                improvement_display_str_comp = "N/A"
                if pd.notna(gm_frac_comp) and pd.notna(cm_frac_comp) and gm_frac_comp != 0: # Добавлена проверка gm_frac_comp != 0
                    diff_frac_comp = gm_frac_comp - cm_frac_comp # Положительное значение = улучшение
                    perc_diff_val_comp = (diff_frac_comp / gm_frac_comp) * 100
                    improvement_display_str_comp = f"{perc_diff_val_comp:+.2f}%" # '+' для положительных значений
                elif pd.notna(gm_frac_comp) and pd.notna(cm_frac_comp) and gm_frac_comp == 0 and cm_frac_comp == 0:
                     improvement_display_str_comp = "0.00% (both zero)"
                elif pd.notna(gm_frac_comp) and pd.notna(cm_frac_comp) and gm_frac_comp == 0 and cm_frac_comp != 0:
                     improvement_display_str_comp = "Worse (Global was zero)"


                comparison_data_list.append({
                    'Model': model_display_name_comp, # Отображаемое имя
                    f'MAPE Cluster {selected_segment_id_str}': f"{cm_frac_comp:.2%}" if pd.notna(cm_frac_comp) else "N/A",
                    'MAPE Global': f"{gm_frac_comp:.2%}" if pd.notna(gm_frac_comp) else "N/A",
                    'Improvement by Clustering': improvement_display_str_comp
                })

            if comparison_data_list:
                comparison_df_to_display = pd.DataFrame(comparison_data_list)
                st.dataframe(comparison_df_to_display.set_index('Model'))
            else:
                st.write("Could not generate comparison data. Ensure model names in MODEL_DISPLAY_NAMES match those in mape_results.csv.")
        else:
            st.write("Not enough MAPE data available for a meaningful comparison between the selected cluster and the global model.")
    else:
        st.write("MAPE data (mape_results.csv) not loaded or missing required columns, comparison not possible.")

st.sidebar.info("Application for visualizing demand forecasts.")
