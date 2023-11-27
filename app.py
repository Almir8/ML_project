import streamlit as st
import pandas as pd
from joblib import load
import warnings
# from db.db import *
import datetime
import os
import sqlite3
import plotly.express as px

XGB_model_loaded = load('XGB_model.joblib')

df = pd.read_csv(r"C:\Users\Almir\Desktop\app\ai4i2020.csv")

# Обработки названий признаков
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(' ', '')
df.columns = df.columns.str.replace("[°c]", '')
df.columns = df.columns.str.replace("[", '')
df.columns = df.columns.str.replace("]", '')

df["airtemperaturek"] = df["airtemperaturek"] - 272.15
df["processtemperaturek"] = df["processtemperaturek"] - 272.15

st.set_page_config(
    page_title="Просмотр набора данных в реальном времени",
    page_icon=":sparkles:",
    layout="wide",
)

class DbManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.con = sqlite3.connect(db_path)
        self.cur = self.con.cursor()
        
    def get_all_history(self):
        for row in self.cur.execute("SELECT * FROM usage_history ORDER BY id"):
            print(row)
            
    def insert_history(self, date, time):
        self.cur.execute(f"INSERT INTO usage_history(date, time) VALUES (?, ?)", (date, time))
        print(self.cur.lastrowid)
        self.con.commit()

    def get_all_results(self):
        for row in self.cur.execute("SELECT * FROM result_history ORDER BY id"):
            print(row)

    def result_history(self, type_frais, air_temp, proc_temp, rotation_speed, torque, tool_wear, result_string):
        self.cur.execute(f"INSERT INTO result_history(type_frais, air_temp, proc_temp, rotation_speed, torque, tool_wear, result_string) VALUES (?, ?, ?, ?, ?, ?, ?)", (type_frais, air_temp, proc_temp, rotation_speed, torque, tool_wear, result_string))
        print(self.cur.lastrowid)
        self.con.commit()

def main():
    # Заголовок страницы
    st.title("Определение вероятности поломки оборудования")
    
    def format_option(option):
        mapping = {1:'M', 2:'L', 3:'H'}
        return mapping.get(option, str(option))
        
    # Создание полей ввода
    options = [1, 2, 3]
    def_option = ['M', 'L', 'H']
    def_val = 'M'
    with st.sidebar:
        type_frais = st.selectbox("Введите тип оборудования", options, format_func=format_option, placeholder='Введите тип оборудования', index=def_option.index(def_val))
        air_temp = st.number_input("Введите температуру воздуха", min_value=0, value=0, step=1)
        proc_temp = st.number_input("Введите температуру процесса", min_value=0, value=0, step=1)
        rotation_speed = st.number_input("Введите скорость вращения", min_value=0, value=0, step=1)
        torque = st.number_input("Введите крутящий момент", min_value=0, value=0, step=1)
        tool_wear = st.number_input("Введите износ инструмента", min_value=0, value=0, step=1)

    # Обработка введенных данных
    result_str, probability, result_color = process_data(type_frais,air_temp,proc_temp,rotation_speed,torque,tool_wear)
    
    # Display the result with the color
    st.markdown(f"### Результат: <span style='color:{result_color}'>{result_str} </span>", unsafe_allow_html=True)
    db_manager = DbManager('db.db')
    x = datetime.datetime.now()
    date, time = str(x).split(' ')
    db_manager.insert_history(date, time)
    db_manager.result_history(type_frais, air_temp, proc_temp, rotation_speed, torque, tool_wear, result_str)
    #st.markdown(f"### Результат: <span style='color:{result_color}'>{result_str} ({probability}%)</span>", unsafe_allow_html=True)
    
def get_result_color(probability):
    if probability < 50:
        return 'red'
    elif probability >= 50 and probability < 75:
        return 'yellow'
    else:
        return 'green'

def process_data(type_frais,air_temp,proc_temp,rotation_speed,torque,tool_wear):
    #формируем тестовый датафрейм
    list_tuples =[[type_frais,air_temp,proc_temp,rotation_speed,torque,tool_wear]]
    X_test = pd.DataFrame(list_tuples, columns=['type', 'airtemperature', 'processtemperature', 
                                                'rotationalspeedrpm', 'torquenm', 'toolwearmin'])
    #прогноз
    y_pred_dec = XGB_model_loaded.predict(X_test).tolist()[0]
    probability_proba = XGB_model_loaded.predict_proba(X_test)
    probability = round(probability_proba.tolist()[0][1]*100,2)
    if probability < 50:
        result_str = 'оборудование подвержено поломке'
    elif probability >= 50 and probability < 75:
        result_str = 'оборудования находится в неоптимальном режиме'
    else:
        result_str = 'оборудование работает в оптимальном режиме'
    result_color = get_result_color(probability)
    return result_str, probability, result_color
    

if __name__ == "__main__":
    main()

#название страницы
st.title("Просмотр набора данных в реальном времени")

#фильтр в верхней части панели
type_frais = st.selectbox("Выбрать тип оборудования", pd.unique(df["type"]))

filtered_df = df[df["type"] == type_frais]

# Создание словаря для переименования параметров
parameter_label_mapping = {
    "airtemperaturek": "Температура воздуха (°C)",
    "processtemperaturek": "Температура процесса (°C)",
    "rotationalspeedrpm": "Скорость вращения (RPM)",
    "torquenm": "Крутящий момент, Нм",
    "toolwearmin": "Износ инструмента"
}

# Создание полей для выбора параметров x и y
x_parameter_selector = st.selectbox("Выберите параметр для оси X", list(parameter_label_mapping.values()))
y_parameter_selector = st.selectbox("Выберите параметр для оси Y", list(parameter_label_mapping.values()))
hist_parametr = st.selectbox("Выберите параметр для гистограммы", list(parameter_label_mapping.values()))

# Показать графики только если фильтрованный датафрейм не пустой
if not filtered_df.empty:
    fig, fig2 = st.columns(2)
    # Построение графика
    with fig:
        fig = px.scatter(
            filtered_df,
            x=[k for k, v in parameter_label_mapping.items() if v == x_parameter_selector][0],
            y=[k for k, v in parameter_label_mapping.items() if v == y_parameter_selector][0],
            title=f'{y_parameter_selector} относительно {x_parameter_selector}',
            labels=parameter_label_mapping
        )
        fig.update_layout(xaxis_title=parameter_label_mapping.get(x_parameter_selector), yaxis_title=parameter_label_mapping.get(y_parameter_selector))
        st.plotly_chart(fig)
    with fig2:
        fig2 = px.histogram(
            filtered_df,
            x=[k for k, v in parameter_label_mapping.items() if v == hist_parametr][0],
            title=f'Распределение параметра {hist_parametr}',
            labels=parameter_label_mapping
        )
        fig2.update_layout(bargap=0.2)
        # Отображение графика
        st.plotly_chart(fig2)