# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:03:53 2022

@author: pc
"""
import streamlit as st
import pandas as pd
import os
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model


if os.path.exists('sourcedata.csv'):
    df = pd.read_csv('sourcedata.csv', index_col=None)

with st.sidebar:
    st.image(
        'https://www.ucf.edu/news/files/2022/10/Medicine-Artifical-Intelligence.jpg')
    st.title('Otomatik Makine Öğrenmesi Arayüzü')
    choice = st.radio('Seçenekler', [
                      'Yükleme', 'Veri İşleme ve Görselleştirme', 'Makine Öğrenmesi', 'İndirme'])
    st.info('Bu uygulama eğitmek istediğiniz csv formatındaki verisetini otomatik işleyip, görselleştirip, makine öğrenmesi modelleriyle eğitip, modelleri karşılaştırıp, en iyi modeli verir')

if choice == 'Yükleme':
    st.title('Lütfen Eğitmek İstediğiniz Veri Setini CSV Formatında Yükleyiniz..!')
    file = st.file_uploader('Veri Setini Buradan Yükleyebilirsiniz')
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('sourcedata.csv', index=None)
        st.dataframe(df)
if choice == 'Veri İşleme ve Görselleştirme':
    st.title('Otomatik Keşfedici Veri Çözümlemesi')
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == 'Makine Öğrenmesi':
    st.title('Hadi Model Eğitelim..!')
    target = st.selectbox('Lütfen Sınıf Etiketi Özelliği Seçiniz', df.columns)
    if st.button('Modeli Eğit'):
        df.info()
        setup(df, target=target, silent=True)
        setup_df = pull()
        st.info('Burası Eğitimde Kullanılacak Ayarlardır.')
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info('Eğitilen Model Burada.')
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'best_model')
if choice == 'İndirme':
    st.title('Eğitilmiş En İyi Modeli İndirmek için Tıklayın')
    with open('best_model.pkl', 'rb')as f:
        st.download_button('Modeli İndir', f, 'trained_model.pkl')
