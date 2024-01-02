import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import warnings
warnings.filterwarnings("ignore")

st.write('# Shop Customer Clustering')
st.write('Berikut adalah dataset dummy yang berisi karakteristik konsumen sebuah toko.')
st.write('Pada program ini dataset konsumen ini akan diklaster menggunakan algoritma KMeans.')
data_path = 'Customers.csv'
df = pd.read_csv(data_path)
st.write(df)

df.isnull().sum()

profession_dis = df.Profession.value_counts()
categorical_columns = ['Gender', 'Profession']

for cat_col in categorical_columns:
  encoder = LabelEncoder()
  df[cat_col] = encoder.fit_transform(df[cat_col])

df.drop(columns=['CustomerID'], inplace=True)
X = df
X_test = df

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans.fit(X)
print(kmeans.cluster_centers_)
df["Cluster"] = kmeans.labels_
kluster = kmeans.labels_
st.write('## Data yang telah diklaster')
st.write('Tiap record akan memiliki label berupa klaster mana dia berada')
st.write(df)
st.write('Data yang telah ditraining di atas diubah menjadi bentuk numerik untuk bisa diolah dalam algoritma KMeans')
st.write('# Masukkan Data yang Akan di Cluster')

gender = st.radio(
    "## Gender :",
    ["Laki - Laki", "Perempuan"],
    )

if gender == 'Laki - Laki':
  g = 1
else:
  g = 0

age = st.slider('## Age : ', 0, 100, 25)

income = st.slider('## Annual Income ($) : ', 1000, 200000,50000)

s_score = st.slider('## Spending Score : ', 1, 100, 25)

exp = st.slider('## Work Experience (in year) : ', 0, 15, 4)

fam = st.slider('## Family Size : ', 1, 10, 4)

prof = st.radio(
    "## Profession :",
    ["Healthcare", "Engineer", "Lawyer", "Artist", "Entertainment", "Doctor", "Homemaker", "Executive", "Marketing"],
    )

if prof == "Healthcare" :
  p = 5
elif prof == "Engineer" :
  p = 2
elif prof == "Lawyer" :
  p = 7
elif prof == "Artist" :
  p = 0
elif prof == "Entertainment" :
  p = 3
elif prof == "Doctor" :
  p = 1
elif prof == "Homemaker" :
  p = 6
elif prof == "Executive" :
  p = 4
elif prof == "Marketing" :
  p = 8

df2 = X_test
bt1 = st.button("Calculate Data")
if bt1 == True :
  st.write("## KMeans Clustering Result")
  X_test.drop(columns=['Cluster'], inplace=True)
  df2.loc[len(df2.index)] = [g, age, income, s_score, p, exp, fam] 
  kmeans_2 = KMeans(n_clusters=3, n_init=10, random_state=42)
  kmeans_2.fit(X_test)
  print(kmeans_2.cluster_centers_)
  df["Cluster"] = kmeans_2.labels_
  st.write(df.iloc[-1])
  bt1 = False





