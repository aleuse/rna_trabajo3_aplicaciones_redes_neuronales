# Este script fue convertido desde un Jupyter Notebook
# Descarga el dataset desde internet en lugar de usar un archivo local

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split


# Cargar el dataset
import os
for dirname, _, filenames in os.walk('../../data/input/recomendation'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from random import randint,uniform
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

df=pd.read_csv("../../data/input/recomendation/Amazon-Products.csv")
#Eliminar primera columna 'Unnamed'
df.drop(columns=['Unnamed: 0'],inplace=True)
df.head()

df.shape

df.info()

df.drop(columns=['image', 'link'],inplace=True)

# Total percentage of null values
print(f"Total null value in product table:\n{df.isnull().sum()/len(df)}")

df['main_category'].value_counts()

# unique values of ratings column
print(f"Unique values of ratings:\n{df['ratings'].unique()}")
# unique values of no_of_ratings column
print(f"Unique values of no_of_ratings:\n{df['no_of_ratings'].unique()}")

# no_of_ratings column
df['no_of_ratings'] = df['no_of_ratings'].str.replace(",","")
# converting to interger column
df['no_of_ratings'] = pd.to_numeric(df['no_of_ratings'], errors="coerce", downcast="integer")

# ratings column
# Identifying string in column based on regex and replacing with default ratings -> 3.0
df = df.replace({
    'ratings': {r'₹\w+[.]\w+':'3.0', r'FREE':'3.0', r'Get':'3.0', r'₹\w+':'3.0'},
}, regex=True)
df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce', downcast='integer')

# convert price and discount column to integer
df['actual_price'] = pd.to_numeric(df['actual_price'].str.replace("₹","").str.replace(",",""), errors='coerce', downcast='integer')
df['discount_price'] = pd.to_numeric(df['discount_price'].str.replace("₹","").str.replace(",",""), errors='coerce', downcast='integer')

# Handling Null Values
df['ratings'].fillna(round(df['ratings'].mean(),1), inplace=True) # replacing null values with mean
df['no_of_ratings'].fillna(round(df['no_of_ratings'].mean()), inplace=True) # replacing null values with mean

# Handling Null Values in price and discount column
# Filling the null values based on sub category, grouping the rows by sub category,
# find the mean -> replace the NULL values

mean_actual_price_by_sub_category = round(df.groupby('sub_category')[['actual_price','discount_price']].mean()).reset_index()

# Creating dictionary of sub_category with respective mean
# For eg {"Air Conditioners ": 54815.0, "All Applicances": 7017..0 ..........}
mean_actual_price_dict = mean_actual_price_by_sub_category.set_index('sub_category')['actual_price'].to_dict()
mean_discount_price_dict = mean_actual_price_by_sub_category.set_index('sub_category')['discount_price'].to_dict()

# replacing the null values with the mean
df['actual_price'] = df.apply(lambda x: x['actual_price'] if pd.notna(x['actual_price']) else mean_actual_price_dict.get(x['sub_category']), axis=1)
df['discount_price'] = df.apply(lambda x: x['discount_price'] if pd.notna(x['discount_price']) else mean_discount_price_dict.get(x['sub_category']), axis=1)

# Drop Duplicates
df = df.drop_duplicates()

#Distribution of product by ratings
sns.countplot(x='ratings',data=df)
plt.title("Distribution of df rating",fontsize=13)
plt.xticks(rotation=90)
plt.show()

df['productId'] = pd.factorize(df['name'])[0] + 1 # + 1 to start from 1, not 0

print(f"Unique Count of product name: {df['name'].nunique()}")
print(f"Unique Count of product Id: {df['productId'].nunique()}")

#Analisis preliminar de dataset actual
print(df.info())
print(df.describe())


#distribución de ratings
plt.figure(figsize=(10,5))
sns.histplot(df['ratings'], bins=10, kde=True)
plt.title("Distribución de Ratings")
plt.xlabel("Rating")
plt.ylabel("Frecuencia")
plt.show()

#Productos con más calificaciones
top_products = df.groupby('name')['ratings'].count().sort_values(ascending=False).head(10)
plt.figure(figsize=(12,6))
sns.barplot(x=top_products.values, y=top_products.index, palette="viridis")
plt.title("Top 10 Productos con Más Calificaciones")
plt.xlabel("Número de Calificaciones")
plt.ylabel("Producto")
plt.show()


DATA_PATH = os.path.join('..', '..', 'data', 'input', 'recomendation')
df.to_csv(os.path.join(DATA_PATH, 'data_processed.csv'), index=False)

