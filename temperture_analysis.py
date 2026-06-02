import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

#cleaning and reading data
df = pd.read_csv('top100cities_weather_data.csv')
#remove duplicates and drop null values
df.drop_duplicates(inplace = True)
#drop null values
df.dropna(inplace=True)

#only keep number ones
df.select_dtypes(include='number')
# I want to only check all the columns that are releated to temperature
df_subset = df.loc[:, ['Temperature (Celsius)', 'Wind Speed (m/s)']]


#normalization
sc = StandardScaler()
X = sc.fit_transform(df_subset)
#distance measure[]
distance_measure = linkage(X, method='ward', metric='euclidean')
plt.figure(figsize=(12, 8))
dendrogram(distance_measure, labels = df['City'].values, leaf_rotation= 90)
#Title of the plot
plt.title('Hierarchical Weather Clustering')
plt.xlabel('City')
plt.ylabel('Ward Linkage Distance')

#Dendogram

plt.tight_layout()
plt.show()

