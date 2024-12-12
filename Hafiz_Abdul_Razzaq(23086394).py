import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("airline9.csv")

data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

print("Missing Values: " , data.isnull().sum())
print()

sns.heatmap(data.isnull(), cmap='viridis')
plt.show()

print("Unique Values: " , data.nunique())
print()

print("Data Info: " , data.info()
)

print("Data Description: ", data.describe())
print()

maxrow = data.loc[data['Revenue'].idxmax()]
print("Max Revenue Row: " ,maxrow)
print()

toprows = data.nlargest(10, 'Revenue')
print("Top 10 Rows by Revenue: " , toprows)
print()

total_revenue = data['Revenue'].sum()
total_number = data['Number'].sum()
print(f"Total Revenue: " , total_revenue)
print(f"Total Number: " , total_number)

print(f"Max Price: " , data['Price'].max())
print(f"Min Price: " , data['Price'].min())

sns.countplot(data=data, x='Month')
plt.title("Count plot - Hafiz Abdul Razzaq(23086394)")
plt.show()

plt.figure(figsize=(14, 5))
sns.scatterplot(data=data, x="Month", y="Number")
plt.title("Scatter plot - Hafiz Abdul Razzaq(23086394)")
plt.show()

entries = data['Month'].value_counts().sort_index()
print("Entries Per Month: " , entries)
print()

monthrevenue = data.groupby('Month')['Revenue'].sum()

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.pie(monthrevenue, labels=months, autopct='%1.1f%%')
plt.title("Monthly Distribution of Revenue in % - Hafiz Abdul Razzaq(23086394)")
plt.show()

sns.boxplot(data['Price'])
plt.title("Boxplot of Prices - Hafiz Abdul Razzaq(23086394)")
plt.show()

sns.boxplot(data['Revenue'])
plt.title('Boxplot of Revenue - Hafiz Abdul Razzaq(23086394)')
plt.show()

features = ['Revenue', 'Price', 'Number']
scaler = MinMaxScaler()
scalingdata = scaler.fit_transform(data[features])
orgdata = pd.DataFrame(scalingdata, columns=['Revenue', 'Price', 'Number'])
data[features] = orgdata

print("Data after scaling: " , data)
print()

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data[['Revenue']])
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Plot for Revenue - Hafiz Abdul Razzaq(23086394)')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (WCSS)')
plt.xticks(range(1, 11))
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(orgdata)
data['Cluster'] = clusters

sns.scatterplot(x='Number', y='Price', hue='Cluster', data=data)
plt.title('K-Means Clustering - Hafiz Abdul Razzaq(23086394)')
plt.xlabel('Number')
plt.ylabel('Revenue')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()
