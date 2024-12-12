#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read_csv('Car Sales (2).csv')
print(df.head())

import pandas as pd
df = pd.read_csv('Car Sales (2).csv')
from IPython.display import display, HTML

display(HTML(df.head().to_html(index=False)))

import pandas as pd
from IPython.display import display, HTML
import matplotlib.pyplot as plt

df = pd.read_csv('Car Sales (2).csv')

# Assuming 'Company' and 'Annual Income' are column names in your dataframe
top_10_companies = df.groupby('Company')['Annual Income'].sum().nlargest(10)

plt.figure(figsize=(10, 6))
plt.bar(top_10_companies.index, top_10_companies.values)
plt.xlabel('Company')
plt.ylabel('Annual Income')
plt.title('Top 10 Companies by Annual Income')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Car Sales (2).csv')

# Group by 'Dealer_Region' and sum 'Annual Income'
top_10_regions = df.groupby('Dealer_Region')['Annual Income'].sum().nlargest(10)

# Create the bar chart
plt.figure(figsize=(12, 6))
plt.bar(top_10_regions.index, top_10_regions.values)
plt.xlabel('Dealer Region')
plt.ylabel('Annual Income')
plt.title('Top Dealer Regions by Annual Income')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Car Sales (2).csv')

# Assuming 'Model' and 'Annual Income' are column names in your dataframe
top_10_models = df.groupby('Model')['Annual Income'].sum().nlargest(10)

plt.figure(figsize=(10, 8))
plt.pie(top_10_models, labels=top_10_models.index, autopct='%1.1f%%', startangle=90)
plt.title('Top 10 Models by Annual Income')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 8))
heatmap_data = pd.pivot_table(df, values='Price ($)', index='Body Style', aggfunc='mean')
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title('Average Price ($) by Body Style')
plt.xlabel('Average Price')
plt.ylabel('Body Style')
plt.show()

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = df[['Annual Income', 'Price ($)']]

# Determine the optimal number of clusters (k) using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# Based on the elbow method, choose the optimal k (e.g., k=3)
k = 3  # Replace with your chosen k value
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Visualize the clusters
plt.scatter(df['Annual Income'], df['Price ($)'], c=df['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# prompt: provide k-means clustering with data points in elbow graph

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Assuming 'Annual Income' and 'Price ($)' are the columns you want to use for clustering
# and 'df' is your DataFrame

# Determine the optimal number of clusters (k) using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(df[['Annual Income', 'Price ($)']])  # Use only the relevant columns
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 6)) # Adjust figure size if needed
plt.plot(range(1, 11), wcss, marker='o', linestyle='--') # Add markers and linestyle
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True) # Add grid for better readability
plt.show()


# In[ ]:




