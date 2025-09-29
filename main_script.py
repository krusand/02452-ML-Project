import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

sns.set_style('darkgrid')
sns.set_theme(font_scale=1.5)

df = pd.read_csv("data/heartDisease.csv")
df = df.drop(labels=["row.names"], axis=1)
famhist_le = LabelEncoder()
df["famhist_num"] = famhist_le.fit_transform(df["famhist"])

print(df.describe())

# pairplot
g = sns.pairplot(df, hue = "chd", corner=True)
g.fig.suptitle("Pairplot of variables in dataset", y=1.01, fontsize=34)

plt.show()

# histograms
fig, axs = plt.subplots(5, 2, figsize=(10, 24))
axs = np.array(axs)
fig.suptitle("Histograms of Soutch African heart disease data attributes", fontsize=28)

for i, ax in enumerate(axs.reshape(-1)): 

    ax.hist(df.iloc[:, i], color=f"C{i}", bins=20, edgecolor='black')
    ax.set_title(df.columns[i])
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
plt.tight_layout()
plt.show()

# Correlation matrix
sns.set_theme(font_scale=0.9)
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True, mask=np.triu(corr_matrix))
plt.suptitle("Correlation matrix of numeric attributes", y=0.95, fontsize=20)
plt.show()
