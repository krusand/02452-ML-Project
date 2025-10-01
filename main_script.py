import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler


# folder to store plots as images
os.makedirs("figures", exist_ok=True)

# seaborn plotting settings
sns.set_style('darkgrid')
sns.set_theme(font_scale=1.5)

# loading the heart disease dataset and encoding the famhist variable
df = pd.read_csv("data/heartDisease.csv")
df = df.drop(labels=["row.names"], axis=1)
famhist_le = LabelEncoder()
df["famhist_num"] = famhist_le.fit_transform(df["famhist"])
print(df.describe())


##### EDA #####
# pairplot
g = sns.pairplot(df, hue = "chd", corner=True)
g.fig.suptitle("Pairplot of variables in dataset", y=1.01, fontsize=34)
g.savefig("figures/pairplot.png", dpi=300, bbox_inches="tight")

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
fig.savefig("figures/histograms.png", dpi=300, bbox_inches="tight")

# correlation matrix
plt.figure(figsize=(10, 8))
sns.set_theme(font_scale=0.9)
corr_matrix = df.corr(numeric_only=True)
heatmap = sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True, mask=np.triu(corr_matrix))
plt.suptitle("Correlation matrix of numeric attributes", y=0.95, fontsize=20)
heatmap.figure.savefig("figures/corr_matrix.png", dpi=300, bbox_inches="tight")


##### PCA ANALYSIS #####
# preparing data for PCA analysis
X = df.drop(labels=["famhist", "famhist_num", "chd"], axis=1)
y = df.loc[:, "chd"]

# standardizing the data
scaler = StandardScaler()
X_tilde = scaler.fit_transform(X)

# obtaining the principal components
pca = PCA()
pca.fit(X_tilde)
V = pca.components_.T

# PCA component coefficients for a few principal components
bw = 0.2
r = np.arange(1, X.shape[1] + 1)
n_components = 3
colors = ['#225ea8', '#41b6c4', '#c7e9b4']

fig = plt.figure(figsize=(12, 8))
plt.title(f"PCA component coefficients - first {n_components} components", fontsize=24)

for i, pc in enumerate(V[:, :n_components].T):
    color = colors[i] if i < len(colors) else None
    plt.bar(r + i * bw, pc, width=bw, label=f"PC{i+1}", color=color)
    
plt.xticks(r + bw, X.columns)
plt.ylabel("Component coefficients", fontsize=18)
plt.xticks(fontsize=16)  
plt.yticks(fontsize=16) 
plt.legend(fontsize=16)
plt.grid(color="white")
fig.savefig("figures/pca_coef.png", dpi=300, bbox_inches="tight")

# plotting attribute coefficients in PC1/PC2-space
PC_idxs = [0, 1]
fig = plt.figure(figsize=(8, 8))
plt.title("Attribute coefficients in PC space", fontsize=24)

for attr_idx, attr_name in enumerate(X.columns):
    plt.arrow(0, 0, V[attr_idx, PC_idxs[0]], V[attr_idx, PC_idxs[1]], color='black', alpha=0.5)
    plt.text(V[attr_idx, PC_idxs[0]], V[attr_idx, PC_idxs[1]], attr_name, fontsize=18)
    
plt.xlabel(f"PC{PC_idxs[0] + 1}", fontsize=18)
plt.ylabel(f"PC{PC_idxs[1] + 1}", fontsize=18)
plt.xticks(fontsize=16)  
plt.yticks(fontsize=16) 
plt.grid()

plt.plot(np.cos(np.arange(0, 2 * np.pi, 0.01)), np.sin(np.arange(0, 2 * np.pi, 0.01)))   # unit circle
plt.axis("equal")
plt.tight_layout()
fig.savefig("figures/attr_coef.png", dpi=300, bbox_inches="tight")

# plotting the proportion of explained variance per component
rho = pca.explained_variance_ratio_
threshold = 0.9

fig = plt.figure(figsize=(8, 6))

plt.plot(range(1, len(rho) + 1), rho, "x-", color="#225ea8")                 # individual proportion of variance explained
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-", color="#41b6c4")      # accumulated proportion of variance explained
plt.plot([1, len(rho)], [threshold, threshold], "k--")                         # threshold

plt.title("Variance explained by principal components", fontsize=24)
plt.xlabel("#Principal components", fontsize=18)
plt.ylabel("Proportion of variance explained", fontsize=18)
plt.xticks(fontsize=16)  
plt.yticks(fontsize=16) 
plt.legend(["Individual", "Cumulative", "Threshold"], fontsize=16)
plt.grid(color="white")
plt.tight_layout()
fig.savefig("figures/expl_var.png", dpi=300, bbox_inches="tight")

# projected scatter onto the PC1/PC2-space
B = pca.transform(X_tilde)
unique_classes = np.unique(y)

fig = plt.figure(figsize=(8, 6))
plt.title("Data projected onto the PCA space", fontsize=24)

for chd_val in unique_classes:
    mask = (y == chd_val)
    plt.scatter(B[mask, PC_idxs[0]], B[mask, PC_idxs[1]], s=30, alpha=0.8, label=f"chd={chd_val}")

plt.xlabel(f"PC{PC_idxs[0] + 1}", fontsize=18)
plt.ylabel(f"PC{PC_idxs[1] + 1}", fontsize=18)
plt.xticks(fontsize=16)  
plt.yticks(fontsize=16) 
plt.legend(loc="upper left", fontsize=16)
plt.tight_layout()
fig.savefig("figures/projected_scatter.png", dpi=300, bbox_inches="tight")
