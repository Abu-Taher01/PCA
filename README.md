# Principal Component Analysis (PCA) Implementation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243-blue.svg)](https://numpy.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive implementation and tutorial of Principal Component Analysis (PCA) - one of the most fundamental dimensionality reduction techniques in machine learning and data science.

## 📖 What is PCA?

Principal Component Analysis (PCA) is a statistical technique used for dimensionality reduction while preserving the most important information in the data. It transforms high-dimensional data into a lower-dimensional representation by finding the directions (principal components) of maximum variance.

### Key Concepts

- **Dimensionality Reduction**: Reducing the number of features while retaining important information
- **Variance Maximization**: Finding directions that capture the most variance in the data
- **Orthogonal Components**: Each principal component is orthogonal to the others
- **Eigenvalues & Eigenvectors**: Mathematical foundation of PCA

## 🎯 Why PCA?

### Applications
- **Data Visualization**: Reducing high-dimensional data to 2D/3D for plotting
- **Feature Engineering**: Creating new features that capture the most variance
- **Noise Reduction**: Removing less important components that might be noise
- **Computational Efficiency**: Reducing computational cost for large datasets
- **Multicollinearity**: Handling correlated features in regression models

### When to Use PCA
- High-dimensional datasets (>10 features)
- Correlated features
- Need for data visualization
- Computational efficiency requirements
- Noise reduction needs

## 📁 Repository Contents

- `PCA.ipynb` - Comprehensive PCA implementation with examples and visualizations
- `Predict_the_Introverts_from_the_Extroverts.ipynb` - Real-world application using PCA
- `README.md` - This documentation file

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Abu-Taher01/PCA.git
   cd PCA
   ```

2. **Install required dependencies**:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn jupyter
   ```

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

## 📚 Usage

### Basic PCA Implementation

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Sample data
X = np.random.randn(100, 5)  # 100 samples, 5 features

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Explained variance ratio
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2%}")
```

### Step-by-Step PCA Process

```python
# 1. Standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Compute covariance matrix
cov_matrix = np.cov(X_scaled.T)

# 3. Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 4. Sort eigenvalues and eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# 5. Project data onto principal components
X_pca = X_scaled.dot(eigenvectors[:, :2])
```

## 🧠 Mathematical Foundation

### The PCA Algorithm

1. **Standardization**: Center and scale the data
   ```
   X_standardized = (X - μ) / σ
   ```

2. **Covariance Matrix**: Compute the covariance matrix
   ```
   Σ = (1/n) * X^T * X
   ```

3. **Eigenvalue Decomposition**: Find eigenvalues and eigenvectors
   ```
   Σ * v = λ * v
   ```

4. **Projection**: Transform data to new coordinate system
   ```
   X_pca = X * V
   ```

### Key Mathematical Concepts

- **Eigenvalues**: Represent the variance explained by each principal component
- **Eigenvectors**: Represent the directions of maximum variance
- **Explained Variance Ratio**: Proportion of total variance explained by each component
- **Cumulative Explained Variance**: Total variance explained by selected components

## 📊 Examples & Visualizations

### 1. Synthetic Data Example
```python
# Generate correlated data
np.random.seed(42)
X = np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], 100)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# Visualize
plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
plt.title('Original Data')
plt.show()

plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
plt.title('Data after PCA')
plt.show()
```

### 2. Explained Variance Plot
```python
# Plot explained variance ratio
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
        pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Component')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')

plt.tight_layout()
plt.show()
```

## 🔍 Key Concepts Explained

### 1. Dimensionality Reduction
- **Purpose**: Reduce the number of features while preserving information
- **Benefit**: Faster computation, better visualization, reduced overfitting
- **Trade-off**: Some information loss (but minimal if done correctly)

### 2. Variance Maximization
- **Goal**: Find directions that capture maximum variance
- **Result**: First principal component explains the most variance
- **Interpretation**: Higher variance = more important information

### 3. Orthogonality
- **Property**: Principal components are orthogonal (perpendicular)
- **Benefit**: No correlation between components
- **Implication**: Each component captures unique information

### 4. Eigenvalues & Eigenvectors
- **Eigenvalues**: Measure of variance explained by each component
- **Eigenvectors**: Direction vectors of principal components
- **Relationship**: Larger eigenvalue = more important component

## 🎯 Learning Objectives

After working through this repository, you'll understand:

### Theoretical Concepts
- **Linear Algebra**: Eigenvalues, eigenvectors, matrix operations
- **Statistics**: Variance, covariance, standardization
- **Geometry**: Orthogonal transformations, projections

### Practical Applications
- **Data Preprocessing**: When and how to apply PCA
- **Feature Engineering**: Creating new features from PCA
- **Model Performance**: Impact of PCA on machine learning models
- **Visualization**: Reducing dimensions for plotting

### Implementation Skills
- **Scikit-learn**: Using PCA from sklearn.decomposition
- **NumPy**: Manual PCA implementation
- **Matplotlib**: Creating informative visualizations
- **Cross-validation**: Evaluating PCA with ML pipelines

## 📈 Best Practices

### 1. When to Use PCA
- ✅ High-dimensional data (>10 features)
- ✅ Correlated features
- ✅ Need for visualization
- ✅ Computational efficiency required

### 2. When NOT to Use PCA
- ❌ Binary/categorical features (use other techniques)
- ❌ Non-linear relationships (consider t-SNE, UMAP)
- ❌ Interpretability is crucial (PCA components are hard to interpret)
- ❌ Small datasets (might lose too much information)

### 3. Implementation Tips
- **Always standardize** data before PCA
- **Check explained variance** to choose number of components
- **Cross-validate** with your ML model
- **Visualize** the results to understand the transformation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Abdullah Al Mamun**
- GitHub: [@Abu-Taher01](https://github.com/Abu-Taher01)
- LinkedIn: [Abdullah Al Mamun](https://www.linkedin.com/in/abdullah-al-mamun-003913205/)

## 🙏 Acknowledgments

This implementation and the underlying concepts were learned from [CampusX](https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH), an excellent resource for machine learning education. The PCA implementation code in this repository is sourced from CampusX. The code is used for educational purposes to demonstrate PCA concepts and implementation.

*Note: Future updates will include original implementations and modifications to better reflect personal learning and coding style.*

## 🔗 Related Links

- [PCA Notebook](https://github.com/Abu-Taher01/PCA/blob/main/PCA.ipynb)
- [Real-world Application](https://github.com/Abu-Taher01/PCA/blob/main/Predict_the_Introverts_from_the_Extroverts.ipynb)

---

⭐ **Star this repository if you found it helpful!** 
