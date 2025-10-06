# Genetic Feature Clustering using K-Means

This repository explores the use of **K-Means clustering** to group genetic markers based on their presence across samples, followed by classification modeling to identify the best-performing supervised algorithm for cluster prediction.

---

## 📁 Project Structure

* **`Kmeans Clustering - Genetic Features.ipynb`**
  Main Jupyter notebook containing all steps of the project:

  * Data preprocessing and cleaning
  * Feature scaling
  * K-Means clustering with elbow method visualization
  * Cluster assignment and merging back into the dataset
  * Comparison of supervised learning models (Decision Trees, Logistic Regression, Random Forest, SVM, Neural Networks, etc.)

* **`Clustering Results.pdf`**
  Summarizes model evaluation results across multiple algorithms with accuracy, precision, and recall metrics.
  Includes explanation of the clustering process and rationale for selecting `k = 7`.

---

## 🧠 Methodology Overview

1. **Clustering Approach**

   * K-Means clustering applied to genetic features.
   * The number of clusters (`k`) was determined using the **elbow method**; although no clear elbow was observed, `k = 7` was chosen based on visual inspection.

2. **Modeling**

   * After clustering, the cluster labels were used as target variables.
   * Multiple classification algorithms were evaluated to predict cluster membership, including:

     * Logistic Regression (L1, L2, and LBFGS solvers)
     * Random Forests with varied depth and min-sample splits
     * K-Nearest Neighbors (k=5–8)
     * Support Vector Machines (linear and RBF kernels)
     * Neural Networks (ReLU and Logistic activations)
     * Decision Trees
     * Linear Discriminant Analysis (with shrinkage options)

3. **Performance Evaluation**

   * Models were compared using Accuracy, Precision, and Recall.
   * Cross-validation was applied to top-performing models to confirm stability.

---

## 📊 Key Results

| Model                       | Accuracy | Precision_0 | Precision_1 | Recall_0 | Recall_1 | Recall_Diff |
| --------------------------- | -------- | ----------- | ----------- | -------- | -------- | ----------- |
| Decision Tree (max_depth=3) | **0.55** | 0.52        | 0.57        | 0.55     | 0.54     | **0.00**    |
| Neural Network (ReLU)       | 0.55     | 0.54        | 0.56        | 0.39     | 0.70     | 0.31        |
| SVM (RBF)                   | 0.55     | 0.55        | 0.56        | 0.38     | 0.71     | 0.33        |

The **Decision Tree with max depth = 3** achieved the most balanced performance, though overall accuracy across all models remained around 55%, suggesting the data may not contain strong separable patterns under current preprocessing assumptions.

---

## 🖼️ Visuals

Below are sample visualizations from the clustering and modeling process.
Replace these placeholders with your own generated plots once available.

### 1. Elbow Method Chart

Used to determine the optimal number of clusters (`k`).
![Elbow Method Plot](images/elbow_plot_sample.png "Elbow Method Plot")

### 2. Cluster Visualization (PCA Projection)

Clusters projected onto two principal components for interpretability.
![Cluster Visualization](images/cluster_visualization_sample.png "Cluster Visualization")

### 3. Model Performance Comparison

Performance of top models compared side-by-side.
![Model Performance Comparison](images/model_performance_sample.png "Model Performance Comparison")

### 4. Decision Tree Example

Visual representation of the decision tree used for classification.
![Decision Tree Visualization](images/decision_tree_sample.png "Decision Tree Visualization")

---

## 📈 Conclusions

* The dataset may lack distinct genetic clustering structures, as indicated by uniformly low model accuracy.
* K-Means clustering with **k=7** provided modest separability but no strong predictive boundaries.
* The **Decision Tree** model was the most stable and interpretable classifier across evaluations.

---

## 🧩 Next Steps

* Explore **dimensionality reduction** (e.g., PCA, t-SNE) before clustering to improve separation.
* Consider **hierarchical clustering** or **DBSCAN** for non-spherical cluster shapes.
* Evaluate **feature selection** to isolate the most discriminative genetic markers.

---

## 🧬 Author

This project was developed as part of an exploration in **unsupervised learning and model evaluation for genetic feature analysis**.

---

**License:** MIT
**Python Version:** 3.10+
**Dependencies:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
