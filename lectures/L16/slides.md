---
title: MBAI 417
separator: <!--s-->
verticalSeparator: <!--v-->
theme: serif
revealOptions:
  transition: 'none'
---

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 50%; position: absolute;">

  # Data Intensive Systems
  ## L.16 | Clustering

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 10%">

  <iframe src="https://lottie.host/embed/216f7dd1-8085-4fd6-8511-8538a27cfb4a/PghzHsvgN5.lottie" height = "100%" width = "100%"></iframe>
  </div>
</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Welcome to Data Intensive Systems.
  ## Please check in by creating an account and entering the provided code.

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 40%; padding-top: 5%">
    <iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=Check In" width = "100%" height = "100%"></iframe>
  </div>
</div>

<!--s-->

## Customer Transactions Exercise

Let's say we have a dataset of customer transactions, and we want to group customers based on their purchasing behavior. We do not have any behavioral labels for the customers, and manually labeling the data would be both time-consuming and expensive. 

Either way, what we're really interested in is the **structure** of the data. In other words, how can we identify similar patterns in customer behavior without any labels?

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how comfortable are you with topics like:

  1. K-Means
  2. Hierarchical Clustering
  3. DBSCAN

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# L.16 Clustering

</div>

<!--s-->

## Agenda

Supervised machine learning is about predicting $y$ given $X$. Unsupervised machine learning is about finding patterns in $X$, without being given $y$.

There are a number of techniques that we can consider to be unsupervised machine learning, spanning a broad range of methods (e.g. clustering, dimensionality reduction, anomaly detection). In this lecture, we will focus on clustering.

### Clustering
1. Partitional Clustering
2. Hierarchical Clustering
3. Density-Based Clustering

<!--s-->

<div class="header-slide">

# Clustering

</div>

<!--s-->

## Clustering | Applications

Clustering is a fundamental technique in unsupervised machine learning, and has a wide range of applications.

<div class = "col-wrapper">
<div class="c1" style = "width: 70%; font-size: 0.75em;">

<div>

| Application | Example | 
| --- | --- |
| Customer segmentation | Segmenting customers based on purchase history. |
| Document clustering | Grouping similar documents together. |
| Image segmentation | Segmenting an image into regions of interest. |
| Anomaly detection | Detecting fraudulent transactions. |
| Recommendation systems | Recommending products based on user behavior. |

</div>

</div>
<div class="c2" style = "width: 30%">

<div>
<img src="https://cambridge-intelligence.com/wp-content/uploads/2021/01/graph-clustering-800px.png" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Cambridge Intelligence, 2016</p>
</div>

</div>
</div>

<!--s-->

## Clustering | Goals

Clustering is the task of grouping a set of objects in such a way that objects in the same group (or cluster) are more similar to each other than to those in other groups.

A good clustering result has the following properties:

- High intra-cluster similarity
- Low inter-cluster similarity

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*vEsw12wO0KxvYne0m4Cr5w.png" height="50%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Yehoshua, 2023</p>

<!--s-->

## Clustering | Distance Metrics

> "Everything in data science seems to come down to distances and separating stuff." - Anonymous PhD

<div class = "col-wrapper">

<div class="c1" style = "width: 50%; font-size: 0.75em;">

### Euclidean Distance

`$ d(x, x') =\sqrt{\sum_{i=1}^n (x_i - x'_i)^2} $`

### Manhattan Distance

`$ d(x, x') = \sum_{i=1}^n |x_i - x'_i| $`

### Cosine Distance 

`$ d(x, x') = 1 - \frac{x \cdot x'}{||x|| \cdot ||x'||} $`

</div>

<div class="c2" style = "width: 50%; font-size: 0.75em;">

### Jaccard Distance (useful for categorical data!)

`$ d(x, x') = 1 - \frac{|x \cap x'|}{|x \cup x'|} $`

### Hamming Distance (useful for strings!)

`$ d(x, x') = \frac{1}{n} \sum_{i=1}^n x_i \neq x'_i $`

</div>
</div>

<!--s-->

## Clustering | Properties

<div style="font-size: 0.65em;">

To determine the distance between two points (required for clustering), we need to define a distance metric. A good distance metric has the following properties:

1. Non-negativity: $d(x, y) \geq 0$
    - Otherwise, the distance between two points could be negative.
    - If Bob is 5 years old and Alice is 10 years old, the distance between them could be -5 years.

2. Identity: $d(x, y) = 0$ if and only if $x = y$
    - Otherwise, the distance between two points could be zero even if they are different.
    - If Bob is 5 years old and Alice is 5 years old, the distance between them should be zero.

3. Symmetry: $d(x, y) = d(y, x)$
    - Otherwise, the distance between two points could be different depending on the order.
    - If the distance between Bob and Alice is 5 years, the distance between Alice and Bob should also be 5 years.

4. Triangle inequality: $d(x, y) + d(y, z) \geq d(x, z)$
    - Otherwise, the distance between two points could be shorter than the sum of the distances between intermediate points.
    - If the distance between Bob and Alice is 5 years and the distance between Alice and Charlie is 5 years, then the distance between Bob and Charlie should be at most 10 years. We don’t want shortcuts in our triangles.

</div>

<!--s-->

## Clustering Approaches

Okay, so we have a good understanding of distances. Now, let's talk about the different approaches to clustering data without labels using those distances. There are **many** clustering algorithms, but they can be broadly categorized into a few main approaches:


<img src="https://www.mdpi.com/sensors/sensors-23-06119/article_deploy/html/images/sensors-23-06119-g003.png" height="50%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Adnan, 2023</p>

<!--s-->

## Clustering Approaches

### Partitional Clustering
- Partitional clustering divides a dataset into $k$ clusters, where $k$ is a tunable hyperparameter.
- Example algorithm: K-Means

### Hierarchical Clustering
- Hierarchical clustering builds a hierarchy of clusters, which can be visualized as a dendrogram.
- Example algorithm: Agglomerative Clustering

### Density-Based Clustering
- Density-based clustering groups together points that are closely packed in the feature space.
- Example algorithm: DBSCAN

<!--s-->

<div class="header-slide">

# Partitional Clustering

</div>

<!--s-->

## Partitional Clustering | Introduction

Partitional clustering is the task of dividing a dataset into $k$ clusters, where $k$ is a hyperparameter. The goal is to minimize the intra-cluster distance and maximize the inter-cluster distance, subject to the constraint that each data point belongs to exactly one cluster.

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*vEsw12wO0KxvYne0m4Cr5w.png" height="50%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Yehoshua, 2023</p>

<!--s-->

## Partitional Clustering | K-Means

K-Means is a popular partitional clustering algorithm that partitions a dataset into $k$ clusters. The algorithm works as follows:

```text
1. Initialize $k$ cluster centroids randomly.
2. Assign each data point to the nearest cluster centroid.
3. Update the cluster centroids by taking the mean of the data points assigned to each cluster.
4. Repeat steps 2 and 3 until convergence.
    - Convergence occurs when the cluster centroids do not change significantly between iterations.
```


<img src="https://ben-tanen.com/assets/img/posts/kmeans-cluster.gif" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Tanen, 2016</p>

<!--s-->

## Partitional Clustering | K-Means

K-Means is a simple and efficient algorithm, but it has some limitations:

- The number of clusters $k$ must be specified in advance.
- The algorithm is sensitive to the initial cluster centroids.
- The algorithm may converge to a local minimum.

There are variations of K-Means that help address limitations (e.g., K-Means++, MiniBatch K-Means).

Specifying the number of clusters $k$ is a common challenge in clustering, but there are techniques to estimate an optimal $k$, such as the **elbow method** and the **silhouette score**. Neither of these techniques is perfect, but they can be informative under the right conditions.

<!--s-->

## Partitional Clustering | K-Means Elbow Method

The elbow method is a technique for estimating the number of clusters $k$ in a dataset. The idea is to plot the sum of squared distances between data points and their cluster centroids as a function of $k$, and look for an "elbow" in the plot.

```text
1. Run K-Means for different values of $k$.
2. For each value of $k$, calculate the sum of squared distances between data points and their cluster centroids.
3. Plot the sum of squared distances as a function of $k$.
4. Look for an "elbow" in the plot, where the rate of decrease in the sum of squared distances slows down.
5. The number of clusters $k$ at the elbow is a good estimate.
```

<img src="https://media.licdn.com/dms/image/D4D12AQF-yYtbzPvNFg/article-cover_image-shrink_600_2000/0/1682277078758?e=2147483647&v=beta&t=VhzheKDjy7bEcsYyrjql3NQAUcTaMBCTzhZWSVVSeNg" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Etemadi, 2020</p>

<!--s-->

## Partitional Clustering | K-Means Silhouette Score

The silhouette score measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The optimal silhouette score is 1, and the worst score is -1. When using the silhouette score to estimate the number of clusters $k$, we look for the value of $k$ that maximizes the silhouette score.

$$ s = \frac{b - a}{\max(a, b)} $$

Where: 
- $a$ is the mean distance between a sample and all other points in the same cluster.
- $b$ is the mean distance between a sample and all other points in the next nearest cluster.

Pseudocode for estimating $k$ using the silhouette score:
```text
1. Run K-Means for different values of $k$.
2. For each value of $k$, calculate the silhouette score.
3. Plot the silhouette score as a function of $k$.
4. Look for the value of $k$ that maximizes the silhouette score.
```

<!--s-->

<div class="header-slide">

# Hierarchical Clustering

</div>

<!--s-->

## Hierarchical Clustering | Introduction

Hierarchical clustering builds a hierarchy of clusters. The hierarchy can be visualized as a dendrogram, which shows the relationships between clusters at different levels of granularity.

### Agglomerative Clustering
- Start with each data point as a separate cluster, and merge clusters iteratively. AKA "bottom-up" clustering.

### Divisive Clustering
- Start with all data points in a single cluster, and split clusters iteratively. AKA "top-down" clustering.

<!--s-->

## Hierarchical Clustering | Agglomerative Clustering

Agglomerative clustering is a bottom-up approach to clustering that starts with each data point as a separate cluster, and merges clusters iteratively based on a linkage criterion. The algorithm works as follows:

```text
1. Start with each data point as a separate cluster.
2. Compute the distance between all pairs of clusters.
3. Merge the two closest clusters.
4. Repeat steps 2-3 until the desired number of clusters is reached.
```
<img src = "https://www.knime.com/sites/default/files/public/6-what-is-clustering.gif" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Hayasaka, 2021</p>

<!--s-->

## Hierarchical Clustering | Agglomerative Clustering Linkage

The distance between clusters can be computed using different linkage methods, such as:

- **Single linkage** (minimum distance between points in different clusters)
- **Complete linkage** (maximum distance between points in different clusters)
- **Average linkage** (average distance between points in different clusters)

The choice of linkage method can have a significant impact on the clustering result.

<img src = "https://www.knime.com/sites/default/files/public/6-what-is-clustering.gif" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Hayasaka, 2021</p>

<!--s-->

## Hierarchical Clustering | Agglomerative Clustering

Agglomerative clustering is a flexible and intuitive algorithm, but it has some limitations:

- The algorithm is sensitive to the choice of distance metric and linkage method.
- The algorithm has a time complexity of $O(n^3)$, which can be slow for large datasets.

<!--s-->

<div class="header-slide">

# Density-Based Clustering

</div>

<!--s-->

## Density-Based Clustering | Introduction

Density-based clustering groups together points that are closely packed in the feature space. The idea is to identify regions of high density and separate them from regions of low density.

One popular density-based clustering algorithm is DBSCAN.

<img src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*WunYbbKjzdXvw73a4Hd2ig.gif" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Seo, 2023</p>

<!--s-->

## Density-Based Clustering | DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that groups together points that are closely packed in the feature space. The algorithm works as follows:

```text
1. For a data point, determine the neighborhood of points within a specified radius $\epsilon$.
2. If the neighborhood contains at least $m$ points, mark the data point as a core point.
3. Expand the cluster by adding all reachable points to the cluster.
4. Repeat steps 1-3 until all points have been assigned to a cluster.
```

DBSCAN has two hyperparameters: $\epsilon$ (the radius of the neighborhood) and $m$ (the minimum number of points required to form a cluster). The algorithm is robust to noise and can identify clusters of arbitrary shape.

<!--s-->

## Density-Based Clustering | DBSCAN

DBSCAN is a powerful algorithm for clustering data, but it has some limitations:

- Sensitive to the choice of hyperparameters $\epsilon$ and $m$.
- May struggle with clusters of varying densities.


<!--s-->

## Summary

- Unsupervised machine learning is about finding patterns in $X$, without being given $y$.
- Clustering is a fundamental technique in unsupervised machine learning, with a wide range of applications.
    - **Partitional clustering** divides a dataset into $k$ clusters, with K-Means being a popular algorithm.
    - **Hierarchical clustering** builds a hierarchy of clusters, with agglomerative clustering being a common approach.
    - **Density-based clustering** groups together points that are closely packed in the feature space, with DBSCAN being a popular algorithm.

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how comfortable are you with topics like:

  1. K-Means
  2. Hierarchical Clustering
  3. DBSCAN

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>