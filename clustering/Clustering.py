"""
Name - Kritika Joshi
Programming Assignment 3 - Clustering on Spiral Dataset
Implements:
  - Task 1: Plot the dataset
  - Task 2: K-Means (from scratch, k=3, 10 runs)
  - Task 3: Hierarchical Clustering (single, complete, average, centroid linkage)
No library functions for k-means or hierarchical clustering are used.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# 0. LOAD DATA

df = pd.read_csv("spiral-dataset.csv",
                 sep="\t", header=None, names=["X", "Y", "Label"])
X_data = df[["X", "Y"]].values          # shape (312, 2)
true_labels = df["Label"].values         # 1, 2, or 3

N = len(X_data)
print(f"Dataset loaded: {N} points, labels = {np.unique(true_labels)}")

os.makedirs("outputs", exist_ok=True)

COLORS = ["blue", "red", "green"]
MARKER = "o"

# HELPER FUNCTIONS

def euclidean(a, b):
    """Euclidean distance between two 1-D vectors."""
    return np.sqrt(np.sum((a - b) ** 2))

def compute_sse(data, labels, centroids):
    """Intrinsic metric: Sum of Squared Errors."""
    sse = 0.0
    for i, point in enumerate(data):
        c = int(labels[i])
        sse += np.sum((point - centroids[c]) ** 2)
    return sse

def compute_rand_index(true_labels, pred_labels):
    """
    Extrinsic metric: Rand Index.
    RI = (TP + TN) / C(n, 2)
    where TP = pairs in same cluster in both, TN = pairs in different clusters in both.
    """
    n = len(true_labels)
    tp_tn = 0
    total_pairs = n * (n - 1) // 2
    for i in range(n):
        for j in range(i + 1, n):
            same_true = (true_labels[i] == true_labels[j])
            same_pred = (pred_labels[i] == pred_labels[j])
            if same_true == same_pred:
                tp_tn += 1
    return tp_tn / total_pairs



# TASK 1: PLOT THE DATASET

def task1_plot(data, labels):
    fig, ax = plt.subplots(figsize=(7, 7))
    label_vals = sorted(np.unique(labels))
    color_map = {label_vals[0]: "green", label_vals[1]: "blue", label_vals[2]: "red"}
    for lv in label_vals:
        mask = labels == lv
        ax.scatter(data[mask, 0], data[mask, 1],
                   c=color_map[lv], s=20, label=f"Cluster {lv}")
    ax.set_title("Figure 1: Spiral Dataset (True Labels)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    plt.tight_layout()
    plt.savefig("task1_spiral_plot.png", dpi=150)
    plt.close()
    print("Task 1: Saved task1_spiral_plot.png")

task1_plot(X_data, true_labels)


# TASK 2: K-MEANS (from scratch)

def kmeans(data, k, seed=None):
    """
    K-Means clustering from scratch.
    Returns: (cluster_labels, centroids, iterations)
    """
    rng = np.random.default_rng(seed)
    N = len(data)

    # Random initialization of k centroids from the data points
    init_idx = rng.choice(N, size=k, replace=False)
    centroids = data[init_idx].copy().astype(float)

    labels = np.zeros(N, dtype=int)

    for iteration in range(1000):  # max iterations safeguard
        # Assignment step
        new_labels = np.zeros(N, dtype=int)
        for i in range(N):
            dists = [euclidean(data[i], centroids[c]) for c in range(k)]
            new_labels[i] = int(np.argmin(dists))

        # Check convergence
        if np.array_equal(new_labels, labels) and iteration > 0:
            break
        labels = new_labels

        # Update step 
        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            members = data[labels == c]
            if len(members) > 0:
                new_centroids[c] = members.mean(axis=0)
            else:
                # Reinitialize empty centroid randomly
                new_centroids[c] = data[rng.integers(N)]
        centroids = new_centroids

    return labels, centroids, iteration + 1


def task2_run():
    k = 3
    num_runs = 10
    results = []

    print("\n" + "="*60)
    print("TASK 2: K-Means Clustering (k=3, 10 runs)")
    print("="*60)

    for run in range(num_runs):
        labels, centroids, iters = kmeans(X_data, k, seed=run * 42)
        sse = compute_sse(X_data, labels, centroids)
        ri = compute_rand_index(true_labels, labels + 1)  # shift labels to 1-based
        results.append((sse, ri, labels, centroids, iters))
        print(f"  Run {run+1:2d}: SSE = {sse:10.4f} | RI = {ri:.6f} | Converged in {iters} iters")

    # Best by SSE
    best_sse_run = min(results, key=lambda r: r[0])
    # Best by RI
    best_ri_run  = max(results, key=lambda r: r[1])

    print(f"\n  Best SSE = {best_sse_run[0]:.4f}  (RI = {best_sse_run[1]:.6f})")
    print(f"  Best RI  = {best_ri_run[1]:.6f}   (SSE = {best_ri_run[0]:.4f})")

    # Plot best SSE result
    best_labels, best_centroids = best_sse_run[2], best_sse_run[3]
    fig, ax = plt.subplots(figsize=(7, 7))
    for c in range(k):
        mask = best_labels == c
        ax.scatter(X_data[mask, 0], X_data[mask, 1],
                   c=COLORS[c], s=20, label=f"Cluster {c+1}")
        ax.scatter(best_centroids[c, 0], best_centroids[c, 1],
                   c=COLORS[c], s=200, marker="*", edgecolors="black", linewidths=0.8)
    ax.set_title(f"Task 2: K-Means (Best Run)\nSSE = {best_sse_run[0]:.2f}  |  RI = {best_sse_run[1]:.4f}")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.legend()
    plt.tight_layout()
    plt.savefig("task2_kmeans_best.png", dpi=150)
    plt.close()
    print("  Saved: task2_kmeans_best.png")
    return results

kmeans_results = task2_run()


# TASK 3: HIERARCHICAL CLUSTERING (from scratch)


def build_initial_distance_matrix(data):
    """Build full pairwise Euclidean distance matrix."""
    N = len(data)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            d = euclidean(data[i], data[j])
            D[i, j] = d
            D[j, i] = d
    return D


def hierarchical_clustering(data, linkage="single", k=3):
    """
    Agglomerative hierarchical clustering from scratch.
    Supports: single, complete, average, centroid linkage.
    Returns cluster labels (0-indexed) for k clusters.
    """
    N = len(data)

    # Each point starts as its own cluster
    # clusters: dict {cluster_id: list of original point indices}
    clusters = {i: [i] for i in range(N)}

    # Pre-compute pairwise distances
    print(f"    Building {N}x{N} distance matrix...")
    D = build_initial_distance_matrix(data)

    # Track cluster order for merging
    active = list(range(N))

    # Compute inter-cluster distance
    def cluster_dist(cid_a, cid_b):
        pts_a = clusters[cid_a]
        pts_b = clusters[cid_b]
        dists = [D[i, j] for i in pts_a for j in pts_b]
        if linkage == "single":
            return min(dists)
        elif linkage == "complete":
            return max(dists)
        elif linkage == "average":
            return sum(dists) / len(dists)
        elif linkage == "centroid":
            ca = data[pts_a].mean(axis=0)
            cb = data[pts_b].mean(axis=0)
            return euclidean(ca, cb)

    merge_count = 0
    total_merges = N - k
    print(f"    Performing {total_merges} merges (to reach k={k})...")

    new_id = N  # next available cluster id
    while len(active) > k:
        # Find closest pair of active clusters
        best_dist = np.inf
        best_i = best_j = -1
        n_active = len(active)
        for ii in range(n_active):
            for jj in range(ii+1, n_active):
                d = cluster_dist(active[ii], active[jj])
                if d < best_dist:
                    best_dist = d
                    best_i, best_j = ii, jj

        # Merge best_i and best_j into new_id
        cid_a, cid_b = active[best_i], active[best_j]
        clusters[new_id] = clusters[cid_a] + clusters[cid_b]
        active.remove(cid_a)
        active.remove(cid_b)
        active.append(new_id)
        del clusters[cid_a]
        del clusters[cid_b]
        new_id += 1

        merge_count += 1
        if merge_count % 50 == 0:
            print(f"      ... {merge_count}/{total_merges} merges done")

    # Assign labels
    labels = np.zeros(N, dtype=int)
    for cluster_idx, cid in enumerate(active):
        for pt_idx in clusters[cid]:
            labels[pt_idx] = cluster_idx

    return labels


def compute_centroids_from_labels(data, labels, k):
    centroids = np.array([data[labels == c].mean(axis=0) for c in range(k)])
    return centroids


def task3_run():
    k = 3
    linkages = ["single", "complete", "average", "centroid"]
    results = {}

    print("\n" + "="*60)
    print("TASK 3: Hierarchical Clustering (4 linkage methods)")
    print("="*60)

    for lnk in linkages:
        print(f"\n  [{lnk.upper()} LINKAGE]")
        labels = hierarchical_clustering(X_data, linkage=lnk, k=k)
        centroids = compute_centroids_from_labels(X_data, labels, k)
        sse = compute_sse(X_data, labels, centroids)
        ri = compute_rand_index(true_labels, labels + 1)
        results[lnk] = (sse, ri, labels, centroids)
        print(f"    SSE = {sse:.4f}  |  RI = {ri:.6f}")

    # Summary table
    print("\n" + "-"*50)
    print(f"  {'Linkage':<12} {'SSE':>12} {'RI':>12}")
    print("-"*50)
    for lnk in linkages:
        s, r, _, _ = results[lnk]
        print(f"  {lnk:<12} {s:>12.4f} {r:>12.6f}")
    print("-"*50)

    best_sse_lnk = min(results, key=lambda l: results[l][0])
    best_ri_lnk  = max(results, key=lambda l: results[l][1])
    print(f"\n  Best SSE: {best_sse_lnk.upper()}  (SSE={results[best_sse_lnk][0]:.4f})")
    print(f"  Best RI:  {best_ri_lnk.upper()}   (RI={results[best_ri_lnk][1]:.6f})")

    # Plot all 4 results in a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    for ax, lnk in zip(axes.flat, linkages):
        sse, ri, labels, centroids = results[lnk]
        for c in range(k):
            mask = labels == c
            ax.scatter(X_data[mask, 0], X_data[mask, 1],
                       c=COLORS[c], s=18, label=f"Cluster {c+1}")
            ax.scatter(centroids[c, 0], centroids[c, 1],
                       c=COLORS[c], s=180, marker="*", edgecolors="black", linewidths=0.8)
        ax.set_title(f"{lnk.capitalize()} Linkage\nSSE={sse:.2f}  RI={ri:.4f}")
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.legend(fontsize=8)
    plt.suptitle("Task 3: Hierarchical Clustering Results", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("task3_hierarchical_all.png", dpi=150)
    plt.close()
    print("  Saved: task3_hierarchical_all.png")

    return results

hier_results = task3_run()

print("\n" + "="*60)
print("ALL TASKS COMPLETE")
print("="*60)

"""
Task 3.e: Best Method Analysis

Best SSE = COMPLETE LINKAGE (SSE = 13004.37)
Best RI = SINGLE LINKAGE   (RI  = 1.000000) (Perfect score)

The spiral dataset is non-convex (curved arms), so K-Means and most
hierarchical methods struggle. Single linkage follows the 'chain' of
nearby points along each spiral arm, perfectly recovering all 3 clusters
(RI = 1.0 = every pair of points correctly grouped). However, its SSE is
high because spiral centroids are geometrically far from many members.

Complete linkage produces more compact, balanced clusters (lowest SSE)
but misclassifies many spiral points where the arms interleave.

Overall winner: SINGLE LINKAGE — a perfect RI of 1.0 means it
correctly clusters every single one of the 312 data points.
"""