# Clustering Algorithms from Scratch

Implemented k-means and hierarchical clustering using only NumPy.
No sklearn clustering functions used.

## Dataset
Spiral dataset with 3 intertwined spiral clusters and around 300 2D points.
A notoriously difficult dataset for distance-based clustering.

## What's Implemented
- K-means clustering with 10 random restarts
- Hierarchical clustering with 4 linkage methods:
  - Single linkage
  - Complete linkage
  - Average linkage
  - Centroid linkage

## Evaluation Metrics
- SSE: Sum of Squared Error (intrinsic, lower is better)
- Rand Index (extrinsic, higher is better)

## Tech
Python, NumPy, matplotlib