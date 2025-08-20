-----

# Rare Label Detection for Email Classification

This project introduces a **supporting algorithm** to accelerate the dataset building process for enterprise email classification.
When constructing a dataset, a common issue is **label imbalance**: some classes dominate up to 90% of the data.
If annotators are required to label everything, the majority labels consume too much time, while the **rare classes** (which are more valuable for balancing the dataset) remain hidden.

Our algorithm helps **surface rare/noisy samples** faster, reducing the need to label redundant majority classes.
Instead of labeling 100% of a dataset, annotators may only need to label \~10% to discover the rare labels.

-----

## Core Idea

1.  **Embedding**: Emails are embedded into high-dimensional vectors (300–1024 dimensions depending on the embedding model).
2.  **Dimensionality Reduction (UMAP/TSNE)**: Reduce embeddings to lower dimensions (e.g., 16–32).
3.  **Clustering (HDBSCAN)**: Identify dense clusters of common labels.
      - Noise (`-1` labels) and sparse samples are discarded.
      - Large clusters (≥1000 samples) represent "common" classes.
4.  **Isolation Forest per Cluster**: Each cluster trains a one-class `IsolationForest` model.
5.  **Sequential Rare Detection**: New data is passed through all cluster models sequentially.
      - If a sample does not belong to any cluster → it is flagged as **rare/noisy**.
      - These rare samples are surfaced for annotators first.

-----

## Note on Dataset

**Important:**
This repo uses **demo data** only (e.g., public datasets like HuggingFace’s *fancyzhx/dbpedia_14 dataset*).
The real enterprise dataset used in production **cannot be shared**.

  - The algorithm is most effective when applied to **short, pre-processed text**.
  - **Why?**
      - UMAP + HDBSCAN work best on moderate-dimensional embeddings.
      - If embeddings have very high dimensions (300–1024) and we reduce aggressively (down to 16–32), **a large amount of semantic information is lost**.
      - This loss is especially critical for **long, complex texts**.
  - Therefore, the pipeline is recommended for **short, pre-processed texts**, where dimensionality reduction preserves more information.
  - For **long text**, the reduced embeddings tend to lose too much signal, making HDBSCAN less accurate.
  - In practice, this algorithm is particularly useful for **extracting noise/rare samples** from datasets with relatively simple text.

If you want to learn how to fine-tune preprocessing to make text more suitable for this algorithm, please check my guide here:
[abc.git](https://www.google.com/search?q=https://abc.git)

---

## Practical Notes from Experiments

**Based on my experiments:**  
- When applying **HDBSCAN** on reduced embeddings, using a latent dimensionality of **16–32 (or even smaller)** gives more **stable and accurate clusters**.  
- Higher dimensions after reduction tend to keep noise, making **unsupervised clustering less clean**.  
- For practical unsupervised text clustering:  
  - **Short, pre-processed text** + **low-dimensional embeddings (≤32)** work best.  
  - This ensures HDBSCAN can effectively separate **dense regions** from **noise**.  

---

## Pipeline Diagram
```text
         ┌──────────────┐
         │  New Emails  │
         └──────┬───────┘
                │
                ▼
         ┌──────────────┐
         │  Embedding   │
         └──────┬───────┘
                │
                ▼
         ┌──────────────┐
         │    UMAP      │
         └──────┬───────┘
                │
                ▼
         ┌──────────────┐
         │ Sequential   │
         │ IsolationF.  │
         └──────┬───────┘
                │
   ┌────────────┴────────────┐
   ▼                         ▼
Common Cluster          Rare Candidate
 (ignore)                  (label)
````
-----

## Repository Structure

```
├── data/                     # Datasets (raw + processed, not public)
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/                # Demo + GPU accelerated version
│   ├── 01_demo_pipeline.ipynb
├── src/
│   ├── __init__.py
│   ├── clustering.py
│   ├── config.py
│   ├── data_loader.py
│   ├── embedding.py
│   ├── iforest_detector.py
│   ├── rare_detection.py
│   └── reduce_dim.py
├── scripts/
│   ├── train_pipeline.py
│   └── detect_rare.py
├── tests/
│   ├── test_clustering.py
│   ├── test_detection.py
│   └── test_embedding.py
├── requirements.txt
├── README.md
└── .gitignore
```

-----

## CPU vs GPU

  - **Default pipeline (src/)** → Uses **CPU-based libraries**:

      - `umap-learn`
      - `hdbscan`
      - `scikit-learn`

  - **GPU option (notebooks/demo\_gpu.ipynb)** → Uses **RAPIDS cuML**:

      - `cuml.UMAP` instead of `umap-learn`
      - `cuml.HDBSCAN` instead of `hdbscan`
      - `cuml.TSNE` instead of sklearn TSNE

For large datasets, switching to GPU (RAPIDS) gives significant acceleration.
Check the notebook example to see how to replace CPU calls with GPU equivalents.

-----

## Installation

1.  Clone the repo:

    ```bash
    git clone https://github.com/yourname/rare-label-detection.git
    cd rare-label-detection
    ```

2.  Install requirements:

    ```bash
    pip install -r requirements.txt
    ```

3.  For GPU acceleration (optional):

    ```bash
    conda install -c rapidsai -c nvidia -c conda-forge cuml=24.02 python=3.10 cudatoolkit=11.8
    ```

-----

## Usage

1.  **Train the pipeline**

    ```bash
    python scripts/train_pipeline.py
    ```

    This will:

      - Embed training data
      - Reduce dimensions with UMAP
      - Cluster with HDBSCAN
      - Train IsolationForest models per cluster
      - Models are saved under `models/`.

2.  **Detect rare samples in new data**

    ```bash
    python scripts/detect_rare.py
    ```

    This will:

      - Embed new emails
      - Pass them through each cluster’s IsolationForest
      - Save rare samples to `data/rare_candidates.csv`

3.  **Run unit tests**

    ```bash
    pytest tests/ -v
    ```

-----

## Example Workflow

1.  Run `train_pipeline.py` on a dataset of common labels.
2.  When new data arrives, run `detect_rare.py` to extract rare candidates.
3.  Annotators only need to label these rare samples.
4.  The final dataset is more balanced and less time-consuming to build.

-----

## Key Benefits

  - **Time saving**: Annotators focus only on \~10% rare data.
  - **Reduced imbalance**: Prevents dominant labels from overwhelming the dataset.
  - **Scalable**: Switch from CPU to GPU easily with RAPIDS cuML.
  - **Modular design**: Embedding, clustering, and detection are independent and testable.

-----

## Dataset

Company data is not public. For demonstration, public datasets can be used.

-----



