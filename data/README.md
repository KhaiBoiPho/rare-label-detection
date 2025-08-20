# Rare Label Detection for Accelerated Data Labeling

## Overview
This project focuses on **rare label detection** in highly imbalanced multi-class datasets.  
The goal is to **reduce manual labeling time** by automatically filtering incoming data so that labelers only review samples that are likely to belong to **rare or underrepresented classes**.  
By doing so, we avoid wasting resources on common, majority classes and rapidly increase labeled data for rare categories.

---

## Motivation
In real-world datasets, certain labels dominate the distribution while others occur only occasionally.  
Training a model without addressing this imbalance can lead to **poor performance on rare classes**.  
Additionally, labeling large volumes of majority-class data is costly and time-consuming.

Our solution:
- **Preserve** all rare and target classes.
- **Downsample** frequent classes according to predefined retention ratios.
- **Deploy an anomaly/rare detection model** to screen new data before labeling.

---

## Data Retention Strategy
The dataset is split into groups with different retention ratios:

1. **Fully retained (100%)** — rare, high-value target labels:
   - `company` (0)
   - `educational_institution` (1)
   - `artist` (2)
   - `athlete` (3)
   - `office_holder` (4)

2. **Partially retained** — moderately frequent labels:
   - `mean_of_transportation`: **90%**
   - `building`, `natural_place`: **80%**

3. **Strongly downsampled** — overly frequent labels:
   - All other labels: **0.1%**

This ensures the dataset reflects the real-world imbalance but prioritizes rare label representation.

---

## Workflow
1. **Initial Dataset Split**  
   Apply the retention strategy to the existing dataset.

2. **Model Training**  
   Train an outlier/rare-label detection model (e.g., Isolation Forest, Local Outlier Factor, Autoencoder) using embeddings.

3. **Incoming Data Screening**  
   - Pass new data through the detection model.
   - Keep only samples flagged as *potentially rare*.
   - Discard most samples likely belonging to majority classes.

4. **Manual Labeling**  
   Labelers focus solely on the filtered set, speeding up annotation.

5. **Iterative Model Update**  
   Add newly labeled rare samples to the dataset and retrain the detection model.

---

## Benefits
- **Faster labeling**: Reduces the volume of data humans must inspect.
- **Higher rare-label coverage**: Maximizes the growth of rare class samples.
- **Lower costs**: Avoids wasting time on majority-class data.
- **Better model performance**: Balanced data leads to improved accuracy for rare labels.

---

## Example Use Case
Imagine a dataset with 20 classes where `company`, `artist`, and `athlete` each have fewer than 500 samples, while `person` has over 100,000.  
Without filtering, annotators would spend most of their time labeling `person` entries.  
With this pipeline, the annotation team focuses almost entirely on rare-class candidates.

---

## Future Improvements
- Adaptive retention ratios based on evolving dataset statistics.
- Multi-model ensemble for rare label detection.
- Integration with active learning loops for optimal sample selection.
