# Genre Classification Using DistilBERT on IMDb and Movie Metadata Datasets

## 📖 Introduction

### Context  
Genre classification in movies is essential for improving recommendation systems and organizing metadata. Accurate classification enhances user experience and facilitates content discovery.

For this project, two datasets were used:  
- **IMDb Dataset**: Previously used in research, providing a robust source for training and testing.  
- **Movies Dataset**: A Kaggle dataset offering additional metadata for feature extraction and prediction.

### Objective  
The goal is to predict movie genres using **DistilBERT** and evaluate its performance on:  
1. **IMDb Dataset** (Single-label Classification)  
   - **Data Source**: [IMDb Genre Classification Dataset](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)  
   - **Size**: 54,214 training samples and 54,200 testing samples  
   - **Code**: [Colab Notebook](https://colab.research.google.com/drive/1jTTG-PPA_0yH9znWCdHawzNwMFO3vRPU?authuser=1#scrollTo=dMBmcZUAVKDi)  

2. **Movies Dataset** (Multi-label Classification)  
   - **Data Source**: [Movies Dataset](https://www.kaggle.com/datasets/bharatnatrayn/movies-dataset-for-feature-extracion-prediction/data)  
   - **Size**: 6,929 training samples and 1,733 testing samples  
   - **Code**: [Colab Notebook](https://colab.research.google.com/drive/1t6AGE1P4gscueBbgA3tb3pr1qqhFroIX?authuser=1#scrollTo=gOdCX-io048w)  

### Why DistilBERT?  
DistilBERT is a lightweight, efficient model derived from BERT, offering high accuracy with reduced computational cost. It is ideal for tasks involving textual data.

---

## 🛠️ Methodology

### 2.1 Preprocessing  
- **IMDb Dataset**: Cleaned text, removed special characters, and balanced genre distributions.  
- **Movies Dataset**: Processed metadata, applied lowercasing, removed duplicates, and encoded multi-genre labels with `MultiLabelBinarizer`.  
- **Text Preparation**:  
  - Combined `title` and `description` fields for richer textual context.  
  - Tokenized inputs using DistilBERT tokenizer.  
  - Applied padding and truncation for consistent input lengths compatible with the model.

### 2.2 Model Development  
- **Model Selection**:  
  Used DistilBERT for its efficiency and pre-trained textual understanding, ideal for multi-class and multi-label classification.  
- **Architecture**:  
  - Added a classification layer with **Softmax** activation for single-label tasks.  
  - Adjusted to **Sigmoid** activation for multi-label classification.  
- **Training Strategy**:  
  - Optimizer: AdamW  
  - Learning Rates: `7e-5` for Movies dataset and `5e-5` for IMDb dataset.  
  - Regularization: Dropout layers to mitigate overfitting.

### 2.3 Hyperparameter Tuning and Augmentations  
- Conducted experiments with different batch sizes, epochs, and max sequence lengths.  
- Found optimal settings:  
  - **Batch Size**: 16  
  - **Max Sequence Length**: 256  
- Applied token-level augmentations (e.g., word dropout, synonym replacement) for the IMDb dataset to improve generalization.

### 2.4 Insights and Evaluation  
- Metrics: Precision, Recall, F1-score, and Accuracy.  
- Tools: Confusion matrices to analyze performance variability across genres.  
- **Key Observations**:  
  - Genre overlaps were challenging.  
  - Improved predictions through better preprocessing and augmentations.

### 2.5 Limitations and Future Work  
- **Current Limitations**:  
  - Nuanced genre overlaps due to textual ambiguity.  
- **Future Improvements**:  
  - Use attention visualizations for better interpretability.  
  - Fine-tune with external datasets to enhance genre diversity.

---

## 📊 Results and Analysis

### IMDb Dataset (Single-label Classification)  
A key achievement was creating a labeled CSV file associating each movie with a predicted genre.  
- **CSV Output**: [Download Labeled Data](https://drive.google.com/file/d/1KXnt5LE2_oRXrcEA5EyRwig10nIJNkDo/view?usp=sharing)  

Example:  
| Movie                                 | Predicted Genre |
|---------------------------------------|-----------------|
| Batman Arkham Origins-Blackgate (2013)| Action          |

### Movies Dataset (Multi-label Classification)  
**Performance Metrics**:  
- **Accuracy**: 31%  
- **Precision (Weighted Avg.)**: 77%  
- **Recall (Weighted Avg.)**: 56%  
- **F1-score (Weighted Avg.)**: 62%  

**Key Observations**:  
- Smaller dataset size impacted generalization.  
- Multi-label datasets introduce class imbalance and challenges for accurate predictions.  

---

## 📌 Conclusion

- **IMDb Dataset**:  
  - Delivered practical outputs in the form of labeled CSV files.  
  - Highlighted potential for real-world applications like recommendation systems and metadata organization.  
- **Movies Dataset**:  
  - Demonstrated the complexity of multi-label classification tasks.  
  - Achieved a weighted F1-score of 62%, highlighting the model’s strengths despite dataset challenges.  

These results underscore the impact of dataset characteristics and training configurations on model performance.

---

## 📂 Repository Content

- Full project code and datasets: [Google Drive Link](https://drive.google.com/drive/u/1/folders/1e-KouzMGkLxsjX0OmRwQgCilgcroNl08)

---

## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/genre-classification
   cd genre-classification

2. Install dependencies:
   ```bash
   pip install -r requirements.txt



3. Run the training script:
   ```bash

   python train.py
   
3.Evaluate the model:
   ```bash

   
   python evaluate.py
