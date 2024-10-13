Here's a paraphrase of your document classification project description:

---

# Document Classification Using CNN

This repository implements a document classification system utilizing convolutional neural networks (CNN) in Keras. The architecture consists of three main components:

1. **Word Embedding:** This represents words in a distributed manner, where words with similar meanings (based on usage) share similar representations.
2. **Convolutional Model:** A feature extraction model that learns to identify important features from documents represented through word embeddings.
3. **Fully Connected Model:** This interprets the extracted features to produce a predictive output.

![Architecture Diagram]
## Dataset
The project uses the BBC News dataset, which can be downloaded from [this link](http://mlg.ucd.ie/datasets/bbc.html). The dataset contains five well-distributed categories: Business, Technology, Sports, Entertainment, and Politics, ensuring there is no class imbalance. Each document's TF-IDF vectors are transformed into corresponding 2D vectors using t-SNE for visualization in a 2D space.

![2D Visualization

## Process
Following visualization, the dataset is converted into a CSV file with two columns (documents and labels), with labels ranging from 1 to 5. A standard cleaning process is applied, which includes removing stopwords, converting words to lowercase, eliminating punctuation, and removing words shorter than two characters. The data is then divided into training and testing sets. The documents in these sets are vectorized using one of three techniques:

1. Keras embeddings (embedding layer from the Keras deep learning library)
2. Word2Vec embeddings
3. GloVe vectors

Hyperparameters are tuned on a validation set to achieve optimal results for each method, with comparisons made. The hyperparameters adjusted include:

1. Number of filters
2. Kernel size
3. Dropout units
4. Dense layer units
5. Number of epochs (with early stopping applied in some cases)
6. Activation and loss functions

The best settings, determined through multiple iterations of trial and error, are documented in the Jupyter notebook.

## Results
The model trained using Keras embeddings achieved the highest validation accuracy of 96.61%, followed by GloVe vectors at 95.13%, and Word2Vec embeddings at 89.19%. Below are the confusion matrices for a detailed representation of the results.

**Confusion Matrix for Word2Vec Vectors:**
![Word2Vec Confusion Matrix]
**Confusion Matrix for GloVe Vectors:**
![GloVe Confusion Matrix]
**Confusion Matrix for Keras Embeddings:**
![Keras Embeddings Confusion Matrix](

--- 

Let me know if you need any further adjustments!