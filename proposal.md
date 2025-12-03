---
layout: page
title: ML Midterm Report
permalink: /proposal/
---
<div class="hero-section">
	<h1>Proposal & Final Report</h1>
	<blockquote>Political Bias Detection in News Articles</blockquote>
</div>

## Introduction and Background

News media serve as a primary channel through which the public learns about politics. However, news articles often reflect subtle political bias through word choice, framing, selection, and source attribution. Detecting such bias is critical for improving transparency and helping readers contextualize coverage. Traditionally, political bias in news has been modeled as a three-way classification task—left, center, right—aligned with known ideological leanings of outlets. Early computational approaches relied on sentiment analysis and lexical statistics but often failed to capture more nuanced forms of bias, such as selective topic coverage or framing strategies [1].

Recent NLP advances have improved political bias detection by combining textual features, metadata, and embeddings [2]. Sentence-BERT [3] enables semantic comparison, while contrastive learning [5] helps capture subtle ideological framing. Social network–aware models like Retweet-BERT [5] enhance predictions by incorporating diffusion patterns. Large language models have also been applied to framing detection, revealing nuanced narrative biases and misinformation cues in headlines and article text [6], [7]. Most recently, Rönnback et al. [8] introduced a large-scale AI-powered bias detector that not only outperforms existing models but also provides interpretable explanations of why outlets are categorized in a particular way.

### Dataset Description

We will utilize a combination of datasets to ensure robustness in our developed methods.

- **BASIL** [9]: Provides event-level stance annotations for articles from Fox (Right), NYT (Center) and HuffPost (left), enabling fine-grained framing analysis.
- **AllSides (Kaggle)** [10]: Large corpus of articles with outlet-level bias labels, for weak supervision and baseline model training
- **GDELT Project [11]**: Massive, real-time global database of news, for outlet-level aggregation to proxy coverage bias
- **Media Bias Fact Check (MBFC) [12]** : Outlet-level bias and factuality ratings across thousands of news domains. 

---

## Problem Definition

### Problem Statement
Current data-driven approaches for political bias detection often focus on superficial indicators such as sensational headlines or biased wording. More subtle forms of bias, including selective topic coverage, underreporting, or framing differences across events, remain largely undetected. This motivates the need for models that not only classify articles by bias (left, center, right) but also provide interpretable insights into how bias manifests across outlets and events. 

### Motivation
Accurate political bias detection has applications in media literacy, fact-checking, and reducing information asymmetry. By combining article-level features, outlet-level aggregations, and advanced NLP models, we aim to capture both overt and subtle bias, going beyond simplistic classification.

---

## Methods

### Data Preprocessing Methods

1. Deduplication, Text Normalization, Tokenization & Stopword Handling, Lemmatization/Stemming, Handling Noise and Class Balance 
2. **TF-IDF**: Idenitfy keywords that are dense within a group of articles but uncommon across all documents. Provides an initial signal for clustering articles by orientation. 
3. **Contextual Embeddings**: BERT produced vector representations of articles that preserve semantic meaning, forming features for our main analysis. 
4. **DAPT (Domain Adaptive Pre-training)**: Fine-tune BERT to adapt to political data to capture vocabulary and discourse nuances. 

### Machine Learning Implementations Overview

#### Supervised

1. Gaussian NB: Sentence BERT (all-MiniLM-L6-v2)  embeddings followed by PCA to recover 91% variance
2. Fine-tuned RoBERTa model trained on labeled political articles to directly predict orientation. Paper by Jiang mentions motivations for political ideology detection in news articles using BERT [5]
3. MLP with TF-IDF: Assigns weights to words based on their frequency in a document relative to their frequency across the entire corpus.

#### Unsupervised
We implemented several **clustering** techniques:
- K-Means & Mini-Batch K-Means
- Gaussian Mixture Model (GMM) and Bayesian Gaussian Mixture Model
- Density-based methods: DBSCAN, HDBSCAN, OPTICS, & Mean Shift
- Hierarchical Clustering: Agglomerative Clustering & Birch
- Spectral Clustering

---

## Data Processing
#### Google Colab Notebook: [Data Pre-processing](https://colab.research.google.com/drive/1F6SQGwr31mf19EBiD0cALkvtX4KvWLsX#scrollTo=AhoKZqufcb8N)

### Data **Process**esing Method Implemented
Our motivation was to gain insights into the data structure, quality, and potential biases before performing more complex analysis. Complex analysis included cleaning the data by filtering out bad data and short texts to clarify our bias distributions. We used methods of Text Embedding and TF-IDF in our data preprocessing. Understanding these distributions helped us to make decisions about further data cleaning and modeling. Our general preprocessing pipeline looked as follows:  

#### 1. Loading the Dataset 
**Process**: We downloaded the AllSides dataset from a kagglehub library and then used pandas.read_csv to load the data into a DataFrame. This allowed us to process the data within a Google Colab environment using Python libraries like pandas. The relevant columns "Text" and "Bias" were selected. The categorical "Bias" labels were mapped to numerical integers (0-4) for easier processing, with 0 = “highly conservative”, 1 = “moderately conservative”, 2 = “centrist / balanced”, 3 = “moderately liberal”, and 4 = “highly liberal”. 

**Result**: Created a DataFrame with 8112 rows and 5 columns.

#### 2. Initial Visualization and Analysis of Distribution 
**Process**: We visualized the distribution of text lengths using histograms, and generated count plots using seaborn to show the frequency of different bias categories and sources. The distribution of the political bias labels was visualized using a bar plot. We also did some rudimentary binning to show the count of each type of bias category for each source.  

**Result**: The data displayed a high count of very short text entries (Figure 1) and the distribution across bias categories and sources. The political bias labels bar plot showed an imbalance in the dataset with a higher number of "highly conservative" articles. (Figure 2) We visualized the spread of article sources in Figure 3. From our bin counting of each type of content for each source, we also observed that each source only produces one type of political bias content. (Figure 4)

![Figure 1](/assets/images/image.png){: .small-img }
<p class="caption"><strong>Figure 1:</strong> Distribution of text lengths</p>

![Figure 2](/assets/images/image2.png){: .small-img }
<p class="caption"><strong>Figure 2:</strong> Dataset spread per category</p>

![Figure 3](/assets/images/image3.png){: .small-img }
<p class="caption"><strong>Figure 3:</strong> Spread per source </p>

![Figure 4](/assets/images/image4.png){: .small-img }
<p class="caption"><strong>Figure 4:</strong> Count of each type of content per source</p>


#### 3. Filtering Out Bad Data: 
To improve the quality of the dataset, we removed entries that are unlikely to be informative for text analysis, such as those with minimal text content.  

**Process**: Created a new column text_length to store the length of the 'Text' column and then filtered the DataFrame to keep only rows where text_length was 50 or greater. 

**Result**: This reduced the dataset to 6209 rows, focusing on more substantial articles.

#### 4. Cleaning the Data: 
To prepare the text data for natural language processing tasks by reducing noise, standardizing terms, and extracting meaningful components. This improves the effectiveness of techniques like TF-IDF vectorization and clustering. 

**Process**: The ftfy library was used to fix text encoding issues. spaCy was employed to perform tokenization and lemmatization, converting words to their base forms and removing stop words. 

**Result**: Standardized the text and extracted keywords for further analysis. 

#### 5. Normalization: 
Textual data often consists of a lot of words that mean the same thing but are spelled differently, that is why we need to normalize the text. This is a way of reducing noise and improving the reliability of text analysis. 

**Process**: Lowercase conversion, punctuation removal, contraction expansion, standardized spelling using SpaCY.  

#### 6. Lemmatization 
Lemmatization is the process of reducing words to their base or dictionary form (lemma) while considering their context and part of speech. For example, “running”, “runs”, and “ran” are all reduced to “run”. Unlike stemming, which simply chops off word endings, lemmatization uses linguistic knowledge from a combination of dictionary lookups for irregular words (e.g., mice → mouse, was → be) and morphological rules based on each word’s part of speech to ensure the resulting word is valid and meaningful.  

**Process**: We did this using SpaCy’s rule-based lemmatizer from the en_core_web_sm model. 

**Result**: This process reduced vocabulary size and grouped different word forms under a single representative lemma, improving the quality of the TF-IDF representation used for modeling. 

#### 7. TF-IDF 
This is a statistical measure used to represent text as numerical features, reflecting how important a word is within a document relative to the entire corpus. It combines two components: Term Frequency (TF) measuring how often a word appears in a document, and Inverse Document Frequency (IDF) which downweights words that appear frequently across many documents, reducing the influence of common terms like “the” or “is”. 
The resulting TF-IDF value is high for words that occur often in a document but rarely elsewhere, making them good indicators of the document’s content. 

**Process**: TfidfVectorizer with a vocabulary limit of 5000 features was used to transform the lemmatized text into numeric vectors. This representation was then used for modeling and to extract the top-weighted words as keywords for each text sample.

#### 8. Text Embedding: 
**Process**: The **all-MiniLM-L6-v2** Sentence Transformer model was used to generate embeddings for the text data 

**Result**: Resulting in a feature matrix X with a shape of (6209, 384).

---


## Machine Learning Algorithms/Models Implemented
### Unsupervised 

#### Google Colab Notebook:[Unsupervised Learning](https://colab.research.google.com/drive/1-Mh7KOF1stMSGIRX6y2SKhv2Z0t-MaxA?usp=sharing#scrollTo=fT2_AmgrY6BF)

We tried several unsupervised clustering methods, including K-means clustering, GMM, density-based methods like DBSCAN, hierarchical clustering, and spectral clustering. After data loading and preprocessing to generate our text embeddings, the steps we took for the unsupervised learning methods as follows:  

#### 1. Dimensionality Reduction:
Principal Component Analysis (PCA) was applied to reduce the dimensionality of the text embeddings from 384 to 50 components (X_pca). A 2-component PCA (X_vis) was also performed specifically for 2D visualizations. 

#### 2. Unsupervised Clustering Techniques: 
We visualized the categorization of our unique political bias labels by setting our number of clusters to 5. We then evaluated these clustering methods by outputting visualizations of each which will be discussed in the results.

Several unsupervised clustering algorithms were initialized: 
- K-Means and Mini-Batch K-Means
- Gaussian Mixture Model (GMM) and Bayesian Gaussian Mixture Model
- Density-based methods: DBSCAN, HDBSCAN, OPTICS, and Mean Shift
- Hierarchical Clustering: Agglomerative Clustering and Birch
- Spectral Clustering

The current status of the models is that they have been fitted and their results visualized in 2D (and one attempt in 3D), which will be discussed below. The performance metrics have been calculated and are available in the computed_metrics DataFrame. Observations have been made regarding the visualizations, and the potential impact of hyperparameters on DBSCAN and Spectral Clustering performance has been discussed. 

### Supervised
#### Learning Method 1: Multi-Layer Perceptron (MLP) with TF-IDF Features
We used pre-processed TF-IDF vectors for this method of supervised learning. Our model architecture is a Shallow Neural Network with TF-IDF. 
Implementation: sklearn.neural_network.MLPClassifier with hidden layers (256, 128), Adam optimizer, early stopping.

Architecture: 
	- Input: 5000 dimensional TF-IDF vectors
	- Hidden layers: [256, 128] with ReLU activation
	- Output: Softmax over bias classes (Left, Center, Right, etc.)
	- Regularization: Early stopping with validation monitoring 

Why This Approach: This baseline establishes a performance floor using traditional NLP features. The shallow architecture with TF-IDF is computationally efficient and interpretable—we can directly examine which terms drive predictions. MLPClassifier's early stopping prevents overfitting on potentially noisy bias labels, and the moderate depth (2 hidden layers) balances expressiveness with generalization. This approach is expected to capture overt bias signals (explicit partisan vocabulary) but may struggle with subtle framing differences. 

Expected Performance: Macro F1 ~0.65-0.75. TF-IDF excels at identifying explicit bias markers but lacks semantic understanding for nuanced framing. 

#### Learning Method 2: Fine-Tuned RoBERTa Model
RoBERTa's pre-training on massive corpora provides rich contextual representations crucial for bias detection. Fine tuning RoBERTa is an advanced supervised contextual classification method that helps us to move beyond static word embeddings and bag-of-words representations. Unlike general embeddings, RoBERTa generates context-sensitive representations that allows us to capture idealogical cues and common political semantic distinctions that might be more subtle in the data. 

Key Advantages: 
- Bidirectional Context: Captures how words interact within entire sentences, detecting framing through word ordering and syntax.
- Transfer Learning: Pre-trained language understanding transfers to political domain with minimal fine-tuning
- Attention Mechanisms: Self-attention layers learn which words/phrases carry ideological signals
- Semantic Sensitivity: Distinguishes between "border security" vs "anti-immigrant policy"—same topic, different framing 
 
Model Architecture
We fine-tuned the roberta base model (12 encoder layers, 768-dimensional hidden states, 12 attention heads), and extended it with a task-specific classification head consisting of a dropout layer (regularization) and a linear projection from RoBERTa’s pooled representation to five political bias logits: left, lean left, center, lean right, right. The gradients were processed through all the transformer layers to allow RoBERTa to adapt to contextual representations of the politcal catagories. 
For configuration we used Hugging Face Trainer API, enabling automatic logging, checkpointing, and metric tracking.

Data Processing: allowing RoBERTa to learn from both surface-level linguistic patterns and deeper contextual sequences.
- The dataset was split into 80% train; 10% validation; 10% test, preserving class balance across splits.
- Each instance included raw text plus preprocessed versions (clean_text, lemmatized, keywords), ensuring consistent and noise-reduced inputs.
- Labels corresponded to the five political bias categories.
  
![Figure 16](/assets/images/SuperTable1.png){: .small-img }


Rationale for Hyperparameters
- Small LR (1e-5): mitigates catastrophic forgetting, preserves pre-trained representations.
- AdamW: adds decoupled weight decay, improving generalization.
- Low batch size: driven by model size and GPU memory constraints.
- Validation-loss based checkpointing: reduces overfitting and captures the best epoch.

Result: During training the validation loss consistently decreased, indicating successful adaptation to the classification task. 

![Figure 5](/assets/images/SupervisedImage0.png){: .small-img }
<p class="caption"><strong>Figure 5:</strong> Training Behavior of Fine-Tuned RoBERTa</p>

![Figure 6](/assets/images/SupervisedImage1.png){: .small-img }
<p class="caption"><strong>Figure 6:</strong> Training Behavior of Fine-Tuned RoBERTa</p>

---

## Results & Discussion

### Project Goals
Our goal is to predict biases towards political ideologies by using text classifications and explainability to highlight certain words and tones that are prevalent in writing that is either left or right-leaning. We also think we can train our model by grouping data from specific news sources into the categories of either left or right-leaning, then by deploying the model on external news data, we can devise a metric to quantify the added bias to the article. 

### Expected Results
- Supervised classification (left/center/right) evaluated using accuracy and macro-F1. 
- Unsupervised ECFD clusters evaluated with silhouette scores, Adjusted Rand Index, and human evaluation for interpretability. 
- Insightful narratives showing selective coverage and framing differences across outlets.

### Quantitative Metrics
1. **Likelihood/Probability Metrics**: Cross-Entropy Loss, Log Loss, Brier Score 
2. **Performance Metrics**: Macro Averaged F1, ROC-AUC 
3. **Explainability Metrics**: Shapley Additive Explanations (SHAP) Values, Integrated Gradients, LIME, Attention Visualization 
4. **Clustering Metrics**: Silhouette Score, Normalized Mutual Information (NMI), Adjusted Rand Index (ARI), Class Balance Checks, Expected Calibration Error (ECE) 

## Unsupervised
### Clustering Visualizations 
**K-Means and Mini-Batch K-Means clustering results**: Visualized in 2D using the 2-component PCA reduced data, showing the cluster assignments and centroids. (Figure 1). The resulting clustering from K-Means shows overlapping clusters that are quite dense, indicating that there is some overlap between different political bias labels. In the future, we plan to reduce the density of the clustering and create more distinct clusters. 
![Figure 7](/assets/images/image5.png){: .small-img } 
<p class="caption"><strong>Figure 7:</strong> K-Means Clustering</p>


**Gaussian Mixture Model (GMM) Results**: Visualized in 2D, including the cluster means and covariance ellipses. 
![Figure 8](/assets/images/image6.png){: .small-img } 
<p class="caption"><strong>Figure 8:</strong> GMM Clustering</p>

**Hierarchical clustering (Agglomerative and Birch) results**: visualized in 2D.  (Figure 3) A dendrogram for hierarchical clustering was generated to show the merging of clusters. (Figure 4) 
![Figure 9](/assets/images/image7.png){: .small-img } 
<p class="caption"><strong>Figure 9:</strong> Agglomerative Clustering</p>
![Figure 10](/assets/images/image8.png){: .small-img } 
<p class="caption"><strong>Figure 10:</strong> Dendogram</p>

**DBSCAN clustering results**: visualized in 2D, and a 3D attempt was made.  First, when we visualized all the points, we noticed that a majority of the points were identified as noise. (Figure 5) We then removed the noise points to just viusalize the DBSCAN clustering. (Figure 6) 
![Figure 9](/assets/images/image9.png){: .small-img } Figure 9. DBSCAN without noise 
<p class="caption"><strong>Figure 9:</strong> DBSCAN without noise</p>
![Figure 10](/assets/images/imagea.png){: .small-img } 
<p class="caption"><strong>Figure 10:</strong> DBSCAN with noise</p>


**Spectral clustering results**: visualized in 2D, and the n_neighbors hyperparameter was adjusted. (Figure 7) 
![Figure 11](/assets/images/imageb.png){: .small-img } 
<p class="caption"><strong>Figure 11:</strong> Spectral Clustering</p>


We analyzed all our visualized clustering algorithms by conducting an evaluation of the metrics and normalizing the scores.
- A dictionary of clustering evaluation metrics was defined, including both intrinsic (Silhouette, Calinski-Harabasz, Davies-Bouldin) and extrinsic (Adjusted Rand Index, NMI, Homogeneity, Completeness, V-Measure) metrics.
- These metrics were computed for the labels obtained from fitting the various clustering algorithms.  

From the clustering evaluation metrics shown below, we can see that together with the results of the graph and the heatmap, DBSCAN, HDBSCAN, and Mean Shift had some of the highest overall performances across the quality metrics.
They show to perform well in Calinski-Harabasz, Completeness, Davies-Bouldin, and Fowlkes-Mallows, with DBSCAN and HDBSCAN showing strong scores and handling cluster density well. 
We found that Agglomerative, GMM, and Birch had the lowest performance across metrics, showing poorer clustering for our bias data. To conclude, as we move forward, DBSCAN, HDBSCAN and Mean Shift are the best performing algorithms and will be prioritized in further modeling for our project. 

![Figure 12](/assets/images/imagec.png){: .small-img } 
<p class="caption"><strong>Figure 12:</strong> Clustering Evaluation Metrics</p>
![Figure 13](/assets/images/imaged.png){: .small-img } 
<p class="caption"><strong>Figure 13:</strong> Clustering Evaluation Heatmap</p>

## Supervised: Multi-Layer Perceptron (MLP) with TF-IDF Features
The first method established a strong baseline using traditional machine learning techniques optimized for text classification.

Feature Extraction: Textual data was transformed using Term Frequency-Inverse Document Frequency (TF-IDF). This technique assigns weights to words based on their frequency in a document relative to their frequency across the entire corpus, prioritizing terms with high discriminative power.
- Parameters: Max features were limited to 5,000, with min_df=2 and max_df=0.95, utilizing unigrams and bigrams (ngram_range=(1, 2)).
- Classifier: A Multi-Layer Perceptron (MLP) (a feed-forward neural network) was used.
- Hidden Layers: The network consisted of two hidden layers with sizes (256, 128), utilizing the ReLU activation function.
- Training: The Adam optimizer was used with an adaptive learning rate and a batch size of 32. Early stopping with a validation fraction of 0.1 was implemented to prevent overfitting.


**Overall Performance**

MLP Results:
Accuracy: 0.8993
Macro F1: 0.8728
Weighted F1: 0.8990

**Per-class Performance**
![MLP Per-class Performance](/assets/images/mlp_per_class.png){: .small-img } 

**Interpretaton of Results**
The model performed strongest on the extreme categories (Right and Left), achieving F1-scores of 0.93 for both. Its primary area of confusion occurred within the centrist categories (Lean Left and Center), with Lean Left having the lowest Macro F1-score (0.79). This high performance suggests that TF-IDF successfully extracted highly discriminative features that separated the political vocabularies.

**Confusion Matrix**
- Quantifies misclassification patterns
- Reveals systematic confusion among ideologically adjacent classes
  
![Figure 14](/assets/images/mlp_confusion_matrix.png){: .small-img } 
<p class="caption"><strong>Figure 14:</strong> MLP Confusion Matrix</p>


## Supervised: Fine-Tuned RoBERTa Model

The model was evaluated on a held-out test set after fine-tuning. We report weighted metrics to account for class imbalance in the distribution of political bias labels. The results indicate moderate classification performance, with a reasonable level of agreement between precision and recall, suggesting no extreme skew toward false positives or false negatives. The F1 score shows that RoBERTa is able to capture some ideological cues, but struggles with fine-grained political distinctions.

**Overall Performance**

![Figure 17](/assets/images/SuperTable2.png){: .small-img }

**Per-class Performance**

![Figure 17](/assets/images/SuperTable3.png){: .small-img }

**Interpretaton of Results**
- Left shows high precision (0.64) but low recall (0.24): the model is selective when predicting "left", but misses some true cases.
- Lean right obtained extremely high recall (0.89) but low precision (0.25): the model over-predicts this class, capturing most true positives at the expense of false positives.
- Right class performance is uniformly low, indicating difficulty separating strongly right-leaning content from adjacent categories.
- Macro vs. weighted averages reflect dataset imbalance: macro averages are lower due to small-class underperformance.
- These trends highlight semantic overlap between neighboring ideological categories and reveal weaknesses in fine-grained discrimination.

**Confusion Matrix**
- Quantifies misclassification patterns
- Reveals systematic confusion among ideologically adjacent classes
  
![Figure 14](/assets/images/SupervisedImage2.png){: .small-img } 
<p class="caption"><strong>Figure 14:</strong> Confusion Matrix</p>


**One-vs-Rest AUROC Curves**
- Assess ranking ability for each bias category
- Provide a threshold-independent view of separability

![Figure 15](/assets/images/SupervisedImage3.png){: .small-img } 
<p class="caption"><strong>Figure 15:</strong> One-vs-Rest AUROC Curvess</p>

#### Overall Analysis 

The model demonstrates **moderate discriminatory ability** in predicting political bias, showing strong performance for certain categories (lean right recall, left precision). Notably, the high recall but low precision for lean right suggests threshold calibration or class-specific re-weighting may help. The fine-tuned model has difficulty with right category reflects semantic similarity with lean right and potential dataset imbalance. The performance indicates that pre-trained transformers are capable of capturing ideological signals, but fine-grained classification remains challenging.

Given the class imbalance and overlapping semantics, future improvements may include:
- Label smoothing or focal loss
- Class-balanced sampling
- Calibration-aware thresholding
- Domain-adaptive pretraining on political corpora
- Incorporating metadata (source, outlet, publication date)

---

## Comparison of the Models Implemented

Comparison of our Supervised Learning Methods:

**1. Gaussian NB**

F1-Score (weighted): 0.4486
Features follow a Gaussian distribution, which is usually a poor fit for sparce data
Too simplistic for nuanced text classification as we're assuming conditional independence between features

**2. Fine-tuned RoBERTa**

F1-Score (weighted): 0.8237​
Self-attention mechanism
Pre-trained on massive amounts of text, allowing it generate contextualized embeddings
May not perform well on too small of a fine-tuning dataset

**3. MLP with TF-IDF**

F1-Score (weighted): 0.899
TF-IDF features successfully capture term importance and differences in vocabulary of each bias group
MLP can learn complex, non-linear relationships

Thus, we can conlclude that the performance of our three supervised learning approaches shows increasing representational power acroos model architectures. The Gaussian Naive Bayes is more computationally simple, but it is limited by its dependence assumptions and the high-dimensional structure of text data. Fine-tuned RoBERTa benefited from contextualized embeddings and transformer mechanisms, achieving a substantially higher accuracy, despite being constrained by the small fine-tuning dataset. Finally, the MLP trained on TF-IDF features produced the strongest results overall, showing that even without deep contextual embeddings, an expressive classifier paired with informative lexical features can capture patterns that allow the model to distinguish between the bias catagories. The findings highlight the use of models whose inductive biases align with dataset characteristics,  combining rich lexical representations with non-linear learning offer the best performance. 

Taken together, our clustering experiments demonstrate that unsupervised learning struggled to recover meaningful structure within the bias categories. Although DBSCAN, HDBSCAN, and Mean Shift showed comparatively stronger performance, especially on density-based metrics, the overall seperation of the data was weak. Thus, no method produced clusters that cleanly divided the bias labels. When viewed alongside one another, we see that supervised models outperform our other methods by learning discriminative patterns and achieve strong F1-scores. The unsupervised models revealed that these patterns are not inherently organized in the feature space in a way that clustering can easily detect. This underscores that bias categories rely on subtle linguistic cues that require advanced supervision to model effectively. Consequently, supervised learning remains far more effective for reliably distinguishing between bias labels in this dataset.

### Next Steps
**Data Improvements:**
Our supervised models show clear sensitivities to class imbalances with minority classes being harder to predict. We suggest balancing the dataset by oversampling minority classes and undersampling majority classes. 

**Model Improvements:**
While of fine-tuned RoBERTa model performed well, we suggest that testing larger RoBERTa models may capture more nuances and better define catagories.'

**Exploring Multi-Label Classification:**
Our current models are made to assume that each text sample expresses only one politcal bias catagory, instead of single-class biases. We can have the models rank the ideals presented in the texts using percentages such as 20% left-leaning and 80% balanced etc. to reflect more detailed political ideals

---
## Novelty Ideas 

### Methods Overview

We implemented two novel contrastive learning approaches to enhance political bias detection:

**SimCSE (Simple Contrastive Learning of Sentence Embeddings)**: This approach uses "Dropout as Data Augmentation" to create positive pairs from the same sentence without requiring external labels. By optimizing for **Alignment** (closeness of self-pairs) and **Uniformity** (spread of all embeddings), SimCSE learns semantic structure from scratch. This unsupervised method is particularly valuable for our domain where labeled data is limited, allowing the model to capture nuanced political language patterns. [14]

**SetFit (Efficient Few-Shot Learning)**: SetFit employs a "Data Amplification" technique that converts N labeled samples into N^2 contrastive pairs, maximizing learning efficiency from small datasets. The method decouples feature learning (Stage 1: Contrastive Loss) from prediction (Stage 2: Frozen Classification Head), preventing "catastrophic forgetting." This two-stage approach is well-suited for political bias detection where obtaining large amounts of labeled data across all bias categories is challenging. [13]

### SimCSE Implementation

#### Dataset Creation

To train SimCSE effectively on political text, we constructed a specialized political corpus by aggregating data from multiple news sources. Our pipeline combines several large-scale datasets to ensure diverse coverage of political discourse:

**Data Sources:**
1. **CC-News**: We filtered articles from major political news outlets including Politico, The Hill, CNN, Fox News, New York Times, Washington Post, Breitbart, and Reuters. This ensures representation across the political spectrum.
2. **AG News**: We extracted sentences from the "World" category (label 1), which contains political, geopolitical, and election-related content.
3. **GDELT Events**: We selected articles tagged with political theme codes (POL, GOV, ELC, PRO) to capture government, electoral, and political protest coverage.

**Preprocessing Pipeline:**
- **Sentence Splitting**: Applied regex-based splitting on sentence boundaries (`.!?`) to create individual training examples
- **Length Filtering**: Retained only sentences with 6+ words to ensure semantic richness
- **Text Cleaning**: Removed newlines and excess whitespace while preserving sentence structure
- **Deduplication**: Applied set-based deduplication to remove redundant sentences
- **Shuffling**: Randomized sentence order to prevent dataset bias

The final corpus contains over 300,000 unique political sentences, providing rich unsupervised training data for SimCSE to learn domain-specific semantic representations without requiring labeled bias annotations.

#### Unsupervised Training Method

Our SimCSE training implementation leverages dropout noise as the sole data augmentation mechanism, following the unsupervised SimCSE framework. The training process operates as follows:

**Dual Forward Pass Architecture:**
For each input sentence, we perform two forward passes through the same RoBERTa-base encoder with dropout enabled. Since dropout masks are randomly applied during each forward pass, the same sentence produces two slightly different representations. These form positive pairs for contrastive learning without requiring manually constructed augmentations or labeled data.

**Pooling Strategy:**
We extract sentence embeddings using mean pooling over all token representations weighted by attention masks, which empirically performs better than CLS token pooling for capturing full sentence semantics. An optional projection head (with tanh activation) can be added to improve contrastive learning stability.

**Contrastive Loss Function:**
The model optimizes an NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss with temperature parameter τ = 0.05. For a batch of N sentences producing 2N embeddings, each embedding treats its dropout-augmented counterpart as the positive example while all other 2N-2 embeddings serve as negatives. This encourages:
- **Alignment**: Embeddings from the same sentence (different dropout) should be similar
- **Uniformity**: Embeddings should be well-distributed across the representation space

**Training Configuration:**
- Base model: RoBERTa-base (125M parameters)
- Batch size: 128 with gradient accumulation
- Learning rate: 2e-5 with linear warmup (10% of steps)
- Max sequence length: 64 tokens
- Optimizer: AdamW with weight decay 0.01

This unsupervised approach allows the model to learn political text representations from unlabeled data, capturing semantic nuances of political discourse that can later be fine-tuned for bias classification tasks.

#### Downstream Classification Experiments

After training the SimCSE model on our political corpus, we evaluated its effectiveness for bias detection through two experimental approaches using the AllSides dataset with 5-class bias labels (Right, Lean Right, Center, Lean Left, Left):

**Experiment 1: SimCSE + Logistic Regression (Frozen Embeddings)**

In this approach, we froze the SimCSE encoder and extracted fixed embeddings for classification. The pipeline consisted of:
1. Generating sentence embeddings using the trained SimCSE model (CLS token pooling)
2. Training a logistic regression classifier with class balancing on the frozen embeddings
3. Standard scaling applied to normalize the embedding space

This experiment tests whether the unsupervised contrastive learning alone produces representations that linearly separate political biases. Results demonstrated strong performance:
- **Accuracy**: 0.7905
- **Macro F1**: 0.76
- **Weighted F1**: 0.79

The confusion matrix revealed the model performs well on extreme categories (Right, Left) with higher prediction confidence, while center and lean categories showed more confusion due to their semantic proximity. The multi-class ROC curves indicated strong discriminative ability across all bias classes.

<div class="img-row">
	<div class="img-item">
		<img src="{{ '/assets/images/experiment_results/simcse_logreg/confusion_matrix.png' | relative_url }}" alt="Figure 19: Confusion Matrix - SimCSE + Logistic Regression" />
		<p class="caption"><strong>Figure 19:</strong> Confusion Matrix - SimCSE + Logistic Regression</p>
	</div>
	<div class="img-item">
		<img src="{{ '/assets/images/experiment_results/simcse_logreg/classification.png' | relative_url }}" alt="Figure 20: Classification Metrics per Class - SimCSE + Logistic Regression" />
		<p class="caption"><strong>Figure 20:</strong> Classification Metrics per Class - SimCSE + Logistic Regression</p>
	</div>
	<div class="img-item">
		<img src="{{ '/assets/images/experiment_results/simcse_logreg/roc_curve.png' | relative_url }}" alt="Figure 21: Multi-class ROC Curve - SimCSE + Logistic Regression" />
		<p class="caption"><strong>Figure 21:</strong> Multi-class ROC Curve - SimCSE + Logistic Regression</p>
	</div>
</div>

**Experiment 2: SimCSE + Fine-tuned Classification Head**

To further improve performance, we fine-tuned the RoBERTa-base encoder's classification head while keeping the pre-trained SimCSE weights as initialization. This allowed the model to adapt its representations specifically for the bias classification task while retaining the semantic structure learned during contrastive pre-training.

Fine-tuning yielded substantial improvements:
- **Accuracy**: 0.839 (+4.9% improvement)
- **Macro F1**: 0.82 (+6% improvement)
- **Weighted F1**: 0.84 (+5% improvement)

The enhanced performance demonstrates that while SimCSE creates a strong semantic foundation through unsupervised learning, task-specific fine-tuning effectively adapts these representations for nuanced bias detection. The classification metrics per class showed more balanced performance across all five bias categories, with particular improvements in the challenging "Lean Left" and "Center" classes.

<div class="img-row">
	<div class="img-item">
		<img src="{{ '/assets/images/experiment_results/simcse_finetune/confusion_matrix.png' | relative_url }}" alt="Figure 19: Confusion Matrix - SimCSE + Fine-tuned Head" />
		<p class="caption"><strong>Figure 19:</strong> Confusion Matrix - SimCSE + Fine-tuned Head</p>
	</div>
	<div class="img-item">
		<img src="{{ '/assets/images/experiment_results/simcse_finetune/classification_report_bars.png' | relative_url }}" alt="Figure 20: Classification Metrics per Class - SimCSE + Fine-tuned Head" />
		<p class="caption"><strong>Figure 20:</strong> Classification Metrics per Class - SimCSE + Fine-tuned Head</p>
	</div>
	<div class="img-item">
		<img src="{{ '/assets/images/experiment_results/simcse_finetune/roc_curve.png' | relative_url }}" alt="Figure 21: Multi-class ROC Curve - SimCSE + Fine-tuned Head" />
		<p class="caption"><strong>Figure 21:</strong> Multi-class ROC Curve - SimCSE + Fine-tuned Head</p>
	</div>
</div>

---

### SetFit Implementation

#### Methods Overview

SetFit is a parameter-efficient few-shot classification approach that trains a lightweight sentence-transformer with a contrastive objective to generate robust embeddings using only a small number of labeled examples, then fits a simple classifier on top. It amplifies supervision by creating many positive/negative pairs from limited labeled data and converges quickly, making it ideal when labels are scarce but domain text is plentiful.

#### Training Configuration

Initialize a SetFit model with `sentence-transformers/all-mpnet-base-v2` as the backbone. Training uses SetFit’s pair sampling to create contrastive examples from the labeled set:
- Batch size: 64
- Epochs: 1 (fast convergence typical for SetFit)
- Num iterations: 20 (pairs generated per sentence)
- Metric: accuracy on the held-out test split

#### Rationale and Results

Compared to SimCSE, SetFit directly leverages labeled examples to shape the embedding space toward bias classes while keeping training lightweight. The `all-mpnet-base-v2` encoder provides strong sentence-level semantics; SetFit’s pair sampling increases supervision density without requiring large datasets. We expect competitive accuracy with fast training times and good performance on extreme classes, with centrist and “lean” classes being more challenging due to semantic proximity.

Using SetFit yielded substantial improvements:
- **Accuracy**: 0.90
- **Macro F1**: 0.873
- **Weighted F1**: 0.901

<div class="img-row">
	<div class="img-item">
		<img src="{{ '/assets/images/experiment_results/setfit/confusion_matrix.png' | relative_url }}" alt="SetFit Confusion Matrix" />
		<p class="caption"><strong>Figure: 22</strong> Confusion Matrix - SetFit</p>
	</div>
	<div class="img-item">
		<img src="{{ '/assets/images/experiment_results/setfit/classification_report_bars.png' | relative_url }}" alt="SetFit Classification Metrics" />
		<p class="caption"><strong>Figure: 23</strong> Classification Metrics per Class - SetFit</p>
	</div>
	<div class="img-item">
		<img src="{{ '/assets/images/experiment_results/setfit/roc_curve.png' | relative_url }}" alt="SetFit ROC Curve" />
		<p class="caption"><strong>Figure: 24</strong> Multi-class ROC Curve - SetFit</p>
	</div>
</div>

#### Why SetFit Performs Better

Our SetFit experiments achieved notably stronger results compared to the SimCSE embeddings. Key reasons for this improvement include:

- **Supervised contrastive signal (label amplification):** SetFit generates N^2 contrastive pairs from N labeled examples, which greatly increases the effective supervision. This forces the encoder to pull together examples of the same bias class and push apart different classes, which is directly aligned with the classification objective.

- **Task-aligned supervision vs. generic semantics:** SimCSE learns general-purpose sentence similarity using dropout augmentation and is unsupervised; it captures broad semantics but not class-discriminative signals. SetFit's supervised pair sampling sculpts the representation space specifically around bias labels, making classes more linearly separable.

- **Stronger sentence encoder backbone:** We use `all-mpnet-base-v2` for SetFit, which provides higher-quality sentence representations out-of-the-box compared with a base RoBERTa encoder used in some SimCSE runs. Better base embeddings amplify the effectiveness of contrastive fine-tuning.

- **Dense supervision with few labels:** Because SetFit produces many contrastive pairs per example, it achieves high sample efficiency — the model sees a rich variety of intra-class and inter-class comparisons even when labeled data is limited.

- **Lightweight, stable training:** SetFit trains fast (often 1 epoch) and requires fewer updates to shape the encoder, reducing overfitting risk while still learning discriminative boundaries.

These factors combine to give SetFit an advantage for our 5-way bias classification task: it directly optimizes the embedding geometry for the downstream classifier while remaining computationally efficient.

**Caveats:**

- SetFit’s supervised benefits depend on label quality; noisy or inconsistent labels will amplify noise as well as signal.
- The method can still struggle with semantically ambiguous classes (e.g., "lean left" vs "center"). Additional techniques like label smoothing, hierarchical labels, or multi-task signals (topic + stance) may further help.



## References

[1] R. M. Entman, “Framing: Toward Clarification of a Fractured Paradigm,” Journal of Communication, vol. 43, no. 4, pp. 51–58, 1993. 

[2] R. Baly, G. Karadzhov, D. Alexandrov, J. Glass, and P. Nakov, “Predicting Factuality of Reporting and Bias of News Media Sources,” in Proc. ACL, 2018. 

[3] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,” in Proc. EMNLP-IJCNLP, 2019. 

[4] P. Khosla et al., “Supervised Contrastive Learning,” in Proc. NeurIPS, 2020. 

[5] J. Jiang, X. Ren, and E. Ferrara, “Retweet-BERT: Political Leaning Detection Using Language Features and Information Diffusion on Social Networks,” Proc. AAAI ICWSM, vol. 17, pp. 459–469, Jun. 2023. doi: 10.1609/icwsm.v17i1.22160. 

[6] A. Pastorino et al., “Decoding News Narratives: A Critical Analysis of Large Language Models in Framing Detection,” 2024. 

[7] Y. Wang, S. Frederick, Y. Duan et al., “Detecting Misinformation through Framing Theory: the Frame Element-based Model,” 2024. 

[8] J. Rönnback, J. Carlsson, C. Calleja, and R. Feldman, “Automatic large-scale political bias detection of news outlets,” PLOS ONE, vol. 19, no. 8, e0321418, Aug. 2024. doi: 10.1371/journal.pone.0321418. 

[9] L. Fan, M. White, E. Sharma, R. Su, P. K. Choubey, R. Huang, and L. Wang, "In plain sight: Media bias through the lens of factual reporting," EMNLP, 2019. doi: 10.48550/arXiv.1909.02670 

[10] S. Haldar, “AllSides : Ratings of bias in electronic media,” Kaggle.com, 2021. https://www.kaggle.com/datasets/supratimhaldar/allsides-ratings-of-bias-in-electronic-media/data 

[11] The GDELT Project, “The GDELT Project,” Kaggle.com, 2015. https://www.kaggle.com/datasets/gdelt/gdelt 

[12] idiap, “GitHub - idiap/Factual-Reporting-and-Political-Bias-Web-Interactions: Mapping the Media Landscape: Predicting Factual Reporting and Political Bias,” GitHub, 2024. https://github.com/idiap/Factual-Reporting-and-Political-Bias-Web-Interactions 

[13] Hugging Face, "SetFit: Efficient Few-Shot Learning", GitHub repository, 2022. https://github.com/huggingface/setfit

[14] W. Gao, H. Yao, and K. Chen, "SimCSE: Simple Contrastive Learning of Sentence Embeddings," arXiv preprint arXiv:2104.08821, 2021. https://arxiv.org/abs/2104.08821

---

## Checklist Progress

### ✅ Completed

- Literature Review
- Dataset Description
- Dataset Link (if applicable)
- Problem Definition
- Motivation
- 3+ Data Preprocessing Methods Identified
- 3+ ML Algorithms/Models Identified
- CS 7641: Unsupervised and Supervised Learning Methods Identified
- CS 4641: Supervised or Unsupervised Learning Methods Identified
- 3+ Quantitative Metrics
- Project Goals (including sustainability and ethical considerations)
- Expected results
- 3+ References (preferably peer reviewed)
- 1+ In-Text Citation Per Reference
