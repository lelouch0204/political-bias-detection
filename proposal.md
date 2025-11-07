---
layout: page
title: ML Project Proposal
permalink: /proposal/
---
<div class="hero-section">
	<h1>CS 7641 Group Project Proposal</h1>
	<blockquote>Political bias detection in news articles</blockquote>
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

### Data Preprocessing Methods Proposed

1. Deduplication, Text Normalization, Tokenization & Stopword Handling, Lemmatization/Stemming, Handling Noise and Class Balance 
2. **TF-IDF**: Idenitfy keywords that are dense within a group of articles but uncommon across all documents. Provides an initial signal for clustering articles by orientation. 
3. **Contextual Embeddings**: BERT produced vector representations of articles that preserve semantic meaning, forming features for our main analysis. 
4. **DAPT (Domain Adaptive Pre-training)**: Fine-tune BERT to adapt to political data to capture vocabulary and discourse nuances. 

### Machine Learning Algorithms/Models Proposed


#### Supervised

1. **Baseline**: Shallow Deep Neural Architecture: Shallow NN/LSTM with TF-IDF inputs. A softmax layer outputs probabilities for Left, Center, or Right. 
2. **Realistic**: Fine-tuned RoBERTa, DoBERTa model trained on labeled political articles to directly predict orientation. Paper by Jiang mentions motivations for political ideology detection in news articles using BERT [5]

#### Unsupervised
1. **Baseline**: K-Means (K=3), GMM, Hierarchical Clustering, DBScan, Spectral Clustering  
2. **SimCSE (Ambitious)**: Use contrastive learning for sentence embeddings to robust text representations, clustering semantically similar sentences and capturing fine-grained political bias cues with a triplet loss function 

## Midterm Implementations 
***Why we chose them and what we implemented*
### Data Processesing Method Implemented
Our data was pulled from a political bias dataset accessed through Kaggle. We first inspected the contents counting the number of article titles and links. Of the total 8112 titles and 8100 links, we found 6021 and 5461 being unique, repectively. The articles in this dataset were split by atributes of text, source, and bias. 
We then filtered the articles to exclude rows with irrelevant text that would not be useful in our model training, and ordering our texts by length. 
### Machine Learning Algorithms/Models Implemented
We implemented our unsupervised learning method and set up the pipeline for our supervised learning method. 

---

## Results and Discussion
---

### Quantitative Metrics

1. **Likelihood/Probability Metrics**: Cross-Entropy Loss, Log Loss, Brier Score 
2. **Performance Metrics**: Macro Averaged F1, ROC-AUC 
3. **Explainability Metrics**: Shapley Additive Explanations (SHAP) Values, Integrated Gradients, LIME, Attention Visualization 
4. **Clustering Metrics**: Silhouette Score, Normalized Mutual Information (NMI), Adjusted Rand Index (ARI), Class Balance Checks, Expected Calibration Error (ECE) 

### Project Goals

Our goal is to predict biases towards political ideologies by using text classifications and explainability to highlight certain words and tones that are prevalent in writing that is either left or right-leaning. We also think we can train our model by grouping data from specific news sources into the categories of either left or right-leaning, then by deploying the model on external news data, we can devise a metric to quantify the added bias to the article. 

### Expected Results

- Supervised classification (left/center/right) evaluated using accuracy and macro-F1. 
- Unsupervised ECFD clusters evaluated with silhouette scores, Adjusted Rand Index, and human evaluation for interpretability. 
- Insightful narratives showing selective coverage and framing differences across outlets. 

## Midterm Results and Discussion
*discussion of finding here*

### Visualizations
*Visualizations here

---
## Novelty Ideas 

Our project introduces several novel components beyond standard bias classification: 

1. **Event-Level Contrastive Learning**: Use contrastive learning on articles about the same event to detect nuanced narrative differences (e.g. triplet loss) across outlets. 
2. **Explainable and interpretable predictions**: Using model interpretability tools, we provide reasoning for each classification, highlighting which textual features and coverage decisions influenced the bias label; focusing on tone, selection, percentage of fact vs. opinion, and size biases. 
3. **Generative de-biasing**: Fine-tune an LLM to generate a de-biased version of an article. 

---
## Gantt Chart
*chart here*
## Contributions 
*table here* 

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
- Expected Results
- 3+ References (preferably peer reviewed)
- 1+ In-Text Citation Per Reference
