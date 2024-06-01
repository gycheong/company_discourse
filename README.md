# Company Discourse
<img src="images/SVG/figure1.svg" width="100%"></img>
There is a wealth of information in the discourse on companies and their products on social media platforms and online forums. This project aims to build and train machine learning (ML) models to predict google star reviews from google text reviews for a target company. This approach is computationally efficient, while maintaining contextual integrity in the data and leveraging complex analytical techniques to gauge audience sentiment through online discourse.

**To view a detailed description of the entire project**, please see our [final Jupyter notebook](https://github.com/dhk628/erdos-companydiscourse/blob/main/final_notebook.ipynb).
<center>
</center>

## Authors
- [Vinicius Ambrosi](https://www.linkedin.com/in/vinicius-ambrosi/) (![Github](http://i.imgur.com/9I6NRUm.png):
    [vambrosi](https://github.com/vambrosi))
- [Gilyoung Cheong](https://www.linkedin.com/in/gycheong/) (![Github](http://i.imgur.com/9I6NRUm.png):
    [gycheong](https://github.com/gycheong))
- [Dohoon Kim](https://www.linkedin.com/in/dohoonkim95/) (![Github](http://i.imgur.com/9I6NRUm.png):
    [dhk628](https://github.com/dhk628))
- [Hannah Lloyd](https://www.linkedin.com/in/hslloyd/) (![Github](http://i.imgur.com/9I6NRUm.png):
    [hslloyd](https://github.com/hslloyd))


## Table of Contents
- [Project Description](#description)
- [Project Structure](#project-structure)
- [Installation, Usage, and Reproducability](#installation-usage-and-reproducability)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Results and Model Comparison](#results-and-model-comparison)
- [References](#references)

## Project Description
This project utilizes Natural Language Processing (NLP) and ML techniques to construct predictive models capable of assessing and rating comments provided by consumers for a **target company**. In this project we used [Costco](https://www.costco.com/) as the target company. By employing these advanced analytical methods, we aim to enhance the accuracy and effectiveness of sentiment analysis in understanding and forecasting consumer behavior.



## Project Structure
- `notebooks/`: Jupyter notebooks for exploratory data analysis, preprocessing, vectorization, model training, and evaluation
- `final_notebook.ipynb` Jupyter notebook with full description of data analysis and results
- `scripts/`: Python scripts for data collection, preprocessing, vectorization, model training, and evaluation
- `models/`: Directory to store trained models
- `.gitignore`: Files and directories to be ignored by git
- `README.md`: Project documentation and instructions
- `erdos_company_discourse.yml`: Project environment

## Installation, Usage, and Reproducability
The final models for the project are stored in the [`models`](https://github.com/dhk628/erdos-companydiscourse/tree/main/models) folder. The model for support vector classification is large and is stored in [Google Drive](https://drive.google.com/file/d/1lqYpduA7rfBSZCMB_yUyadeiGKJsFb9B/view?usp=sharing). To apply them to reviews you need to:
1. Vectorize a list of reviews using `SentenceTransformer("thenlper/gte-large").encode(reviews)` from the package [sentence-transformers](https://www.sbert.net/).
2. Load one of the models and apply `model.predict(review_vectors)` to the corresponding vector list.

To reproduce the training and testing done for this project you need to:
1. Download the complete review data from [Google Reviews Data](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/) to the folder `data/raw/` (omitted in the GitHub repo due to file size).
2. Run `google_preprocessing.ipynb` to extract the reviews of your target company (Costco in our case).
3. `sbert_vectorizing.py` contains the necessary code to vectorize and store the reviews.
4. Use `scikit_models.py` to train scikit-learn models, and `xgboost_training.ipynb` and `neural_network_implementation.ipynb` to train the respective models.

## Exploratory Data Analysis
### Overview
We used the dataset [Google Local Data](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/) to train our models. This dataset includes all Google Maps reviews from 2021 in the United States, up to September 2021. We extracted all reviews associated with a Costco location (usually there is more than one Google Map ID for each Costco warehouse), and we excluded all reviews that were not in English. We did not alter the review text in any way before vectorizing.

After exclusions, the dataset includes 788766 reviews from 2473 unique Google Maps locations. We use Sentence Transformers, a state-of-the-art text embedding NLP, to vectorize the reviews. These vectors then serve as input features for building predictive models for ratings. The pre-trained model for our sentence transformer is GTE (General Text Embeddings with Multi-stage Contrastive Learning) developed by Alibaba Group NLP team in [[5]](https://arxiv.org/abs/2308.03281).

We use 80% of the review vectors with ratings as our training data and the rest as test data. The training data is heavily biased towards 5 stars, with the distribution being:

<center>

| Rating | Count  | Percentage |
|:-------|-------:|-----------:|
| 5      | 528011 | 66.94%     |
| 4      | 154481 | 19.59%     |
| 3      | 48264  | 6.12%      |
| 2      | 18805  | 2.38%      |
| 1      | 39205  | 4.97%      |
    
</center>

We used 66.94% as the baseline accuracy for our models, which corresponds to always predicting 5 stars. The following is the complete list of model generation techniques we used:

- Baseline Model (i.e., always predict 5 stars)
- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine for Classification
- XGBoost Classifier
- Feedforward Neural Network

### Data Visualization

<center>
<img src="images/SVG/figure2.svg" width="100%"></img>
</center>

* **A**: the scatter plot of our training set projected on the first and second principal components
* **B**: the histogram for ratings in our testing set

### Results and Model Comparison
#### Accuracy
<center>
<img src="images/SVG/model_comparison_accuracy.svg" width="60%"></img>
</center>

#### Cross Entropy
<center>
<img src="images/SVG/model_comparison_ce.svg" width="60%"></img>
</center>

## References

<a id="1">[1]</a> 
T. Chen, P. Samaranayake, X. Cen, M. Qi, and Y. Lan. (2022). "**The Impact of Online Reviews on Consumers’ Purchasing Decisions: Evidence From an Eye-Tracking Study**," Front Psychol. **13**: 865702.

<a id="1">[2]</a> 
Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. (2019). "**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**," Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)

<a id="1">[3]</a> 
M. S. Kim, K. W. Lee, J. W. Lim, D. H. Kim, and S. Hong. (2023). "**Ranking Roughly Tourist Destinations Using BERT-Based Semantic Search**", Shakya, S., Du, KL., Ntalianis, K. (eds) Sentiment Analysis and Deep Learning. Advances in Intelligent Systems and Computing, vol 1432. Springer, Singapore.

<a id="1">[4]</a>
Jiacheng Li, Jingbo Shang, and Julian McAuley. (2022). "**UCTopic: Unsupervised Contrastive Learning for Phrase Representations and Topic Mining**," Annual Meeting of the Association for Computational Linguistics (ACL)

<a id="1">[5]</a>
Z. Li, X. Zhang, Y. Zhang, D. Long, P. Xie, and M. Zhang. (2023)."**Towards General Text Embeddings with Multi-stage Contrastive Learning**," arXiv preprint: https://arxiv.org/abs/2308.03281

<a id="1">[6]</a>
N. Reimers and I. Gurebych. (2019). "**Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**," Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, p.3982–3992, Hong Kong, China, November 3–7, 2019.

<a id="1">[7]</a>
An Yan, Zhankui He, Jiacheng Li, Tianyang Zhang, and Julian Mcauley. (2023). "**Personalized Showcases: Generating Multi-Modal Explanations for Recommendations**,"The 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)

<a id="1">[8]</a>
L. Yang, Y. Li, J. Wang, and R. S. Sherratt. (2020). "**Sentiment Analysis for E-Commerce Product Reviews in Chinese Based on Sentiment Lexicon and Deep Learning**," IEEE Access **8**: p.23522-23530.

<a id="1">[9]</a>
Qiang Ye, R. Law, B. Gu, and W. Chen. (2011). "**The influence of user-generated content on traveler behavior: An empirical investigation on the effects of e-word-of-mouth to hotel online bookings**,"  Computers in Human Behavior **2**: p.634-639.

<a id="1">[10]</a>
Zhang and Zhong. (2019). "**Mining Users Trust From E-Commerce Reviews Based on Sentiment Similarity Analysis**," IEEE Access **7**: p.13523-13535.

<a id="1">[11]</a>
Y. Zhao, L. Wang, H. Tang, and Y. Zhang. (2020). "**Electronic word-of-mouth and consumer purchase intentions in social e-commerce**," Electronic Commerce Research and Applications, **41**: 100980.
