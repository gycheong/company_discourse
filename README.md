# erdos-companydiscourse
<img src="images/SVG/figure1.svg" width="100%"></img>
There is a wealth of discourse on companies and their products on social media platforms and online forums. While many approaches leverage analytical techniques to gauge audience sentiment through online discourse, they lack the ability to be both computationally efficient while factoring in context and maintining complex analytical integrity.

Full information of this project can be found in the final [Jupyter notebook](final_notebook.ipynb).
<center>
</center>


## Table of Contents
- [Project Title](#project-title)
- [Table of Contents](#table-of-contents)
- [Project Description](#description)
- [Project Structure](#project-structure)
- [Installation & Usage /Reproducability](#installation--usage-reproducability)
- [Data](#data)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Description
This project utilizes Natural Language Processing (NLP) and Machine Learning (ML) techniques to construct predictive models capable of assessing and rating comments provided by consumers for a **target company**. In this project we used [Costco](https://www.costco.com/) as the target company. By employing these advanced analytical methods, we aim to enhance the accuracy and effectiveness of sentiment analysis in understanding and forecasting consumer behavior. **To view a detailed description of the entire project for Costco as the target company**, please run the [full jupyter notebook](https://github.com/dhk628/erdos-companydiscourse/blob/main/final_notebook.ipynb)



## Project Structure
- `data/`: **Omitted** in the GitHub repo due to file size.
  - `raw/`: Contains raw scraped data.
  - `processed/`: Contains processed data ready for analysis.
- `notebooks/`: Jupyter notebooks for exploratory data analysis, preprocessing, vectorization, model training, and evaluation.
- `notebooks/final_notebook.ipynb` jupyter notebook with full description of data analysis and results
- `scripts/`: Python scripts for data collection, preprocessing, vectorization, model training, and evaluation.
- `models/`: Directory to store trained models.
- `.gitignore`: Files and directories to be ignored by git.
- `README.md`: Project documentation and instructions.
- `erdos_company_discourse.yml`: project environment

## Installation & Usage /Reproducability
The final models for the project are stored in the [`models`](https://github.com/dhk628/erdos-companydiscourse/tree/main/models) folder. To apply them to reviews you need to:
1. Vectorize a list of reviews using `SentenceTransformer("thenlper/gte-large").encode(reviews)` from the package [sentence-transformers](https://www.sbert.net/).
2. Load one of the models and apply `model.predict(review_vectors)` to the corresponding vector list.

To reproduce the training and testing done for this project you need to:
1. Download the complete review data from [Google Reviews Data](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/) to the folder `data/raw/` (omitted in the GitHub repo due to file size).
2. Run `google_preprocessing.ipynb` to extract the reviews of your target company (Costco in our case).
3. `sbert_vectorizing.py` contains the necessary code to vectorize and store the reviews.
4. Use `xgboost_training.ipynb` and `neural_network_implementation.ipynb` to train the respective models.

## Data
### Overview
We used the dataset [Google Local Data](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/) to train our models. This dataset includes all Google Maps reviews from 2021 in the United States, up to September 2021. We extracted all reviews associated with a Costco location (usually there is more than one Google Map ID for each Costco warehouse), and we excluded all reviews that were not in English. We did not alter the review text in any way before vectorizing.

After exclusions, the dataset includes 788766 reviews from 2473 unique Google Maps locations. The data is heavily biased towards 5 stars, with the distribution being:

<center>

| Rating | Count  | Percentage |
|:-------|-------:|-----------:|
| 5      | 528011 | 66.94%     |
| 4      | 154481 | 19.59%     |
| 3      | 48264  | 6.12%      |
| 2      | 18805  | 2.38%      |
| 1      | 39205  | 4.97%      |

</center>

In the [Results](#results) section, we used 66.94% as the baseline accuracy for our models, which corresponds to always predicting 5 stars.

<center>
<img src="images/SVG/figure2.svg" width="100%"></img>
</center>


## Models
We modeled the data using the following models:
- [Baseline Model](https://github.com/dhk628/erdos-companydiscourse/tree/main/models)
- [Logistic Regression](https://github.com/dhk628/erdos-companydiscourse/tree/main/models)
- [K-Nearest Neighbors](https://github.com/dhk628/erdos-companydiscourse/tree/main/models)
- [Support Vector Machine for Classification](https://github.com/dhk628/erdos-companydiscourse/tree/main/models)
- [XGBoost Classifier](https://github.com/dhk628/erdos-companydiscourse/tree/main/models)
- [Feedforward Neural Network](https://github.com/dhk628/erdos-companydiscourse/tree/main/models)
### Results & Model Comparison
#### Accuracy
<center>
<img src="images/SVG/model_comparison_accuracy.svg" width="60%"></img>
</center>

#### Cross Entropy
<center>
<img src="images/SVG/model_comparison_ce.svg" width="60%"></img>
</center>


## Contributing
- [Vinicius Ambrosi](personalWebsiteLink)(![Github](http://i.imgur.com/9I6NRUm.png):
    [vambrosi](https://github.com/vambrosi))
- [Gilyoung Cheong](personalWebsiteLink)(![Github](http://i.imgur.com/9I6NRUm.png):
    [gycheong](https://github.com/gycheong))
- [Dohoon Kim](personalWebsiteLink)(![Github](http://i.imgur.com/9I6NRUm.png):
    [dhk628](https://github.com/dhk628))
- [Hannah Lloyd](https://hslloyd.github.io/)(![Github](http://i.imgur.com/9I6NRUm.png):
    [hslloyd](https://github.com/hslloyd))


