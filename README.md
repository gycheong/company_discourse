# erdos-companydiscourse
There is a wealth of discourse on companies and their products on social media platforms and online forums. While many approaches leverage analytical techniques to gauge audience sentiment through online discourse, they lack the ability to be both targeted and customizable while maintaining complex analytical integrity.
<center>
<img src="images/SVG/figure1.svg" width="100%"></img>
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
This project utilizes Natural Language Processing (NLP) and Machine Learning (ML) techniques to construct predictive models capable of assessing and rating comments provided by consumers for a **target company**. In this project we used [Costco](https://www.costco.com/) as the target company. By employing these advanced analytical methods, we aim to enhance the accuracy and effectiveness of sentiment analysis in understanding and forecasting consumer behavior. **To view a detailed description of the entire project for Costco as the target company **, please run the [full notebook](https://github.com/dhk628/erdos-companydiscourse/blob/main/final%20notebook.ipynb)



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
The final models for the project are stored in the folder `models`. To apply them to reviews you need to:
1. Vectorize a list of reviews using `SentenceTransformer("thenlper/gte-large").encode(reviews)` from the package [sentence-transformers](https://www.sbert.net/).
2. Load one of the models and apply `model.predict(review_vectors)` to the corresponding vector list.
To reproduce the training and testing done for this project you need to:
1. Download the complete review data from [Google Reviews Data](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/) to the folder `data/raw/` (omitted in the GitHub repo due to file size).
2. Run `google_preprocessing.ipynb` to extract the reviews of your target company (Costco in our case).
3. `sbert_vectorizing.py` contains the necessary code to vectorize and store the reviews.
4. Use `xgboost_training.ipynb` and `neural_network_implementation.ipynb` to train the respective models.

## Data
### Overview
We used the dataset [Google Reviews Data](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/) to train our models. This dataset includes all Google Maps reviews from 2021 in the United States, up to September 2021. We extracted all reviews associated with a Costco location (usually there is more than one Google Map ID for each Costco warehouse), and we excluded all reviews that were not in English. We did not alter the review text in any way before vectorizing.

<center>
<img src="images/SVG/figure2.svg" width="100%"></img>
</center>


## Models
We modeled the data useing the following models:
- [Baseline Model](#baseline-model)
- [Logistic Regression](#logistic-regression)
- [K-Nearest Neighbors](#k-nearest-neighbors)
- [Support Vector Machine for Classification](#support-vector-machine-for-classification)
- [XGBoost Classifier](#xgboost-classifier)
- [Feedforward Neural Network](#feedforward-neural-network)
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


