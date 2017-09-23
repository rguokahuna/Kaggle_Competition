Kaggle competition [Bag of Words Meets Bags of Popcorn](https://www.kaggle.com/c/word2vec-nlp-tutorial).

Data visualization of word cloud for each sentiment can be found [here](https://public.tableau.com/profile/rui.guo#!/vizhome/SentimentAnalysisonMovieReviews/WordCloud-)

Goal: Sentiment analysis on movie reviews

1. main.ipynb
  - Read in data
  - Ensemble bag of words, TF-IDF, word2vec and doc2vec models
  - Ensemble logistic regression, stochastic gradient descent and naive bayes models
  
2. auxiliary.py
  - Perform data cleaning: html tags removing, stopwords removing
  - Perform data wrangling 

3. Word2Vec_auxiliary.ipynb
  - Train word2vec model using both labeled and unlabeled data
  
4. doc2vec_auxiliary.ipynb
  - Train doc2vec model using both labeled and unlabled data

Results: 97.044% classification accuracy
