# Content-Based-Movie-Recommender-System-with-sentiment-analysis-using-AJAX


Content Based Recommender System recommends movies similar to the movie user likes and analyses the sentiments on the reviews given by the user for that movie.



The details of the movies(title,director_names,actor_names) are fetched from the different data sources like Kaggle,wikipedia,Imdb website by using web scraping.I have used Auto Scraper library for `web scraping`.Also collected reviews from the Imdb site and performed sentiment analysis on those reviews.

check out my project here: https://movie-recommender-by-kalyani.herokuapp.com/


## Movie Recommender System User interface :

![screen1](C:/Users/srava/OneDrive/Pictures/screen1.png)

![screen2](C:/Users/srava/OneDrive/Pictures/screen2.png)

![screen3](C:/Users/srava/OneDrive/Pictures/screen3.png)

## Similarity Score : 

   How does it decide which item is most similar to the item user likes? Here we use the similarity scores.
   
   It is a numerical value ranges between zero to one which helps to determine how much two items are similar to each other on a scale of zero to one. This similarity score is obtained measuring the similarity between the text details of both of the items. So, similarity score is the measure of similarity between given text details of two items. This can be done by cosine-similarity.

   ## How Cosine Similarity works?
  Cosine similarity is a metric used to measure how similar the documents are irrespective of their size. Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space. The cosine similarity is advantageous because even if the two similar documents are far apart by the Euclidean distance (due to the size of the document), chances are they may still be oriented closer together. The smaller the angle, higher the cosine similarity.
  
  ![image](https://user-images.githubusercontent.com/36665975/70401457-a7530680-1a55-11ea-9158-97d4e8515ca4.png)


  ### Sources of the datasets 

1. [IMDB 5000 Movie Dataset](https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset)
2. [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset)
3. [List of movies in 2018](https://en.wikipedia.org/wiki/List_of_American_films_of_2018)
4. [List of movies in 2019](https://en.wikipedia.org/wiki/List_of_American_films_of_2019)
5. [List of movies in 2020](https://en.wikipedia.org/wiki/List_of_American_films_of_2020)
