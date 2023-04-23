# Spotify-Music-Recommendation-System

This Music Recommendation System returns a list of song recommendations by analyzing a spotify dataset of songs. The system enters a dataset and list of songs the user likes into a model and formulates a list of recommendations from that song data. Its a simple Exploratory data analysis (EDA) program using KMEANS clustering. The dataset can either be a Spotify user generated playlist or an imported CSV file. The goal of this program is provide better songs recommendations than the existing spotify recommendation program. 

Here's a visual of the KMEANS song clustering in case anyone is confused. Basically song datapoints on a scatter plot which can be modeled into smaller, more specific clusters. Please refer to this [Kaggle Notebook](https://www.kaggle.com/code/vatsalmavani/music-recommendation-system-using-spotify-dataset) for more clarification.

# How to Set Up:

1. Create a free spotify account and create a playlist of songs you really like.

2. Go to the Spotify developer dashboard page and create an APP using this link:
- https://developer.spotify.com/dashboard/login. 
- Copy the client keys into the secret.py file. 
- Your redirect_uri should always be *http://localhost:8888/callback*.

3. The code to connect to the Spotify Client Module and Spotify Oauth2 Module is already written in recommendation.py file. Just update your API keys and scopes. I used the Spotify Authorization code flow to allow the user access for playlist modifictaion and token refreh. 

4. Assign the playlist name variable to playlist you want to pull song data from. Once the reccomendation system runs, it'll automically pull your 5 most popular songs from that dataset to enter into the model. You can manually enter a list of songs you like as well. 

5. To manually enter a list of songs you want reccomendations similar too, follow this format: 
```
resultant = recommend_songs([{'name': 'Drunk in Love'},
                 {'name': 'Smells Like Teen Spirit'},
                 {'name': 'No Church in the wild'},
                 {'name': 'Umbrella'},
                 {'name': 'Rich spirit'}],  data)
  NOTE: Your data variable will be equal to playlist dataset.
```
6. Now you can finally run the program. Here's an example output the recommendation system will return: 
```
 [{'name': 'Aura', 'artists': 'Kanyun', 'id': '02AGPtoDY9XXpCQDuRK65f'}, {'name': 'NIGHTY NIGHT', 'artists': 'YUXiANG', 'id': '23bAqUPuxNAGahF8q4cPdL'}, {'name': 'nighttime reading', 'artists': '1000LI', 'id': '34uHo5HoSDR4Gy46YkIe0r'}, {'name': 'a penny for your thoughts', 'artists': 'ombræ', 'id': '7vUeIAH5LGG4IguBI530MQ'}]
```
See the requirments.txt file to install the required libraries. 

