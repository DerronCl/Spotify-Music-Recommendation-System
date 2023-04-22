
import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline 
from scipy.spatial.distance import cdist
from collections import defaultdict
import spotipy
from spotipy.oauth2 import SpotifyOAuth 
from secret import *
import warnings 
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 10); 
pd.set_option('display.max_columns', 16); 


'''SPOTIFY RECCOMMENDATION SYSTEM'''


playlist_name = 'Chill Beats' # playlist to pull data from: 
scope = ["user-library-read", "playlist-modify-public"]  #Scope grants user access to read personal library and modify playlists
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=clientid, #Creates a Spotify API client. Import API keys from secret.py file 
                                               client_secret=secretkey,
                                               redirect_uri=redirect_uri,
                                               scope=scope))

sp_oauth = spotipy.oauth2.SpotifyOAuth(client_id=clientid,client_secret=secretkey,redirect_uri=redirect_uri,scope=scope) #setup spotify oauth2 to implement Authorization Code Flow for token refresh
cached_token = sp_oauth.get_cached_token() 
access_token = sp_oauth.get_access_token()
if not access_token: 
    auth_url = sp_oauth.get_authorize_url()
    code = sp_oauth.parse_response_code(auth_url)
    access_token = sp_oauth.get_access_token(code)
if access_token != cached_token: #Refresh api token if expired cached token doesn't match api token
    sp_oauth.refresh_access_token(access_token['refresh_token'])

username = sp.me()['id'] 
data = pd.DataFrame() #Create dataframe for playlist data
playlists = sp.user_playlists(username)
playlist_uri = 0
for i, item in enumerate(playlists['items']): #Finding requested playlist 
    if item['name'] == playlist_name: 
        playlist_uri = item['id'] 
current_playlist = sp.playlist_tracks(playlist_uri)
for tracks in sp.playlist_tracks(playlist_uri)["items"]: #extracting song data from playlist
    track_uri = tracks["track"]['id']
    track_name = tracks["track"]['name']
    artist_name = tracks["track"]["artists"][0]["name"]
    artist_uri = tracks["track"]["artists"][0]["uri"]
    artist_info = sp.artist(artist_uri)
    artist_pop = artist_info["popularity"]
    added_data = pd.Series({'artists': artist_name, "name": track_name, "popularity": artist_pop}) 
    audio_features = pd.Series(sp.audio_features(track_uri)[0])
    audio_features= pd.concat([audio_features, added_data]) #Appending extracted song data to dataframe 
    data = pd.concat([data, audio_features.to_frame().T], ignore_index=True)
data = data.reset_index(drop=True)
try: 
    data = data.drop(columns='time_signature',axis=1) #Drop unneeded columns
except: 
    pass

for columns in data:
    try:
        data[columns] = data[columns].astype(float, errors = 'raise') #Convert all numerical values to floats for reccommendation model
    except:
        pass

five_songs = data.nlargest(5, ['popularity']) #Get the five most popular songs from playlist to input into model to find similar reccommendations.
top_5 = [{'name': str(row['name'])} for index, row in five_songs.iterrows()]
 
def find_song(name): #Find song data if song is not in created spotify dataset.
    song_data = defaultdict()
    results = sp.search(q= 'track: {}'.format(name), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)

def get_song_data(song, spotify_data): #Get song data from spotify dataset
    
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name'])].iloc[0]
        return song_data
    
    except IndexError:
        return find_song(song['name']) #Find song data from Spotify if data is missing from spotify_dataset 
        

def get_mean_vector(song_list, spotify_data): #Calculate a mean vector from the song list. Need this for model
    
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database. This song was removed from song list'.format(song['name']))
            song_list.remove(song)  #Remove songs that do not exist. 
            continue

        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list): #Flatten song list dictionary 
    
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict

number_cols = ['valence', 'acousticness', 'danceability', 'duration_ms', 'energy',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo'] 

#Creates list of recommended songs using KMeans clustering and Exploratory data analysis (EDA) 
def recommend_songs( song_list, spotify_data, n_songs=5): #Spotify data is the dataset we created from our specific playlist. N_songs is the number of song recommendations to return

    song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), # Instantiate model pipeline
                                ('kmeans', KMeans(n_clusters=20,  #StandardScaler standardnizes data to a mean value 0 and standard deviation of 1.
                                verbose=False)) 
                                ], verbose=False)
    metadata_cols = ['name', 'artists', 'id'] #Final columns to be shown in dictionary of recommended songs.
    song_dict = flatten_dict_list(song_list)
    song_center = get_mean_vector(song_list, spotify_data) #Creates standardized average of songs for KMEANs clustering
    scaler = song_cluster_pipeline.steps[0][1] 
    X = spotify_data.select_dtypes(np.number)
    song_cluster_pipeline.fit(X) #Fit model
    song_cluster_labels = song_cluster_pipeline.predict(X)  
    data['cluster_label'] = song_cluster_labels

    if song_center.all() > 0: #Make sure song vector is true and contains values 
        scaled_data = scaler.transform(spotify_data[number_cols]) #Column values to be used in Clustering process
        scaled_song_center = scaler.transform(song_center.reshape(1, -1)) 
        distances = cdist(scaled_song_center, scaled_data, 'cosine') 
        index = list(np.argsort(distances)[:, :n_songs][0])
        
        rec_songs = spotify_data.iloc[index]
        rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])] #filtering maniuplated dataframe  
        return rec_songs[metadata_cols].to_dict(orient='records') #Returns a dictionary of songs with track name, artist name, and track id 
    else: 
        return f'Cannot perform StandardScaler() functions since no songs remain in song list'

#Add recommended songs to user generated playlist. The reccommended tracks will be saved to a playlist called 'Recommended Tracks'
def create_playlist(songs):

    song_list_ids = [reccommendation['id'] for reccommendation in songs]
    user_playlist = sp.user_playlists(username)
    id = False #User playlist id 
    playlist_names =  [playlist['name'] for playlist in user_playlist['items']]
    for single_playlist in user_playlist['items']:
        if single_playlist['name'] == 'Recommended Tracks':
            id = single_playlist['id']
            break
    if id:
        try:
            sp.user_playlist_remove_all_occurrences_of_tracks(username,id,song_list_ids) #Delete duplicate tracks if they exist
            print("Duplicates tracks removed from playlist")
        except:
            pass
        sp.playlist_add_items(id, song_list_ids)
        return print("Songs were added to an existing playlist. Here are the added songs:\n", songs)
    else:
        if 'Recommended Tracks' not in playlist_names:  #Create a new playlist if Recommended Track playlist doesn't exist
            new_playlist = sp.user_playlist_create(username, 'Recommended Tracks', public=True, description='Playlist of Similar Songs recommended from spotify data and songs list')
            id = new_playlist['id']
            sp.playlist_add_items(id, song_list_ids)

            return print("New playlist named Recommended Tracks was created. These songs were added:\n", songs)


# Example:  how to enter songs into recommendation system manually with a song dataset.
# resultant = recommend_songs([{'name': 'Drunk in Love'},
#                 {'name': 'Smells Like Teen Spirit'},
#                 {'name': 'No Church in the wild'},
#                 {'name': 'Switch it up'},
#                 {'name': 'Rich spirit'}],  data)
resultant = recommend_songs(top_5, data)
create_playlist(resultant)