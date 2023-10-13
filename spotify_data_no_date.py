import matplotlib.pyplot as plt
import numpy as np
import json
import csv

import spotipy
from spotipy import oauth2
import spotipy.util as util

token = util.oauth2.SpotifyClientCredentials(client_id='81da2e7e661749f0a86024a8c2a04dd3', client_secret='3d4324c79f2a441bb64c7a55d6a226fa')
cache_token = token.get_access_token()
spotify = spotipy.Spotify(cache_token)
sp = spotipy.Spotify(auth=cache_token)

artist_l = []
track_l = []
score_l = []
label_l = []
with open('score_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count != 0:
            artist_l.append(row[0])
            track_l.append(row[1])
            score_l.append(row[2])
            label_l.append(row[3])
        line_count += 1

m = len(artist_l)
artist_fl = []
track_fl = []
score_fl = []
label_fl = []
danceability = np.zeros([m,1])
energy = np.zeros([m,1])
key = np.zeros([m,1])
loudness = np.zeros([m,1])
mode = np.zeros([m,1])
speechiness = np.zeros([m,1])
acousticness = np.zeros([m,1])
instrumentalness = np.zeros([m,1])
liveness = np.zeros([m,1])
valence = np.zeros([m,1])
tempo = np.zeros([m,1])

j = 0
for i in range(0,m):
    artist = artist_l[i]
    track = track_l[i]
    score = score_l[i]
    label = label_l[i]
    track_info = sp.search(q='artist:' + artist + ' track:' + track, type='track')
    track_id = track_info['tracks']
    track_id2 = track_id['items']
    if track_id2 != []:
            track_id3 = track_id2[0]
            track_id4 = track_id3['id']
            feat_t = sp.audio_features(tracks=track_id4)
            feat = feat_t[0]
            artist_fl.append(artist_l[i])
            track_fl.append(track_l[i])
            score_fl.append(score_l[i])
            label_fl.append(label_l[i])
            danceability[j] = feat['danceability']
            energy[j] = feat['energy']
            key[j] = feat['key']
            loudness[j] = feat['loudness']
            mode[j] = feat['mode']
            speechiness[j] = feat['speechiness']
            acousticness[j] = feat['acousticness']
            instrumentalness[j] = feat['instrumentalness']
            liveness[j] = feat['liveness']
            valence[j] = feat['valence']
            tempo[j] = feat['tempo']
            j += 1

with open('complete_project_data_no_date.csv', mode='w') as spotify_features:
    #sp_writer = csv.writer(spotify_features, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')
    sp_writer = csv.writer(spotify_features, delimiter=',', lineterminator = '\n')
    sp_writer.writerow(['Artist','Track','ArtistScore','Danceability','Energy','Key','Loudness','Mode','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo','Label'])
    for i in range(0,j-1):
        sp_writer.writerow([artist_fl[i],track_fl[i],score_fl[i],danceability[i][0],energy[i][0],key[i][0],loudness[i][0],mode[i][0],speechiness[i][0],acousticness[i][0],instrumentalness[i][0],liveness[i][0],valence[i][0],tempo[i][0],label_fl[i]])
