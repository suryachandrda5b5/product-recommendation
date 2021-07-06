from django.shortcuts import render,redirect
from django.http import JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import warnings
#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

import numpy as np

# Create your views here.


warnings.filterwarnings('ignore')

# Load CSV Metadata
metadata = pd.read_csv('datav2.csv', low_memory=False)
metadata['index'] = metadata.index
bins = [-1,500,1000,1500,2000,2500,3000,3500,100000]
labels = ["<500","500-1000","1000-1500","1500-2000","2000-2500","2500-3000","3000-3500",">3500"]
metadata['NetPrice_bracket'] = pd.cut(metadata['NetPrice'], bins, labels=labels)

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
metadata['OpportunityId'] = metadata['OpportunityId'].fillna('')
metadata['AccountName'] = metadata['AccountName'].fillna('')
metadata['ProductID'] = metadata['ProductID'].fillna('')
# metadata['AccountName'] = metadata['AccountName'].fillna('')
metadata['features'] = metadata['OpportunityId'] + metadata['AccountName'] + metadata['ProductID'] + metadata['OpptyName']
# metadata['features'] = metadata['AccountName']

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['features'])
tfidf.get_feature_names()[0:12038]

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(metadata.index, index=metadata['ProductName']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the Product Names that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all Products with that ProductInput
    sim_scores = list(enumerate(cosine_sim[idx]))
    #print(sim_scores)
    # Sort the Products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: np.sum(x[1]), reverse=True)
    #print(sim_scores)
    # Get the list of the 10 most similar Products
    sim_scores = sim_scores[1:10]
    #print(sim_scores)
    # Get the ProductName indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar Products
    return metadata['ProductID'].iloc[movie_indices]


@csrf_exempt
def index(request):
    if request.method == "POST":
        received_json_data = json.loads(request.body.decode("utf-8"))
        name = received_json_data["name"]
        context = {name : str(get_recommendations(name))}
        print("HERE", context)
    else:
        context = { "response" : "POST Response Expected"}

    return JsonResponse(context)

def firstPage(request):
    context = { "response" : "POST Response Expected"}
    return render(request,'app1/first.html',context)

def about(request):
    context = { "response" : "POST Response Expected"}
    return render(request,'app1/about.html',context)

def contact(request):
    context = { "response" : "POST Response Expected"}
    return render(request,'app1/contact.html',context)