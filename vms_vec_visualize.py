# Takahashi Tokenizer
import re, io, os
import urllib.request
from collections import defaultdict
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from wordcloud import WordCloud


def tokenize(data):
	index = defaultdict(str)

	with urllib.request.urlopen(data) as file:
		for line in file.read().decode('latin-1').splitlines():
			# pull out takahashi lines
			m = re.match(r'^<(f.*?)\..*;H> +(\S.*)$', line)
			if not m:
				continue

			transcription = m.group(2)
			pg = str(m.group(1))

			# ignore entire line if it has a {&NNN} or {&.} code
			if re.search(r'\{&(\d|\.)+\}', transcription):
				continue

			# remove extraneous chracters ! and %
			s = transcription.replace("!", "").replace("%", "")

			# delete all end of line {comments} (between one and three observed)
			# ...with optional line terminator
			# allow 0 occurences to remove end-of-line markers (- or =)
			s = re.sub(r'([-=]?\{[^\{\}]+?\}){0,3}[-=]?\s*$', "", s)

			# delete start of line {comments} (single or double)
			s = re.sub(r'^(\{[^\{\}]+?\}){1,2}', "", s)

			# simplification: tags preceeded by -= are word breaks
			s = re.sub(r'[-=]\{[^\{\}]+?\}', '.', s)

			# these tags are nulls
			# plant is a null in one case where it is just {plant}
			# otherwise (above) it is a word break
			# s = re.sub(r'\{(fold|crease|blot|&\w.?|plant)\}', "", s)
			# simplification: remaining tags in curly brackets
			s = re.sub(r'\{[^\{\}]+?\}', '', s)

			# special case .{\} is still a word break
			s = re.sub(r'\.\{\\\}', ".", s)

			# split on word boundaries
			# exclude null words ('')
			words = [str(w) for w in s.split(".") if w]
			paragraph = ' '.join(words).lstrip()

			index[pg] += (paragraph)

	return index

def vis_vectorizer(documents):
    # do tfidf ~magic~
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
    vms_tfidf = tfidf_vectorizer.fit_transform(documents).todense()
    
    # grab the first document vector
    #first_vec = vms_tfidf[0]
    #df = pd.DataFrame(first_vec, index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
    #df.sort_values(by=["tfidf"], ascending=False)
    #df.to_csv("./out/first_vec.csv")
   
    pca = PCA(n_components=2).fit(vms_tfidf)
    data2D = pca.transform(vms_tfidf)
    plt.scatter(data2D[:,0], data2D[:,1])#, c=documents.target)
    plt.show()


def build_vectorizer(documents, mapping):
	tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
	tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2)
	vms_tfidf = tfidf_vectorizer.fit_transform(documents)
	vms_tf= tf_vectorizer.fit_transform(documents)
	dir_path = os.path.dirname(os.path.realpath(__file__))
	models_path = "{}/{}".format(dir_path, "models")

	with open("{}/tfidf_vectorizer.pk".format(models_path),"wb") as f: # vectorizers 
		pickle.dump(tfidf_vectorizer, f)
	with open("{}/tf_vectorizer.pk".format(models_path),"wb") as f:
		pickle.dump(tfidf_vectorizer, f)
	
	with open("{}/vms_tfidf.pk".format(models_path),"wb") as f: # models
		pickle.dump(vms_tfidf, f)
	with open("{}/vms_tf.pk".format(models_path),"wb") as f:
		pickle.dump(vms_tf, f)

	with open("{}/vms_mapping.pk".format(models_path), "wb") as f: # mapping
		pickle.dump(mapping, f)

def make_cloud(documents):
    # take all the documents and concatenate them into one long comma-separated string
    long_string = ','.join(list(documents))
    # Create a WordCloud object
    wordcloud = WordCloud(width=600, height=800, background_color="white", max_words=5000, contour_width=3, contour_color='steelblue', colormap='winter')# Generate a word cloud
    wordcloud.generate(long_string)# Visualize the word cloud
    wordcloud.to_file('./out/cloud.png')


    fig = plt.figure(figsize=(6,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    index = tokenize("https://raw.githubusercontent.com/rachelbari/voynich-topic-modeling/master/data/text16e6.evt")
    documents = [index[key] for key in index.keys()]
    vms_mapping = [k for k in index.keys()]
    #vis_vectorizer(documents, vms_mapping) 
    make_cloud(documents)
