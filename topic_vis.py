import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.preprocessing import Normalizer

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

def print_results(model, feature_names, no_top_words, topics, mapping):
    display_topics(model, feature_names, no_top_words)
    #print("\n")
    #for t in range(len(topics)):
    #    print("{}: {}".format(mapping[t], topics[t].argmax()))

def nmf(tfidf_vectorizer, vms_tfidf, mapping, num_topics):
    nmf = NMF(n_components=num_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(vms_tfidf)
    nmf_topics = nmf.transform(vms_tfidf)
    print("NMF\n")
    print_results(nmf, tfidf_vectorizer.get_feature_names(), 10, nmf_topics, mapping)

def lda(tf_vectorizer, vms_tf, mapping, num_topics):
    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(vms_tf)
    lda_topics = lda.transform(vms_tf)
    print("LDA\n")
    print_results(lda, tf_vectorizer.get_feature_names(), 10, lda_topics, mapping)

def lsa(tfidf_vectorizer, vms_tfidf, mapping, num_topics):
    svd_model = TruncatedSVD(n_components=num_topics)
    lsa_topics = svd_model.fit_transform(vms_tfidf)
    print("LSA\n")
    print_results(svd_model, tfidf_vectorizer.get_feature_names(), 10, lsa_topics, mapping)



if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    models_path = "{}/{}".format(dir_path, "models")

    with open("{}/tfidf_vectorizer.pk".format(models_path), "rb") as f:
        tf_vectorizer = pickle.load(f)
    with open("{}/vms_tf.pk".format(models_path), "rb") as f:
        vms_tf = pickle.load(f)

    with open("{}/vms_mapping.pk".format(models_path), "rb") as f:
        vms_mapping = pickle.load(f)
    with open("{}/tf_vectorizer.pk".format(models_path), "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    with open("{}/vms_tfidf.pk".format(models_path), "rb") as f:
        vms_tfidf = pickle.load(f)

    num_topics = 4

    lda(tf_vectorizer, vms_tf, vms_mapping, num_topics)
    nmf(tfidf_vectorizer, vms_tf, vms_mapping, num_topics)
    lsa(tfidf_vectorizer, vms_tf, vms_mapping, num_topics)
    
