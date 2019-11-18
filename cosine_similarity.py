import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def print_similarities(vms_tfidf, vms_mapping):
    cosine_similarities = cosine_similarity(vms_tfidf, vms_tfidf)
    sim_scores = []

    for row in cosine_similarities:
        related_docs_indices = row.argsort()[::-1][1:6] #[::-1] reverse arg; grab top 5 (not including first)
        scores = row[related_docs_indices]
        related_docs_indices_2 = [vms_mapping[i] for i in related_docs_indices]
        sim_scores.append(list(zip(related_docs_indices_2, scores)))
        
    for doc in range(len(sim_scores)):
        print(vms_mapping[doc])
        print(sim_scores[doc])
        print("\n\n")

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    models_path = "{}/{}".format(dir_path, "models")
    with open("{}/vms_tfidf.pk".format(models_path), "rb") as f:
        vms_tfidf = pickle.load(f)
    with open("{}/vms_mapping.pk".format(models_path), "rb") as f:
        vms_mapping = pickle.load(f)

    print_similarities(vms_tfidf, vms_mapping)