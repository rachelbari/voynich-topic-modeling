import pickle
from sklearn.metrics.pairwise import cosine_similarity

def print_similarities():
    with open("vms_tfidf.pk", "rb") as f:
        vms_tfidf = pickle.load(f)
    with open("vms_mapping.pk", "rb") as f:
        vms_mapping = pickle.load(f)
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
    print_similarities()