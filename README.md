# voynich-topic-modeling

# How to use

**TFIDF Vectorizer** 
* The file ```vms_vectorize.py``` builds a "model" for for the tfidf vectorizer and a ```vms_mapping``` so you can look up VMS document names by row ID in the vms_tfidf vectorizer
* ```vms_tfidf.pk``` is the vectorized matrix of VMS documents. Once you load, ```vms_tfidf.pk```, you can use the word vectorizers!
    - To load: 
        ````
        with open("vms_tfidf.pk", "rb" as f: 
            vms_tfidf = picke.load(f)
        ````
    - vms_tfidf is a DxV matrix, where D is the number of documents (225) and V is the number of unique words in the vocabulary (9778)
    - if a word wi appears in document dk, there will be a tfidf weighted value for that word at ```vms_tfidf[dk][wi]```, otherwise the value is 0
* ```vms_mapping.pk``` maps document (row) ID to its folio number 
    - To load: 
        ````
        with open("vms_mapping.pk", "rb" as f: 
            vms_mapping = picke.load(f)
        ````
        - see ```cosine_similarity.py``` for usage

**Cosine Similarity** 
* ```cosine_similarities = cosine_similarity(vms_tfidf, vms_tfidf)``` creates a 225x225 matrix, where the value at index i, j is the similarity of document i and j
* To print the similarities, run ```python3 cosine_similarities.py```
