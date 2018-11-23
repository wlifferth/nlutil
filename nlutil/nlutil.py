from collections import Counter
from nltk import word_tokenize



def tfidf(document, corpus, tokenizer=None, priors=1):
    """
    Produces a dictionary mapping a token to a tfidf value given a document and a corpus.
    Accepts a document in the form of a space delmited string and a corpus either in the form of
    a single space delmited string, or an iterable of space delmited strings. It is assumed that
    the document is _not_ part of the corpus, so document counts will be added to corpus counts.
    If the document is already part of the corpus, this will effect the final tfidf
    representation.
    """
    document_counter = Counter(tokenize(document))
    corpus_counter = Counter()
    tfidf_dict = dict()
    if not isinstance(corpus, str):
        for corpus_doc in corpus:
            corpus_counter.update(tokenize(corpus_doc))
    else:
        corpus_counter.updadte(tokenize(corpus))
    corpus_counter.update(tokenize(document))
    for word, count in document_counter.most_common():
        tfidf_dict[word] = count / (corpus_counter[word] + priors)
    return tfidf_dict

def tokenize(text):
    tokens = [word.lower() for word in word_tokenize(text)]
    return list(filter(lambda word: word.isalpha(), tokens))
