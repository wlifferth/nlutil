import wikipedia as wiki

from collections import Counter
from nltk import word_tokenize


def tfidf(document, corpus, tokenizer=None, priors=1):
    """
    Produces a dictionary mapping a token to a tfidf value given a document and a corpus.
    Accepts a document and a corpus each in the form of a space delmited string or an iterable of
    token strings. That is to say, you may provide a tokenized document and corpus, otherwise it
    will be tokenized for you. It is assumed that the document is _not_ part of the corpus, so
    document counts will be added to corpus counts. If the document is already part of the corpus,
    this will effect the final tfidf representation.
    """
    document_counter = Counter()
    corpus_counter = Counter()
    tfidf_dict = dict()

    if not isinstance(corpus, str):
        corpus_counter.update(corpus)
    else:
        corpus_counter.update(tokenize(corpus))

    if not isinstance(document, str):
        document_counter.update(document)
        corpus_counter.update(document)
    else:
        document_counter.update(tokenize(document))
        corpus_counter.update(tokenize(document))

    for word, count in document_counter.most_common():
        tfidf_dict[word] = count / (corpus_counter[word] + priors)
    return tfidf_dict

def tokenize(text):
    tokens = [word.lower() for word in word_tokenize(text)]
    return list(filter(lambda word: word.isalpha(), tokens))

def quick_corpus(term, results=10):
    topics = wiki.search('Democracy', results=results)
    tokens = []
    for topic in topics:
        print("Downloading {}:{}...".format(topic, ' '*(max(40 - len(topic), 1))), end='\r')
        tokens += tokenize(wiki.page(topic).content)
        print("Downloading {}:{}DONE".format(topic, ' '*(max(40 - len(topic), 1))))
    return tokens
