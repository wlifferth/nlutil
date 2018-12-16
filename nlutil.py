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
    blacklist = set()
    finished_topics = set()
    tokens = list()
    unfinished_topics = set(wiki.search(term, results=results))
    while unfinished_topics:
        topic = unfinished_topics.pop()
        print("Downloading {}:{}...".format(topic, ' '*(max(40 - len(topic), 1))), end='\r')
        try:
            topic_tokens = tokenize(wiki.page(topic).content)
        except wiki.exceptions.DisambiguationError:
            blacklist.add(topic)
            # We now have a blacklisted topic, so we'll need to grab one extra in the future
            results += 1
            new_topics = wiki.search(term, results=results)
            for new_topic in new_topics:
                if new_topic not in blacklist and new_topic not in finished_topics and new_topic not in unfinished_topics:
                    print("Note: because {} was a disambiguation page, we're not adding it to the corpus, and trying {} instead.".format(topic, new_topic))
                    unfinished_topics.add(new_topic)
        else:
            tokens += topic_tokens
            finished_topics.add(topic)
            print("Downloading {}:{}DONE".format(topic, ' '*(max(40 - len(topic), 1))))
    return tokens
