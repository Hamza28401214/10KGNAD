import pandas as pd
import string
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.snowball import GermanStemmer
import re
import spacy
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

STOPWORDS = set(stopwords.words('german'))
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])



def remove_freqwords(text, FREQWORDS):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])


stemmer = GermanStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

nlp = spacy.load('models//de_core_news_md-2.3.0')
def lemmatizer(text):
     doc = nlp(text)
     return ' '.join([x.lemma_ for x in doc])


def preprocessing(text):
    cnt = Counter()
    for word in text.split():
        cnt[word] += 1
    FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])

    text = text.lower()
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = remove_freqwords(text,FREQWORDS)
    text = stem_words(text)
    text = lemmatizer(text)
    text = re.sub("\d+"," ",text)
    return text

# def vectorize(corpus):
#     corpus = [corpus]
#     tfidf = TfidfVectorizer(min_df=1, max_df=1, ngram_range=(1, 2), max_features=5000, norm='l2')
#     try:
#         features = tfidf.fit_transform(corpus)
#         vect = pd.DataFrame(features.todense(), columns=tfidf.get_feature_names())
#     except ValueError:
#         return render_template("index.html", msg="your message cannot be treated")
#
#     return vect
#
# def padd_sequence(vect):
#     """ vect : output of vectorize
#       type : dataframe
#     """
#     if len(vect) < 5000:
#         for i in range(0, (5000 - vect.shape[1])):
#             vect['oov' + str(i)] = 0
#     return vect