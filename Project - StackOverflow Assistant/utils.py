import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'data/word_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    good_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = good_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.
    Args:
      embeddings_path - path to the embeddings file.
    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """

    starspace_embeddings={}
    dim=0
    for line in open(embeddings_path):
    	tokenizer = nltk.tokenize.WhitespaceTokenizer()
    	emb = tokenizer.tokenize(line)
    	dim = len(emb)-1
    	starspace_embeddings[emb[0]] = [float(emb[i]) for i in range(1, len(emb))]
    return starspace_embeddings, dim


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""

    res = np.zeros(dim)
    if len(question) == 0:
    	return res
    else:
    	tokenizer = nltk.tokenize.WhitespaceTokenizer()
    	tokens = tokenizer.tokenize(question)
    	count = 0
    	for token in tokens:
    		if token in embeddings:
    			res = res + embeddings[token]
    			count += 1
    	if count == 0:
    		return res
    	else:
    		return res/count


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
