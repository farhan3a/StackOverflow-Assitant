import os
from sklearn.metrics.pairwise import pairwise_distances_argmin

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer
from utils import *
import scipy


class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        # HINT: you have already implemented a similar routine in the 3rd assignment.
        
        #### YOUR CODE HERE ####
        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim)
        best_thread = pairwise_distances_argmin(np.array(question_vec).reshape(1,-1), thread_embeddings, metric='cosine')[0]
        
        return thread_ids[best_thread]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.paths = paths
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)
        self.__init_chitchat_bot()

    def __init_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        # Hint: you might want to create and train chatterbot.ChatBot here.
        # Create an instance of the ChatBot class.
        # Create a trainer (chatterbot.trainers.ChatterBotCorpusTrainer) for the ChatBot.
        # Train the ChatBot with "chatterbot.corpus.english" param.
        
        ########################
        #### YOUR CODE HERE ####
        ########################
        self.chatbot = ChatBot("Farhan")
        self.trainer = ChatterBotCorpusTrainer(self.chatbot)
        self.trainer.train("chatterbot.corpus.english")
        self.trainer = ListTrainer(self.chatbot)
        self.trainer.train([
            "Hey",
            "Hi there!",
            "How are you doing?",
            "I am good great! How about you?"     
        ])
        
        # remove this when you're done
        # raise NotImplementedError(
        #     "Open dialogue_manager.py and fill with your code. In case of Google Colab, download"
        #     "(https://github.com/hse-aml/natural-language-processing/blob/master/project/dialogue_manager.py), "
        #     "edit locally and upload using '> arrow on the left edge' -> Files -> UPLOAD")
       
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        
        #### YOUR CODE HERE ####
        # self.intent_recognizer = unpickle_file(self.paths['INTENT_RECOGNIZER'])
        # self.tfidf_vectorizer = unpickle_file(self.paths['TFIDF_VECTORIZER'])
        prepared_question = text_prepare(question)
        features = self.tfidf_vectorizer.transform([prepared_question])
        features = scipy.sparse.csr_matrix.toarray(features)
        intent = self.intent_recognizer.predict(np.array(features).reshape(1,-1))[0]

        # Chit-chat part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            #### YOUR CODE HERE ####
            response = self.chatbot.get_response(question)
            return response
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            #### YOUR CODE HERE ####
            # self.tag_classifier = unpickle_file(self.paths['TAG_CLASSIFIER'])
            tag = self.tag_classifier.predict(np.array(features).reshape(1,-1))[0]
            
            # Pass prepared_question to thread_ranker to get predictions.
            #### YOUR CODE HERE ####
            thread_id = self.thread_ranker.get_best_thread(question, tag)
            
            return self.ANSWER_TEMPLATE % (tag, thread_id)
