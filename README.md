# StackOverflow Assitant bot

*An interactive conversational chatbot that assists with a search on StackOverflow and holds a dialogue using a dialogue manager*

### Overview:
The dialogue chat bot is able to perform the following task:-
* answer programming-related questions (using StackOverflow dataset);
* chit-chat and simulate dialogue on all non programming-related questions.
For a chit-chat mode I have used a pre-trained neural network engine available from [ChatterBot](https://github.com/gunthercox/ChatterBot).

**Note:** The virtual AI bot has been deployed and integrated on Telegram messenger and hosted on Amazon AWS cloud server - [StackOverFlow Assistant bot](https://web.telegram.org/#/im?p=@Farhan3a_bot).

![](https://imgs.xkcd.com/comics/twitter_bot.png)
©[xkcd](https://xkcd.com)

**Note:** 
The project is buit as part of the final project of the Natural Language Processing (NLP) course of the Advanced Machine Learning Specialization offered by the National Research University Higher School of Economics through Coursera platform. The other assignments for each week are present in the respective folders covering the following problems:- 
* Week1: Predict tags on StackOverflow with linear models
* Week2: Recognize named entities on Twitter with LSTMs
* Week3: Find duplicate questions on StackOverflow by their embeddings
* Week4: Learn to calculate with seq2seq model

### Data Description:

To detect *intent* of users questions we will need two text collections:
- `tagged_posts.tsv` — StackOverflow posts, tagged with one programming language (*positive samples*).
- `dialogues.tsv` — dialogue phrases from movie subtitles (*negative samples*).

For those questions, that have programming-related intent, I have proceeded as follows- predict programming language (only one tag per question allowed here) and rank candidates within the tag using embeddings.
For the ranking part, we will need:
- `word_embeddings.tsv` — It has word embeddings that have been trained with StarSpace package from the 3rd assignment.

As a result of this notebook, we should obtain the following new objects that we will then use in the running bot:
- `intent_recognizer.pkl` — intent recognition model;
- `tag_classifier.pkl` — programming language classification model;
- `tfidf_vectorizer.pkl` — vectorizer used during training;
- `thread_embeddings_by_tags` — folder with thread embeddings, arranged by tags.
    
## Part I. Intent and language recognition

We want to write a bot, which will not only **answer programming-related questions**, but also will be able to **maintain a dialogue**. We would also like to detect the *intent* of the user from the question (we could have had a 'Question answering mode' check-box in the bot, but it wouldn't fun at all, would it?). So the first thing we need to do is to **distinguish programming-related questions from general ones**.

It would also be good to predict which programming language a particular question referees to. By doing so, we will speed up question search by a factor of the number of languages (10 here), and exercise our *text classification* skill a bit.

### Data preparation
At first, I have preprocessed the texts and did TF-IDF tranformations as I have done in the first assignment (Predict tags on StackOverflow with linear models). In addition, I have also dumped the TF-IDF vectorizer with pickle to use it later in the running bot.

### Intent recognition
I have done a binary classification on TF-IDF representations of texts. Labels will be either `dialogue` for general questions or `stackoverflow` for programming-related questions. Firstly, I have prepared the data for this task:
- concatenated `dialogue` and `stackoverflow` examples into one sample
- split it into train and test in proportion 9:1, using *random_state=0* for reproducibility
- transform it into TF-IDF features

Later, I have trained the **intent recognizer** using LogisticRegression on the train set with the following parameters: *penalty='l2'*, *C=10*, *random_state=0*. Here I have achieved an accuracy of 99%.

### Programming language classification 
Once the `stackoverflow` tag has been prediocted, I have trained one more classifier for the programming-related questions. It will predict exactly one tag (=programming language) and will be also based on Logistic Regression with TF-IDF features. 

First, I have prepared the data for this task by reusing the TF-IDF vectorizer. And then *Trained* the **tag classifier** using OneVsRestClassifier wrapper over LogisticRegression using the following parameters: *penalty='l2'*, *C=5*, *random_state=0*.

## Part II. Ranking  questions with embeddings

To find a relevant answer (a thread from StackOverflow) on a question I have used vector representations to calculate similarity between the question and existing threads. For this I have first transformed the question into a vector as I did in assignment 3.

However, it would be costly to compute such a representation for all possible answers in *online mode* of the bot (e.g. when bot is running and answering questions from many users). This is the reason why I have created a *database* with pre-computed representations. These representations will be arranged by non-overlaping tags (programming languages), so that the search of the answer can be performed only within one tag each time. This will make our bot even more efficient and allow not to store all the database in RAM. 

For the transformations, I have loaded StarSpace embeddings which were trained on Stack Overflow posts. These embeddings were trained in *supervised mode* for duplicates detection on the same corpus that is used in search. We can account on that these representations will allow us to find closely related answers for a question. 

Now for each `tag` I have created two data structures, which will serve as online search index:
* `tag_post_ids` — a list of post_ids with shape `(counts_by_tag[tag],)`. It will be needed to show the title and link to the thread;
* `tag_vectors` — a matrix with shape `(counts_by_tag[tag], embeddings_dim)` where embeddings for each answer are stored.

## Part III. Putting all together

In the end, I have combined everything that we have done so far and enabled the bot to maintain a dialogue. The bot has learned to sequentially determine the intent and, depending on the intent, select the best answer.

**Note:** Other Important files in the project folder includes:-
- `utils.py` — contains functions for text preprocessing, loading embeddings and transforming questions to vectors;
- `dialogue_manager.py` — conatins classes for ranking questions with embeddings and fclasses for generating answers;
- `main_bot.py` — Contain classes which implements all the backend of the bot;
