# Spam classifier using NLP

## Explanation and code

### Install all the packages

#Install Packages\
pip install wordcloud\
%matplotlib inline\
import matplotlib.pyplot as plt\
import csv\
import sklearn\
import pickle\
from wordcloud import WordCloud\
import pandas as pd\
import numpy as np\
import nltk\
from nltk.corpus import stopwords\
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer\
from sklearn.tree import DecisionTreeClassifier\
from sklearn.model_selection import GridSearchCV,train_test_split,StratifiedKFold,cross_val_score,learning_curve

### Read the data
* Importing the Dataset spam.csv.We need to remove the unwanted columns.

**data = pd.read_csv('data/spam.csv', encoding='latin-1')**

### The data has ham and spam messages labelled.

* The distribution of ham and spam messages looks like:-
* Overall distribution of spam and ham messages

#Overall length of length of spam and ham messages\
**data.hist(column='length', by='label', bins=100, figsize=(20,7))**

### Creating a corpus of spam and ham messages

ham_words = ''\
spam_words = ''\
import nltk\
nltk.download('punkt')\
#### Creating a corpus of spam messages
for val in data[data['label'] == 'spam'].text:\

      text = val.lower()\
      tokens = nltk.word_tokenize(text)\
      for words in tokens:\
          spam_words = spam_words + words + ' '\
#### Creating a corpus of ham messages 

for val in data[data['label'] == 'ham'].text:\

      text = val.lower()\
      tokens = nltk.word_tokenize(text)\
      for words in tokens:\
          ham_words = ham_words + words + ' '

### Creating Spam and Ham word clouds
Creating a word cloud of spam messages.  Word Cloud is a data visualization technique used for representing text data in which the size of each word indicates its frequency or importance.

spam_wordcloud = WordCloud(width=500, height=300).generate(spam_words)\
ham_wordcloud = WordCloud(width=500, height=300).generate(ham_words)\

#### Spam Word cloud

plt.figure( figsize=(10,8), facecolor='w')\
plt.imshow(spam_wordcloud)\
plt.axis("off")\
plt.tight_layout(pad=0)\
plt.show()\

#### Creating Ham wordcloud
plt.figure( figsize=(10,8), facecolor='g')\
plt.imshow(ham_wordcloud)\
plt.axis("off")\
plt.tight_layout(pad=0)\
plt.show()

### Data pre-processing of SMS Spam
* Removing punctuations and stopwords from the text data.

import string\
def text_process(text):\
text = text.translate(str.maketrans('', '', string.punctuation))\
text = [word for word in text.split() if word.lower() not in stopwords.words('english')]

      return " ".join(text)\
data['text'] = data['text'].apply(text_process)

### Converting text to vectors
Now we will proceed by converting the text to vectors for the model to easily classify it. Two such techniques are Bag of Words and TF-IDF Vectorizer. The basic requirements would be it should not result in the sparse matrix and it should retain most of the linguistic information. The problem with a bag of words is that it assigns the same importance value(Weights) to all the words. This is resolved when we TF-IDF as it assigns different weights to the words.

#### Text to Vector
def text_to_vector(text):\
    word_vector = np.zeros(vocab_size)\
    for word in text.split(" "):
    
        if word2idx.get(word) is None:
            continue
        else:
            word_vector[word2idx.get(word)] += 1
    return np.array(word_vector)
      #Convert all titles to vectors
      word_vectors = np.zeros((len(text), len(vocab)), dtype=np.int_)
      for i, (_, text_) in enumerate(text.iterrows()):
       word_vectors[i] = text_to_vector(text_[0])

#### Converting words to vector using TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer\
vectorizer = TfidfVectorizer()\
vectors = vectorizer.fit_transform(data['text'])\
vectors.shape\
#features = word_vectors\
features = vectors

### Split the data using sklearn library
* Splitting the data into train test and applying machine learning models to it. Further, we will split the data into training sets and testing sets. 85% of data were used for training and 15% for testing purposes.

#### Train-Test Split
**X_train, X_test, y_train, y_test = train_test_split(features, data['label'], test_size=0.15, random_state=111)**

## Training using multiple machine learning models

from sklearn.linear_model import LogisticRegression\
from sklearn.svm import SVC\
from sklearn.naive_bayes import MultinomialNB\
from sklearn.tree import DecisionTreeClassifier\
from sklearn.neighbors import KNeighborsClassifier\
from sklearn.ensemble import RandomForestClassifier\
from sklearn.metrics import accuracy_score\
svc = SVC(kernel='sigmoid', gamma=1.0)\
knc = KNeighborsClassifier(n_neighbors=49)\
mnb = MultinomialNB(alpha=0.2)\
dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111)\
lrc = LogisticRegression(solver='liblinear', penalty='l1')\
rfc = RandomForestClassifier(n_estimators=31, random_state=111)\
clfs = {'SVC' : svc,'KN' : knc, 'NB': mnb, 'DT': dtc, 'LR': lrc, 'RF': rfc}\
def train(clf, features, targets):

    clf.fit(features, targets)
    
def predict(clf, features):

    return (clf.predict(features))
    
pred_scores_word_vectors = []\
for k,v in clfs.items():

    train(v, X_train, y_train)
    pred = predict(v, X_test)
    pred_scores_word_vectors.append((k, [accuracy_score(y_test , pred)]))

### Model Prediction
def find(x):

    if x == 1:
        print ("Message is SPAM")
    else:
        print ("Message is NOT Spam")
        
text = ["Free tones Hope you enjoyed your new content"]\
integers = vectorizer.transform(text)\
x = mnb.predict(integers)[0]\
find(x) 

![two1.png](attachment:two1.png)

### Final Thoughts
We used various machine learning algorithms to classify the text message and compared accuracy set across these models. Naive Bayes classifier gives the best result among all with an accuracy of over 98%.

# TF IDF , Bag of words and WORD2VEC

### Bag of Words (BoW) Model

The Bag of Words (BoW) model is the simplest form of text representation in numbers. Like the term itself, we can represent a sentence as a bag of words vector (a string of numbers).\
Let’s recall the three types of movie reviews we saw earlier:\
Review 1: This movie is very scary and long\
Review 2: This movie is not scary and is slow\
Review 3: This movie is spooky and good\
We will first build a vocabulary from all the unique words in the above three reviews. The vocabulary consists of these 11 words: ‘This’, ‘movie’, ‘is’, ‘very’, ‘scary’, ‘and’, ‘long’, ‘not’,  ‘slow’, ‘spooky’,  ‘good’.\
We can now take each of these words and mark their occurrence in the three movie reviews above with 1s and 0s. This will give us 3 vectors for 3 reviews

![v1.webp](attachment:v1.webp)

Vector of Review 1: [1 1 1 1 1 1 1 0 0 0 0]

Vector of Review 2: [1 1 2 0 0 1 1 0 1 0 0]

Vector of Review 3: [1 1 1 0 0 0 1 0 0 1 1]

And that’s the core idea behind a Bag of Words (BoW) model.

**Term Frequency (TF)**
* Let’s first understand Term Frequent (TF). It is a measure of how frequently a term, t, appears in a document, d:

![v2.jpg](attachment:v2.jpg)

Here,

Vocabulary: ‘This’, ‘movie’, ‘is’, ‘very’, ‘scary’, ‘and’, ‘long’, ‘not’,  ‘slow’, ‘spooky’,  ‘good’

Number of words in Review 2 = 8

TF for the word ‘this’ = (number of times ‘this’ appears in review 2)/(number of terms in review 2) = 1/8

Similarly,

TF(‘movie’) = 1/8\
TF(‘is’) = 2/8 = 1/4\
TF(‘very’) = 0/8 = 0\
TF(‘scary’) = 1/8\
TF(‘and’) = 1/8\
TF(‘long’) = 0/8 = 0\
TF(‘not’) = 1/8\
TF(‘slow’) = 1/8\
TF( ‘spooky’) = 0/8 = 0\
TF(‘good’) = 0/8 = 0

![v3.webp](attachment:v3.webp)

**Inverse Document Frequency (IDF)**
* IDF is a measure of how important a term is. We need the IDF value because computing just the TF alone is not sufficient to understand the importance of words:

![v4.jpg](attachment:v4.jpg)

We can calculate the IDF values for the all the words in Review 2:\

IDF(‘this’) =  log(number of documents/number of documents containing the word ‘this’) = log(3/3) = log(1) = 0\

Similarly,\

IDF(‘movie’, ) = log(3/3) = 0\
IDF(‘is’) = log(3/3) = 0\
IDF(‘not’) = log(3/1) = log(3) = 0.48\
IDF(‘scary’) = log(3/2) = 0.18\
IDF(‘and’) = log(3/3) = 0\
IDF(‘slow’) = log(3/1) = 0.48\
We can calculate the IDF values for each word like this. Thus, the IDF values for the entire vocabulary would be:



![v5.webp](attachment:v5.webp)

Hence, we see that words like “is”, “this”, “and”, etc., are reduced to 0 and have little importance; while words like “scary”, “long”, “good”, etc. are words with more importance and thus have a higher value.

We can now compute the TF-IDF score for each word in the corpus. Words with a higher score are more important, and those with a lower score are less important:



![v6.jpg](attachment:v6.jpg)

We can now calculate the TF-IDF score for every word in Review 2:

TF-IDF(‘this’, Review 2) = TF(‘this’, Review 2) * IDF(‘this’) = 1/8 * 0 = 0

Similarly,

TF-IDF(‘movie’, Review 2) = 1/8 * 0 = 0\
TF-IDF(‘is’, Review 2) = 1/4 * 0 = 0\
TF-IDF(‘not’, Review 2) = 1/8 * 0.48 = 0.06\
TF-IDF(‘scary’, Review 2) = 1/8 * 0.18 = 0.023\
TF-IDF(‘and’, Review 2) = 1/8 * 0 = 0\
TF-IDF(‘slow’, Review 2) = 1/8 * 0.48 = 0.06\
Similarly, we can calculate the TF-IDF scores for all the words with respect to all the reviews:

![v7.webp](attachment:v7.webp)

import nltk  
import numpy as np  
import random  
import string

import bs4 as bs  
import urllib.request  
import re  

raw_html = urllib.request.urlopen('https://en.wikipedia.org/wiki/Natural_language_processing')  
raw_html = raw_html.read()

article_html = bs.BeautifulSoup(raw_html, 'lxml')

article_paragraphs = article_html.find_all('p')

article_text = ''

for para in article_paragraphs:

    article_text += para.text
    

corpus = nltk.sent_tokenize(article_text)

for i in range(len(corpus )):

    corpus [i] = corpus [i].lower()
    corpus [i] = re.sub(r'\W',' ',corpus [i])
    corpus [i] = re.sub(r'\s+',' ',corpus [i])

print(corpus[30])

**Output:**

in the 2010s representation learning and deep neural network style machine learning methods became widespread in natural language processing due in part to a flurry of results showing that such techniques 4 5 can achieve state of the art results in many natural language tasks for example in language modeling 6 parsing 7 8 and many others 

## Word2Vec

import bs4 as bs\
import urllib.request\
import re\
import nltk\

scrapped_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Artificial_intelligence')\
article = scrapped_data .read()\

parsed_article = bs.BeautifulSoup(article,'lxml')

paragraphs = parsed_article.find_all('p')

article_text = ""

for p in paragraphs:

    article_text += p.text
### Cleaing the text
processed_article = article_text.lower()\
processed_article = re.sub('[^a-zA-Z]', ' ', processed_article )\
processed_article = re.sub(r'\s+', ' ', processed_article)

### Preparing the dataset
all_sentences = nltk.sent_tokenize(processed_article)

all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

### Removing Stop Words
from nltk.corpus import stopwords\
for i in range(len(all_words)):

    all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]

### Creating Word2Vec Model

from gensim.models import Word2Vec

word2vec = Word2Vec(all_words, min_count=2)

vocabulary = word2vec.wv.vocab

print(vocabulary)

sim_words = word2vec.wv.most_similar('intelligence')

('ai', 0.7124934196472168)\
('human', 0.6869025826454163)\
('artificial', 0.6208730936050415)\
('would', 0.583903431892395)\
('many', 0.5610555410385132)\
('also', 0.5557990670204163)\
('learning', 0.554862380027771)\
('search', 0.5522681474685669)\
('language', 0.5408136248588562)\
('include', 0.5248900055885315)

## Conclusion
we implemented a Word2Vec word embedding model with Python's Gensim Library. We did this by scraping a Wikipedia article and built our Word2Vec model using the article as a corpus


```python

```
