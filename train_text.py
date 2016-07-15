
# coding: utf-8

# ### Import Libraries for General Processing

# In[1]:

import matplotlib
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
matplotlib.use('Agg') 

import numpy as np
import os
import pandas as pd
from PIL import Image
import re

from collections import Counter
import ujson as json
from pathlib import Path
import random

# look at dataset
from imp import reload
import image_grid
reload(image_grid)

import warnings
warnings.filterwarnings("ignore")


# ### Import Libraries for Text Processing

# In[2]:

#import spacy.en
#from spacy.strings import StringStore, hash_string

import snowballstemmer
import nltk
from nltk.stem.porter import PorterStemmer


# ### Import Libraries for Machine Learning

# In[3]:

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import  RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfVectorizer, HashingVectorizer
)

#from cross_validation import cross_val_apply
#from stacked_classifier import StackedClassifier


# ### Loading VQA data

# In[4]:

def read_captions(caption_dir, section):
    ans = 'captions_%s2014.json' % section
    with (caption_dir / ans).open() as file_:
        ans_data = json.load(file_)
    caption_by_id = {}
    for answer in ans_data['annotations']:
        image = str(answer['image_id'])
        caption = answer['caption']
        caption_by_id[image] = str(caption)
    return caption_by_id


# In[5]:

train_captions = read_captions(Path('../Train'), 'train')
eval_captions = read_captions(Path('../Val'), 'val')


# In[6]:

eval_captions['4576']


# In[7]:

def reading_vqa_data(vqa_dir, section, captions):
    ans = 'mscoco_%s2014_annotations.json' % section
    with (vqa_dir / ans).open() as file_:
        ans_data = json.load(file_)
    caption_by_id = {}
    answers_by_id = {}
    for answer in ans_data['annotations']:
        image = str(answer['image_id'])
        mca = answer['multiple_choice_answer']
        caption = captions[image]
        caption_by_id[answer['question_id']] = caption
        answers_by_id[answer['question_id']] = mca
    filename = ('MultipleChoice_mscoco_'
                '%s2014_questions.json' % section)
    with (vqa_dir / filename).open() as file_:
        ques_data = json.load(file_)
    for question in ques_data['questions']:
        text = question['question']
        ques_id = question['question_id']
        options = question['multiple_choices']
        image_path = caption_by_id[ques_id]
        yield ques_id, caption_by_id[ques_id], text, options, answers_by_id[ques_id]


# In[8]:

IMAGE_SIZE = 64
train_data = list(reading_vqa_data(Path('Train'), 'train', train_captions))
eval_data = list(reading_vqa_data(Path('Val'), 'val', eval_captions))


# In[9]:

def get_answers(data, top_n=1000):
    freqs = Counter()
    ans2id = {}
    id2ans = {}
    ans2ques = {}
    id2ques = {}
    for ques_id, _, _2, _3, answer in data:
        freqs[answer] += 1
        ans2ques[answer] = ques_id
    most_common = freqs.most_common(top_n)
    #most_common = freqs.most_common()
    for i, (string, _) in enumerate(most_common):
        ans2id[string] = i+1
        id2ans[i+1] = string
        id2ques[i+1] = ans2ques[string]
    return ans2id, id2ans, id2ques


# In[10]:

#top_n = 16140
top_n = 5000
ans2id, id2ans, id2ques = get_answers(train_data, top_n)
most_common_a = id2ans[1]
most_common_q = id2ques[1]


# In[13]:

def image_answers(data):
    image_files = {}
    for ques_id, image_path, text, opt, answer in data:
        #if exclude_missing and answer not in answers:
        if answer not in ans2id:
            ans2id[answer] = 0
            id2ans[0] = most_common_a
            id2ques[0] = most_common_q
        idx = ans2id[answer]
        image_paths = image_files.get(idx)
        if (image_paths == None):
            image_paths = []
        image_paths.append((image_path))
        image_files[idx] = image_paths
    return image_files


# In[14]:

def encode_answers(data, exclude_missing=False):
    encoded = []
    for ques_id, caption, text, opt_ans, answer in data:
        for a in opt_ans:
            if a not in ans2id:
                ans2id[a] = 0
                id2ans[0] = most_common_a
                id2ques[0] = most_common_q
        opt = [ans2id[a] for a in opt_ans]
        encoded.append((ques_id, caption, text, opt_ans, opt, answer, ans2id.get(answer, 0)))
    return encoded


# In[16]:

training_images = image_answers(train_data)
train_data = encode_answers(train_data)
eval_images = image_answers(eval_data)
eval_data = encode_answers(eval_data)


# In[18]:

n_train = 0
n_eval = 0
for answer in training_images:
    n_train = n_train + len(training_images[answer])
for answer in eval_images:
    n_eval = n_eval + len(eval_images[answer])

print(n_train, n_eval)


# ### Text Pre-processing

# In[19]:

X_tr = [q[2] for q in train_data]
X_te = [q[2] for q in eval_data]


# In[20]:

y_tr = [q[6] for q in train_data]
y_te = [q[6] for q in eval_data]


# In[21]:

len(X_tr), len(y_tr), len(X_te), len(y_te)


# In[22]:

X_te[0]


# ### Extract text features

# In[23]:

unigram_vect = CountVectorizer(ngram_range=(1, 1), lowercase=False, token_pattern=r'\b\w+\b', min_df=1)
unigram_analyze = unigram_vect.build_analyzer()
unigram_analyze(X_tr[0])


# In[24]:

def tags_st(tks):
    return ' '.join([w[1] for w in tks])


# In[25]:

class POSTags(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, x, y = None):
        tokens = [unigram_analyze(row) for row in x]
        self.tags = [nltk.pos_tag(row) for row in tokens]
        return self

    def transform(self, x):
        return [tags_st(row) for row in self.tags]


# In[26]:

tagger = POSTags()
tags = tagger.fit_transform(X_tr[0:10])


# In[27]:

X_tr[0:5]


# In[28]:

tags[0:5]


# ### Transformer for feature extraction
#     the sentence itself --> unigrams and bi-grams features
#     the sentence in lowercase --> unigrams and bi-grams features
#     the sentence pos-tags --> unigrams and bi-grams features

# In[29]:

combined_features_1 = FeatureUnion(
        transformer_list=[

            # Pipeline for pulling words features - lowercased
            ('lower_count', TfidfVectorizer(ngram_range=(1, 1),
                                                token_pattern=r'\b\w+\b', min_df=1, stop_words='english')),

            # Pipeline for pulling POS-Tag features
            ('tags', Pipeline([
                ('postag', POSTags()),
                ('pos_vect', TfidfVectorizer(ngram_range=(1, 1), lowercase=False,
                                                token_pattern=r'\b\w+\b', min_df=1, stop_words='english')),
            ])),

        ]
    )


# In[30]:

combined_features_2 = FeatureUnion(
        transformer_list=[
            # Pipeline for pulling words features - lowercased
            ('lower_count', TfidfVectorizer(ngram_range=(1, 2),
                                                token_pattern=r'\b\w+\b', min_df=1, stop_words='english')),

            # Pipeline for pulling POS-Tag features
            ('tags', Pipeline([
                ('postag', POSTags()),
                ('pos_vect', TfidfVectorizer(ngram_range=(1, 2), lowercase=False,
                                                token_pattern=r'\b\w+\b', min_df=1, stop_words='english')),
            ])),

        ]
    )


# In[30]:

X = list(X_tr + X_te)


# In[31]:

len(X_tr)


# In[32]:

X_ngrams = combined_features_1.fit_transform(X)


# In[33]:

X_ngrams.shape


# In[34]:

X_tr_ngrams = X_ngrams[0:248349]
X_te_ngrams = X_ngrams[248349:369861]


# In[35]:

X_tr_ngrams.shape, X_te_ngrams.shape


# In[36]:

a = X_tr_ngrams.getrow(0).todense()[0]
a = list(np.array(a).reshape(-1,))
np.unique(a)


# ### Training with Random Forest Classifier

# In[37]:

rfc = RandomForestClassifier(n_jobs=-1)


# In[ ]:

rfc.fit(X_tr_ngrams, y_tr)


# In[ ]:

start = 30000
rfc.fit(X_tr_ngrams[start:], y_tr[start:])


# In[46]:

X1 = X_te_ngrams[0:30000]
X2 = X_te_ngrams[30000:60000]
X3 = X_te_ngrams[60000:90000]
X4 = X_te_ngrams[90000:121512]


# In[47]:

y_eval_1 = rfc.predict(X1)


# In[48]:

y_eval_2 = rfc.predict(X2)


# In[49]:

y_eval_3 = rfc.predict(X3)


# In[50]:

y_eval_4 = rfc.predict(X4)


# In[51]:

y_eval = np.concatenate([y_eval_1, y_eval_2, y_eval_3, y_eval_4])


# In[52]:

pd.DataFrame(y_eval).to_csv("predictions.csv", header = False, sep="\t", index=False)


# In[53]:

r = []
i = 0
for ques_id, _, _2, _3, _4, _5, _6 in eval_data:
    a = id2ans[y_eval[i]]
    d = {'answer' : a, 'question_id': ques_id}
    r.append(d)
    i = i + 1


# In[54]:

with open('results_text.json', 'w') as f:
    json.dump(r, f)


# In[ ]:



