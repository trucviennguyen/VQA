
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

import cPickle
import gzip

# preparing data for gpu
import theano
import theano.tensor as T


# ### Import Libraries for Deep Network

# In[ ]:

import sys
sys.path.append('src/')
import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, ReLU


# ### Loading VQA data

# In[4]:

def reading_vqa_data(vqa_dir, section):
    ans = 'mscoco_%s2014_annotations.json' % section
    with (vqa_dir / ans).open() as file_:
        ans_data = json.load(file_)
    image_by_id = {}
    answers_by_id = {}
    for answer in ans_data['annotations']:
        image = str(answer['image_id'])
        mca = answer['multiple_choice_answer']
        img = '0'*(12 - len(image)) + image
        s = '/data/%s/images' % section
        s = s + '/COCO_%s2014_' % section + img + '.jpg'
        image_by_id[answer['question_id']] = s
        answers_by_id[answer['question_id']] = mca
    filename = ('MultipleChoice_mscoco_'
                '%s2014_questions.json' % section)
    with (vqa_dir / filename).open() as file_:
        ques_data = json.load(file_)
    for question in ques_data['questions']:
        text = question['question']
        ques_id = question['question_id']
        options = question['multiple_choices']
        image_path = image_by_id[ques_id]
        image = Image.open(image_path)
        if min(image.size) < IMAGE_SIZE:
            image_path = prev_image
            image_by_id[ques_id] = image_path
        else:
            if (answers_by_id[ques_id] == 'yes'):
                prev_image = image_path
        yield ques_id, image_by_id[ques_id], text, options, answers_by_id[ques_id]


# In[5]:

IMAGE_SIZE = 64
image_thres = 200
train_data = list(reading_vqa_data(Path('/data/train'), 'train'))
eval_data = list(reading_vqa_data(Path('/data/val'), 'val'))


# In[6]:

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


# In[7]:

top_n = 16140
ans2id, id2ans, id2ques = get_answers(train_data, top_n)
most_common_a = id2ans[1]
most_common_q = id2ques[1]


# In[10]:

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
        #if (len(image_paths) < image_thres):
            #image_paths.append((image_path))
        image_files[idx] = image_paths
    return image_files


# In[11]:

def encode_answers(data, exclude_missing=False):
    encoded = []
    for ques_id, image, text, opt_ans, answer in data:
        for a in opt_ans:
            if a not in ans2id:
                ans2id[a] = 0
                id2ans[0] = most_common_a
                id2ques[0] = most_common_q
        opt = [ans2id[a] for a in opt_ans]
        encoded.append((ques_id, image, text, opt_ans, opt, answer, ans2id.get(answer, 0)))
    return encoded


# In[13]:

training_images = image_answers(train_data)
train_data = encode_answers(train_data)
eval_images = image_answers(eval_data)
eval_data = encode_answers(eval_data)


# In[ ]:

n_train = 0
n_eval = 0
for answer in training_images:
    n_train = n_train + len(training_images[answer])
for answer in eval_images:
    n_eval = n_eval + len(eval_images[answer])

print(n_train, n_eval)


# ### Image Pre-processing

# In[21]:

def prepare_image(image, true_size, grayscale=True):
    factor = min(image.size) / float(true_size)
    #if min(image.size) < true_size:
    #    raise ValueError('image to small, is {}, '
    #                     'needs to be min: {}'.format(min(image.size), true_size))
    image.thumbnail(np.ceil(np.array(image.size) / factor))
    image_narray = np.array(image)
    if grayscale:
        if image_narray.ndim == 3:
            image_narray = image_narray.mean(axis=2)
        width, height = image_narray.shape
    else:
        if image_narray.ndim == 2:
            image_narray = np.dstack([image_narray, image_narray, image_narray])
        width, height, _ = image_narray.shape
    offset_width = (width - true_size) / 2
    offset_height = (height - true_size) / 2
    image_narray = image_narray[offset_width:offset_width+true_size, offset_height:offset_height+true_size]
    if grayscale:
        if image_narray.shape != (true_size, true_size):
            raise ValueError('Preparing failed. '
                             'Image should be {}, but is'
                             '{}'.format(image_narray.shape,[true_size, true_size, 3] ))
    else:
        if image_narray.shape != (true_size, true_size, 3):
            raise ValueError('Preparing failed. '
                             'Image should be {}, but is'
                             '{}'.format(image_narray.shape,[true_size, true_size, 3] ))

    return image_narray


# In[22]:

print(len(training_images), len(eval_images))


# In[23]:

print(len(train_data), len(eval_data))


# In[24]:

print(len(training_images), len(eval_images))


# In[25]:

# load and process tofeatures

IMAGE_SIZE = 64
dim_features = IMAGE_SIZE**2
n_train_features = n_train
train_features = np.ones([n_train_features, dim_features]) * np.nan
train_labels = np.ones([n_train_features]) * np.nan

n_test_features = n_eval
test_features = np.ones([n_test_features, dim_features]) * np.nan
test_labels = np.ones([n_test_features]) * np.nan


# In[26]:

index = 0
for answer_index, answer in enumerate(training_images):
    image_paths = training_images[answer]
    for feature_file in image_paths:
        image = Image.open(feature_file)
        feature_data = prepare_image(image, IMAGE_SIZE).reshape(dim_features)    
        train_features[index] = feature_data
        train_labels[index] = answer
        index = index + 1


# In[27]:

index = 0
for answer_index, answer in enumerate(eval_images):
    image_paths = eval_images[answer]
    for feature_file in image_paths:
        image = Image.open(feature_file)
        feature_data = prepare_image(image, IMAGE_SIZE).reshape(dim_features)     
        test_features[index] = feature_data
        test_labels[index] = answer
        index = index + 1


# In[28]:

def check_array(array_to_check):
    if np.any(np.isnan(array_to_check)):
        raise ValueError('Nan found')

check_array(train_features)
check_array(train_labels)
check_array(test_features)
check_array(test_labels)
train_labels = np.array(train_labels, dtype=np.int)
test_labels = np.array(test_labels, dtype=np.int)


# In[33]:

# normalizing features

def norm_features(train_features, test_features):
    max_array = train_features.max(0)
    train_features_0_1 =train_features / max_array
    test_features_0_1 = test_features / max_array
    mean_array = train_features_0_1.mean(0)
    train_features_normed = train_features_0_1 - mean_array
    test_features_normed = test_features_0_1 - mean_array
    return train_features_normed, test_features_normed
train_features_normed, test_features_normed = norm_features(train_features, test_features)

print(train_features_normed.min(), train_features_normed.max(), train_features_normed.mean())
print(test_features_normed.min(), test_features_normed.max(), test_features_normed.mean())


# In[34]:

train_features_normed.shape, len(train_labels), test_features_normed.shape, len(test_labels)


# In[35]:

def make_shared_GPU(data):
    """Place the data into shared variables.  This allows Theano to copy
    the data to the GPU, if one is available.

    """
    shared_x = theano.shared(
        np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(
        np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
    return shared_x, T.cast(shared_y, "int32")
    #return shared_x, T.cast(shared_y, "int64")
training_data = make_shared_GPU([train_features_normed, train_labels])
validation_data = make_shared_GPU([test_features_normed, test_labels])
test_data = validation_data


# In[36]:

train_features_normed.shape, len(train_labels), test_features_normed.shape, len(test_labels)


# ### Learn neural network

# In[ ]:

## adding a conv layer
#THEANO_FLAGS="exception_verbosity=high"
mini_batch_size = 83
net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 64, 64), 
                      filter_shape=(20, 1, 5, 5), 
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        FullyConnectedLayer(n_in=20*30*30, n_out=100, activation_fn=ReLU),
        SoftmaxLayer(n_in=100, n_out=16141)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, 
            validation_data, test_data)

