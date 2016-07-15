
# coding: utf-8

# ### Import Libraries for General Processing

# In[1]:

import numpy as np
import pandas as pd

from collections import Counter
import ujson as json
from pathlib import Path
import random

import warnings
warnings.filterwarnings("ignore")


# ### Loading VQA data

# In[2]:

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
        yield ques_id, image_by_id[ques_id], text, options, answers_by_id[ques_id]


# In[3]:

train_data = list(reading_vqa_data(Path('Train'), 'train'))
eval_data = list(reading_vqa_data(Path('Val'), 'val'))


# In[4]:

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


# In[5]:

top_n = 16140
ans2id, id2ans, id2ques = get_answers(train_data, top_n)
id2ans[0] = 'yes'
most_common_a = id2ans[1]
most_common_q = id2ques[1]


# ### From predictions to json

# In[9]:

y_eval = np.asarray(pd.read_csv("predictions.csv", header = None, sep = "\t")[0])


# In[12]:

r = []
i = 0
for ques_id, _, _2, _3, _4 in eval_data:
    a = id2ans[y_eval[i]]
    d = {'answer' : a, 'question_id': ques_id}
    r.append(d)
    i = i + 1


# In[13]:

with open('results.json', 'w') as f:
    json.dump(r, f)

