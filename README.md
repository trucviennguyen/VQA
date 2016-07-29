# Visual Question Answering - VQA

In this project, I developed a machine learning model for Visual Question Answering. It is a combination of deep learning with text features. Given an image and a question related to this image, the system will automatically learn to generate an answer for this question. For example, if people give an image about Lugano lake, and ask a question like "Which season is that?", the system will analyze the image, the question and automatically generate the answer.

The system is written in Python and uses two kinds of features from images and the text of the question. It makes use of convolution neural network for image features and a simple "bag-of-words" model for text features.

It was presented at Data Science Retreat and Python meetup in Berlin, Germany, July 2016.

The problem is described at: http://visualqa.org/

### Description of files:

Assume that all the data were downloaded and put into the "Train" and "Val" folders.

1. train_text.ipynb or train_text.py: Train and generate predictions for validation set using text features.  
2. train_images.ipynb or train_images.py: Train and generate predictions for validation set using image features.  
3. In the first case with text features, it generates file results_text.json  
4. In the second case with image features, during the process, it generates the prediction file predictions.csv.  
Then, you need to run the file predict_to_json.ipynb or predict_to_json.py to generate file results.json from prediction file.  
5. eval.ipynb or eval.py is the evaluation file provided by VQA organizers.
