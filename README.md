# Visual Question Answering - VQA

### Description of files:
Assume that all the data were downloaded and put into the "Train" and "Val" folders.  
1. train_text.ipynb or train_text.py: Train and generate predictions for validation set using text features.  
2. train_images.ipynb or train_images.py: Train and generate predictions for validation set using image features.  
3. In the first case with text features, it generates file results_text.json  
4. In the second case with image features, during the process, it generates the prediction file predictions.csv.  
Then, you need to run the file predict_to_json.ipynb or predict_to_json.py to generate file results.json from prediction file.  
5. eval.ipynb or eval.py is the evaluation file provided by VQA organizers.
