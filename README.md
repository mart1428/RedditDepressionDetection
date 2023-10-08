# RedditDepressionDetection

## **Data**
The data was collected from Kaggle. Link: https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned
The data has been cleaned and it has 7650 instances, with ~50% split for depression and non-depression.

## **Context**
The motivation behind this project is to create a Deep Learning model that has the ability to detect Depression based on subreddits. Mental health has been a great problem in today's society and it is important to detect and help people with mental illness.

## **Model**
Model uses LSTM Method with Dropout and stacked layers. It then uses 3 fully connected layers as classifiers.

Training Hyperparameters:
* Learning Rate = 0.001
* batch_size = 128
* Weight_decay = 0.001
* Dropout Probability= 0.4

Final model:
* Test Accuracy reached ~72%.
* Test Loss: 0.571, Test Error: 0.014, Test Acc: 71.57%, Precision: 100%, Recall: 42.71%
* Train Loss: 0.252, Train Error: 0.072, Train Acc: 91.57%
* Val Loss: 0.683, Val Error: 0.028, Val Acc: 72.45%

## **Future Research**
Further considerations to be made in the future:
* Reduce overfitting 
* Use transfer learning or transformers to train model.
* More data points to differentiate depression and not depression.
* Better data: Current clean data might have confused GloVe embeddings and model when training.
