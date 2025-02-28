# credit-risk-classification

## Overview of the Analysis

In this exercise, a dataset of historical lending activity from a peer-to-peer lending services company was used to build a model that can identify the creditworthiness of borrowers. To do this, a Logistic Regression machine learning model was built to predict loan risk based on the provided data. The ultimate goal is to test how well a Logistic Regression Model predicts healthy or high-risk loans.

The first step was to split the data into training and testing sets. The loan status data was defined as the label (y variable), and the remaining columns were defined as the features (x variable). Next the train_test_split function was used to perform the split, and the data was found to have 62028 training samples and 15508 testing samples.

Finally the LogisticRegression model was used from sklearn.linear_model to fit the model on our training data. Then the predictions were made using the predict(X_test) function. Finally, a confusion matrix was generated using the confusion_matrix(y_test, y_pred) fucntion, and subsequnetly a classification report was generated using the classification_report(y_test, y_pred) function.


## Results
The results of the model are summarized below:

Confusion Matrix:
[[14924    77]
 [   31   476]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     15001
           1       0.86      0.94      0.90       507

    accuracy                           0.99     15508
   macro avg       0.93      0.97      0.95     15508
weighted avg       0.99      0.99      0.99     15508


* Machine Learning Model 1: Logistic Regression
    * Model 1 Accuracy: 99% 
       * Healthy Loans: 
        * Precision: 1.0
        * Recall score: 0.99
        * F1 Score: 1.0

       * High-Risk Loans:  
        * Precision: 0.86
        * Recall score: 0.94
        * F1 Score: 0.90

    * Confusion Matrix Breakdown:
      * True Negatives: 14,924
      * False Positives: 77
      * False Negatives: 31
      * True Positives: 476


## Summary

From the result of the model, we can understand that overall our Logistic Regression Model performed well, with a high accuracy of 99% and a high recall rate of 94% for high risk loans. But there are some discrepancies in its ability to predict the `0` (healthy loans) vs `1` (high-risk loans). The model seems to predict healthy loans almost perfectly, with accurancy ranging from 99% - 100%, respectively. 

However, at 94%, the model's accuracy for predicting high-risk loans is slightly lower. And even though 31 high-risk loans were false negatives, we can conclude that though the model is not perfect for predicting high-risk loans, it is still viable as it still offers a relatively good degree of accuracy.

Therefore, the model performs well for both healthy and high-risk loans, but performs better for healthy loans overall. This is not ideal, because it would be more benficial for financial institutions to identify high-risk loans, and it may be beneficial to bridge this imbalance in the predictive capapbility for both healthy and high-risk loans. If more accuracy is absolutely desired, other models may be tested. 

But the current accuracy level of this model is acceptable and therefore recommended for use.