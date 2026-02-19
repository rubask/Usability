# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 05:02:40 2025

@author: Ruba
"""


Dataset shape: (924, 44)
Data prepared: (924, 41)
Creating winning feature combination...
Features engineered: 117 total (76 new features)

Training Accuracy: 1.000000 (100.0000%)
Test Accuracy: 0.983784 (98.3784%)

Cross-validation verification...

CV Mean: 0.971874 ± 0.017217
CV Range: 0.9405 - 0.9892
Classification Report:
--------------------------------------------------
                         precision    recall  f1-score   support
     Cluster 1 (Low IA)       1.00      1.00      1.00        38
Cluster 2 (Moderate IA)       0.98      0.99      0.98        95
    Cluster 3 (High IA)       0.98      0.96      0.97        52
               accuracy                           0.98       185
              macro avg       0.99      0.98      0.99       185
           weighted avg       0.98      0.98      0.98       185
 
 Confusion Matrix:
[[38  0  0]
 [ 0 94  1]
 [ 0  2 50]]


Accuracy: 0.983784 (98.3784%)
Model saved: champion_97_3_model.pkl
