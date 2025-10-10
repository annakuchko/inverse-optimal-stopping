Final implementation of inverse optimal stopping algorithm via IQ-Learning. The following modifications are included in the implementation:
1) IQ-Learning with binary actions
2) IQ-Learning with SMOTE oversampling
3) IQ-Learning with Confidence-Score based SMOTE
4) Model-based IQ-Learning with dynamics approximation
5) Model-based IQ-Learning with SMOTE oversampling
6) Model-based IQ-Learning with Confidence-Score based SMOTE
7) DO-IQS
8) DO-IQS-LB
9) Classifier
10) Classifier-SMOTE

The implementation also includes the following environments:
1) Car environment (basic example for optimal stopping of a car)
2) 2D brownian motion dataset
3) Bessel2 dataset
4) STAR and RADIAL environment
5) Changepoint detection examples


All the models are developed to be trained offline, but can be extended to online scenario (including incorporation of Actr-Critic etc) for improved performance in situations where online data might become available. The focuse of the current work is to implement algorithms that will show a good performance in application where online interactions with the enironment might not be available due to safety concernes (which is common for problems where optimal stopping is involved). The above frameworks can also be used for offline-pre-training of the online-models to increase the performance and training outcomes when running them online.

run.py is the main script aggregating different models training and testing. 

