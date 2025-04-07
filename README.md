# Car Evaluation Using SMOTE

This project utilizes the Car Evaluation Dataset, originally developed by M. Bohanec and V. Rajkovic in 1997, to predict the acceptability of cars. The target attribute categorizes cars into four classes: unacceptable (`unacc`), acceptable (`acc`), good (`good`), and very good (`vgood`). 

The dataset contains categorical attributes such as buying price (`low`, `medium`, `high`, `very high`) and others, with no numerical or structural details about the cars. 

To address the class imbalance in the dataset, this project applies SMOTE (Synthetic Minority Oversampling Technique). SMOTE generates synthetic samples for the minority classes, ensuring a more balanced dataset and improving the performance of machine learning models.

For more details, refer to the `script.py` file, which contains the implementation.