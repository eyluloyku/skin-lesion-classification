# skin-lesion-classification

This project has been implemented as a graduation project. For more information about techniques and approaches please refer to FinalReport.pdf. Or reach out to me for further questions!  

## Dataset
https://challenge.isic-archive.com/data/#2019

## Challenge
This project, Skin Lesion Classification with Computer Vision, addresses the critical issue of diagnosing cancerous skin lesions, a growing concern in dermatology. Each year, thousands of people are diagnosed with skin cancer, and early detection is pivotal for successful treatment. However, due to the complexity and variability of skin lesions, skin lesion classification poses significant challenges. Our project aimed to develop a computer vision based deep learning algorithm to accurately classify skin lesions.

## File Descriptions
- training.py: Baseline training model for ResNeXt50 archtitecture. Model parameters, resolutions and hyperparameters are to be fine-tuned if other models used.
- preprocessing.py: Preprocessing methods such as hair removal, black corner removal and shades of gray are applied.
  
| Original   | Preprocessed |
|---------------|-------------|
| <img width="224" alt="image" src="https://github.com/eyluloyku/skin-lesion-classification/assets/116841987/9e4baf0a-7235-431b-be14-aa6f986b6d19"> | <img width="218" alt="image" src="https://github.com/eyluloyku/skin-lesion-classification/assets/116841987/0f086ad6-f63b-4e98-967e-e13c03374cbd"> |


- augmentation.py: Excessive data augmentation to combat data imbalance in the class by equalizing the number of images in each class.

| Original   | Augmented 1 | Augmented 2 | Augmented 3 |
|---------------|-------------| -------------| -------------|
| <img width="145" alt="image" src="https://github.com/eyluloyku/skin-lesion-classification/assets/116841987/d4791dc9-c10a-4cca-a61f-55f6d3791f8c">| <img width="141" alt="image" src="https://github.com/eyluloyku/skin-lesion-classification/assets/116841987/530791e3-684d-4bd9-8b3c-e1c365035641">| <img width="141" alt="image" src="https://github.com/eyluloyku/skin-lesion-classification/assets/116841987/76aceba5-cdcd-4dff-9af9-00dea31e8467"> | <img width="140" alt="image" src="https://github.com/eyluloyku/skin-lesion-classification/assets/116841987/7af97460-1816-4a1b-a6e4-36541023d6ee"> |

- tta.py: Test time augmentation by combining different augmentation techniques for each image in test set.
- ensemble.py: Soft voting, hard voting, weighted soft voting and weighted hard voting approaches applied. Fine tuned using Optuna.

## Results

Weighted Soft Voting Accuracy: 0.738

| Class                                  | Precision | Recall | F1-Score | Support |
|----------------------------------------|-----------|--------|----------|---------|
| Melanoma (MEL)                         | 0.63      | 0.72   | 0.67     | 904     |
| Melanocytic Nevus (NV)                 | 0.85      | 0.80   | 0.82     | 2575    |
| Basal Cell Carcinoma (BCC)             | 0.64      | 0.89   | 0.75     | 664     |
| Actinic Keratosis (AK)                 | 0.52      | 0.67   | 0.58     | 173     |
| Benign Keratosis (BKL)                 | 0.71      | 0.40   | 0.51     | 524     |
| Dermatofibroma (DF)                    | 0.65      | 0.47   | 0.54     | 47      |
| Vascular Lesion (VASC)                 | 0.89      | 0.48   | 0.62     | 50      |
| Squamous Cell Carcinoma (SCC)          | 0.76      | 0.46   | 0.57     | 125     |
| **Macro Average**                      | **0.70**  | **0.61** | **0.63** | **5062** |
| **Weighted Average**                   | **0.75**  | **0.74** | **0.73** | **5062**


## Future Work
The study shows that while several approaches improve model performance, careful consideration must be given to class imbalances and the unique characteristics of each class to achieve a more uniform performance across the board. This not only points to the need for research in deep learning for skin lesion classification but also the requirement of collaboration with external entities that can collect and provide diverse and more comprehensive datasets.
