# Cat or Dog Classification Model

## Dataset

The dataset can be found on Kaggle. You can download it using the following link:

[Cat or Dog Classification Dataset](https://drive.google.com/drive/folders/1gWctRWtAzVqZQ4zxVk3LMU3X-ZNk8meE?usp=sharing)

## Data Splitting

The dataset was split into training and testing sets with an 80-20 ratio. The splitting was performed using a Python script, and the details can be found in the `data_preparation.ipynb` file in this repository.

## Data Preprocessing

Data preprocessing was done using the `ImageDataGenerator` class from Keras. Augmentation techniques such as rotation, width shift, height shift, shear, and zoom were applied to the training set to enhance model generalization.

## Model Architecture

The base of the model was a pre-trained VGG16 architecture. Additional layers were added to fine-tune the model for the cat or dog classification task. The complete architecture can be found in the `cat_dog_classifier.py` file.

## Training

The model was trained using the preprocessed training data for 10 epochs. The training script is available in the `train_model.py` file.

## Model Evaluation

After training, the model achieved an accuracy of 91% on the test set. Evaluation metrics such as accuracy, precision, recall, and F1-score were computed and can be found in the `evaluate_model.ipynb` Jupyter Notebook.

## Saved Model

The trained model has been saved and can be downloaded using the following link:

[Saved Cat or Dog Classification Model](https://drive.google.com/file/d/1XL6eKbiflzc3lSZ-a2FALpjKvgg6R7Bo/view?usp=sharing)

## Dependencies

Make sure to install the required dependencies listed in the `requirements.txt` file before running the scripts.

```bash
pip install -r requirements.txt
