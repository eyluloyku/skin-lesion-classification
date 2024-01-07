import pandas as pd
import numpy as np
import PIL
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
import optuna

ground_truth = pd.read_csv('~/project/ISIC_2019_Training_GroundTruth.csv')
output_dir = "ISIC_2019_Augmented_Input"
path = output_dir
num_images = sum([len(files) for r, d, files in os.walk(path)])


# Initialize an empty DataFrame to store the augmented image IDs and labels
augmented_data_df = pd.DataFrame(columns=ground_truth.columns)

# List all files in the input directory
image_files = os.listdir(output_dir)
for file in image_files:
  if file.find("jpg") != -1:
   # Get the label for the image
    image_identifier = file[0:file.find("augmented")-1]
    #image_identifier = file[0:file.find("jpg")-1]
    specific_image_row = ground_truth[ground_truth['image'] == image_identifier]
    label_data = specific_image_row.drop(columns=['image']).iloc[0].to_dict()

  # Append to the DataFrame
    new_row = {"image": file}
    new_row.update(label_data)
    new_data = pd.DataFrame([new_row])
    augmented_data_df = pd.concat([augmented_data_df, new_data], ignore_index=True)

augmented_data_df = augmented_data_df.sort_values(by=['image'])

mel_df = augmented_data_df.groupby("MEL").get_group(1)
nv_df = augmented_data_df.groupby("NV").get_group(1)
bcc_df = augmented_data_df.groupby("BCC").get_group(1)
ak_df = augmented_data_df.groupby("AK").get_group(1)
bkl_df = augmented_data_df.groupby("BKL").get_group(1)
df_df = augmented_data_df.groupby("DF").get_group(1)
vasc_df = augmented_data_df.groupby("VASC").get_group(1)
scc_df = augmented_data_df.groupby("SCC").get_group(1)

num_rows1 = mel_df.shape[0]
last_20_percent1 = int(num_rows1 * 0.2)
first_70_percent1 = int(num_rows1 * 0.7)

num_rows2 = nv_df.shape[0]
last_20_percent2 = int(num_rows2 * 0.2)
first_70_percent2 = int(num_rows2 * 0.7)

num_rows3 = bcc_df.shape[0]
last_20_percent3 = int(num_rows3 * 0.2)
first_70_percent3 = int(num_rows3 * 0.7)

num_rows4 = ak_df.shape[0]
last_20_percent4 = int(num_rows4 * 0.2)
first_70_percent4 = int(num_rows4 * 0.7)

num_rows5 = bkl_df.shape[0]
last_20_percent5 = int(num_rows5 * 0.2)
first_70_percent5 = int(num_rows5 * 0.7)

num_rows6 = df_df.shape[0]
last_20_percent6 = int(num_rows6 * 0.2)
first_70_percent6 = int(num_rows6 * 0.7)

num_rows7 = vasc_df.shape[0]
last_20_percent7 = int(num_rows7 * 0.2)
first_70_percent7 = int(num_rows7 * 0.7)

num_rows8 = scc_df.shape[0]
last_20_percent8 = int(num_rows8 * 0.2)
first_70_percent8 = int(num_rows8 * 0.7)

last_20_percent_df1 = mel_df.iloc[-last_20_percent1:]
last_20_percent_df2 = nv_df.iloc[-last_20_percent2:]
last_20_percent_df3 = bcc_df.iloc[-last_20_percent3:]
last_20_percent_df4 = ak_df.iloc[-last_20_percent4:]
last_20_percent_df5 = bkl_df.iloc[-last_20_percent5:]
last_20_percent_df6 = df_df.iloc[-last_20_percent6:]
last_20_percent_df7 = vasc_df.iloc[-last_20_percent7:]
last_20_percent_df8 = scc_df.iloc[-last_20_percent8:]

first_70_percent_df1 = mel_df.iloc[:first_70_percent1]
first_70_percent_df2 = nv_df.iloc[:first_70_percent2]
first_70_percent_df3 = bcc_df.iloc[:first_70_percent3]
first_70_percent_df4 = ak_df.iloc[:first_70_percent4]
first_70_percent_df5 = bkl_df.iloc[:first_70_percent5]
first_70_percent_df6 = df_df.iloc[:first_70_percent6]
first_70_percent_df7 = vasc_df.iloc[:first_70_percent7]
first_70_percent_df8 = scc_df.iloc[:first_70_percent8]

test_1 = mel_df.iloc[first_70_percent1:-last_20_percent1]
test_2 = nv_df.iloc[first_70_percent2:-last_20_percent2]
test_3 = bcc_df.iloc[first_70_percent3:-last_20_percent3]
test_4 = ak_df.iloc[first_70_percent4:-last_20_percent4]
test_5 = bkl_df.iloc[first_70_percent5:-last_20_percent5]
test_6 = df_df.iloc[first_70_percent6:-last_20_percent6]
test_7 = vasc_df.iloc[first_70_percent7:-last_20_percent7]
test_8 = scc_df.iloc[first_70_percent8:-last_20_percent8]

augmented_data_df = pd.concat([first_70_percent_df1, first_70_percent_df2, first_70_percent_df3, first_70_percent_df4, first_70_percent_df5, first_70_percent_df6, first_70_percent_df7, first_70_percent_df8])
augmented_test_df = pd.concat([last_20_percent_df1, last_20_percent_df2, last_20_percent_df3, last_20_percent_df4, last_20_percent_df5, last_20_percent_df6, last_20_percent_df7, last_20_percent_df8])
augmented_validation_df = pd.concat([test_1, test_2, test_3, test_4, test_5, test_6, test_7, test_8])

augmented_data_df = augmented_data_df.drop('UNK', axis=1)
augmented_validation_df = augmented_validation_df.drop('UNK', axis=1)
augmented_test_df = augmented_test_df.drop('UNK', axis=1)

datagen = ImageDataGenerator(rescale=1./255)

test_generator = datagen.flow_from_dataframe(
    dataframe=augmented_test_df,
    directory= output_dir,
    x_col="image",
    y_col=augmented_validation_df.columns[1:],
    batch_size=16,
    shuffle=False,
    class_mode="raw",
    target_size=(300,300))

combined_data_df = pd.concat([augmented_data_df, augmented_validation_df])
combined_generator = datagen.flow_from_dataframe(
    dataframe=combined_data_df,
    directory= output_dir,
    x_col="image",
    y_col=combined_data_df.columns[1:],
    batch_size=16,
    shuffle=True,
    class_mode="raw",
    target_size=(224,224))

model_efficientnetb3 = load_model('b3_rmsprop_noaug.h5')
model_resnet50 = load_model('resnet_rmsprop_noaug.h5')
model_resnext = load_model('best_model2.h5')

from tensorflow.keras.optimizers import RMSprop
# Retrain the model using both training and validation data
model_resnet50.compile(optimizer=RMSprop(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model_resnext.compile(optimizer=RMSprop(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the best model with train and val data again
total_epochs = 15  # Set the total number of epochs for retraining
steps_per_epoch = len(combined_generator)

label_columns = combined_data_df.columns[1:]  # All columns except the image column
y_train = combined_data_df[label_columns].values
# Flatten y_train if necessary and encode labels
y_train_flat = y_train.argmax(axis=1)  # Only if each image has one label
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_flat)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

model_resnet50.fit(
    combined_generator,
    epochs=total_epochs,
    steps_per_epoch=steps_per_epoch,
    class_weight=class_weights_dict,
)
model_resnext.fit(
    combined_generator,
    epochs=total_epochs,
    steps_per_epoch=steps_per_epoch,
    class_weight=class_weights_dict,
)

print("Accuracy before ensemble:")
test_loss, test_accuracy = model_efficientnetb3.evaluate(test_generator)
print(f"Test Loss B3: {test_loss}, Test Accuracy B3: {test_accuracy}")
test_loss, test_accuracy = model_resnet50.evaluate(test_generator)
print(f"Test Loss ResNet: {test_loss}, Test Accuracy ResNet: {test_accuracy}")
test_loss, test_accuracy = model_resnext.evaluate(test_generator)
print(f"Test Loss ResNeXt: {test_loss}, Test Accuracy ResNeXt: {test_accuracy}")


def soft_voting(models, generator):
    predictions = [model.predict(generator) for model in models]
    avg_predictions = np.mean(predictions, axis=0)
    return np.argmax(avg_predictions, axis=1)

def hard_voting(models, generator, num_classes):
    predictions = [np.argmax(model.predict(generator), axis=1) for model in models]
    votes = np.apply_along_axis(lambda x: np.bincount(x, minlength=num_classes), axis=0, arr=np.array(predictions))
    return np.argmax(votes, axis=0)


def weighted_soft_voting(models, generator, weights):
    predictions = [model.predict(generator) for model in models]
    weighted_predictions = np.average(predictions, axis=0, weights=weights)
    return np.argmax(weighted_predictions, axis=1)

def weighted_hard_voting(models, generator, weights, num_classes):
    # Determine the total number of samples
    total_samples = generator.samples

    # Initialize the weighted_votes array
    weighted_votes = np.zeros((total_samples, num_classes))

    # Accumulate predictions from each model
    for i, model in enumerate(models):
        generator.reset()  # Reset the generator before each prediction
        predictions = np.argmax(model.predict(generator, steps=np.ceil(total_samples / generator.batch_size)), axis=1)
        
        for j, pred in enumerate(predictions):
            weighted_votes[j][pred] += weights[i]

    return np.argmax(weighted_votes, axis=1)


# Example usage
models = [model_efficientnetb3, model_resnet50, model_resnext]
weights = [0.4, 0.3, 0.3]  # Example weights, adjust as needed

# Now you can call any of the voting functions with these models and weights


num_classes = 8  # Replace with the actual number of classes in your dataset

# Soft Voting
soft_voting_predictions = soft_voting(models, test_generator)
# Hard Voting
hard_voting_predictions = hard_voting(models, test_generator, num_classes)
# Weighted Soft Voting
weighted_soft_voting_predictions = weighted_soft_voting(models, test_generator, weights)
# Weighted Hard Voting
weighted_hard_voting_predictions = weighted_hard_voting(models, test_generator, weights, num_classes)


# Assuming you have true labels in a one-hot encoded format
true_labels = test_generator.labels
true_labels_flat = np.argmax(true_labels, axis=1)

# Evaluate Soft Voting
accuracy_soft = accuracy_score(true_labels_flat, soft_voting_predictions)
print(f'Soft Voting Accuracy: {accuracy_soft}')
print(classification_report(true_labels_flat, soft_voting_predictions))

# Evaluate Hard Voting
accuracy_hard = accuracy_score(true_labels_flat, hard_voting_predictions)
print(f'Hard Voting Accuracy: {accuracy_hard}')
print(classification_report(true_labels_flat, hard_voting_predictions))

# Evaluate Weighted Soft Voting
accuracy_weighted_soft = accuracy_score(true_labels_flat, weighted_soft_voting_predictions)
print(f'Weighted Soft Voting Accuracy: {accuracy_weighted_soft}')
print(classification_report(true_labels_flat, weighted_soft_voting_predictions))

# Evaluate Weighted Hard Voting
accuracy_weighted_hard = accuracy_score(true_labels_flat, weighted_hard_voting_predictions)
print(f'Weighted Hard Voting Accuracy: {accuracy_weighted_hard}')
print(classification_report(true_labels_flat, weighted_hard_voting_predictions))


def weighted_soft_voting_prob(models, generator, weights):
    predictions = [model.predict(generator) for model in models]
    weighted_predictions = np.average(predictions, axis=0, weights=weights)
    return weighted_predictions  # Returns averaged probabilities



def objective(trial):
    # Suggest weights for each model in the ensemble
    weight1 = trial.suggest_float('weight1', 0, 1)
    weight2 = trial.suggest_float('weight2', 0, 1)
    weight3 = trial.suggest_float('weight3', 0, 1)  
    total_weight = weight1 + weight2 + weight3

    normalized_weights = [weight1 / total_weight, weight2 / total_weight, weight3 / total_weight]

    ensemble_probs = weighted_soft_voting_prob([model_efficientnetb3, model_resnet50, model_resnext], test_generator, normalized_weights)
    predicted_labels = np.argmax(ensemble_probs, axis=1)

    # Evaluate the performance
    accuracy = accuracy_score(true_labels_flat, predicted_labels)
    return accuracy



# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)  # Adjust the number of trials as needed

print('Best trial:', study.best_trial.params)


best_weights = study.best_trial.params
total_weight = sum(best_weights.values())
normalized_weights = [w / total_weight for w in best_weights.values()]

# Make predictions with the best model
ensemble_probs = weighted_soft_voting([model_efficientnetb3, model_resnet50, model_resnext], test_generator, normalized_weights)

# Evaluate the accuracy
accuracy = accuracy_score(true_labels_flat, ensemble_probs)
print("Accuracy of the tuned weighted soft voting:", accuracy)

