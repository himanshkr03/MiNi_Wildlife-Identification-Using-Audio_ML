# MiNi_Wildlife-Identification-Using-Audio_ML
### Audio Classification of Dog Barks, Cat Meows, and Bird Chirps

## Project Overview

This mini-project aims to classify audio recordings into three categories: dog barks, cat meows, and bird chirps. It leverages deep learning techniques, specifically convolutional neural networks (CNNs), to extract features from audio spectrograms and perform classification.

## Dataset

The project utilizes the UrbanSound8K dataset, a collection of urban sound recordings.  However, for this specific project, we focus on a subset of the dataset containing only audio files related to the three selected classes: dog barks, cat meows, and bird chirps.

- **Dataset Source:** https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz
- **Selected Classes:** 'dog_bark', 'cat_meow', 'chirping_birds'

## Libraries

The following Python libraries are used in this project:

- **librosa:** For audio loading, feature extraction (mel spectrograms), and audio processing.
- **audiomentations:** For potential data augmentation (not used in this version).
- **matplotlib:** For visualizing spectrograms.
- **torch:** For deep learning model building and training.
- **torchvision:** For dataset and data loading utilities.
- **seaborn:** For creating visually appealing confusion matrices.
- **wget:** For downloading the dataset.
- **numpy:** For numerical computations.
- **pandas:** For data manipulation and loading the metadata.
- **sklearn:** For data splitting, model evaluation (classification report, confusion matrix).
- **tqdm:** For displaying progress bars during training (not used in this version).

## Methods

1. **Data Preprocessing:**
    - Audio files are loaded using `librosa.load`.
    - Mel spectrograms are generated using `librosa.feature.melspectrogram`.
    - Spectrograms are converted to decibels using `librosa.power_to_db`.
    - Spectrograms are padded/truncated to a fixed length for consistency.
    - Spectrograms are normalized using z-score normalization.

2. **Model Building:**
    - A CNN model is implemented using `torch.nn` modules.
    - The model consists of convolutional layers, max pooling, and fully connected layers.
    - The architecture is designed to learn relevant features from the spectrograms.

3. **Training:**
    - The model is trained using the Adam optimizer and cross-entropy loss.
    - Data is split into training and testing sets.
    - Training is performed for a fixed number of epochs.
    - Performance metrics (loss, accuracy, precision, recall, F1-score) are tracked during training.

4. **Evaluation:**
    - The trained model is evaluated on the testing set.
    - A classification report and confusion matrix are generated to assess the model's performance.

## Steps to Run the Project

1. **Install Libraries:** Use the following command to install the required libraries
2. **Download Dataset:** Download and extract the UrbanSound8K dataset.
3. **Run the Python Code:** Execute the provided Python code, which performs the following steps:
    - Loads the dataset and metadata.
    - Filters the dataset for the selected classes.
    - Preprocesses the audio data.
    - Splits the data into training and testing sets.
    - Defines and trains the CNN model.
    - Evaluates the model and prints the results.
    - Saves the trained model.

## Analysis Report

- **Model Performance:** The model achieves [insert accuracy, precision, recall, F1-score values here] on the testing set.
- **Confusion Matrix:** The confusion matrix shows the distribution of predictions across the different classes, revealing any potential areas of confusion for the model.
- **Observations:** [Include any insights or observations about the model's performance, potential improvements, etc.]

## Potential Improvements

- **Data Augmentation:** Explore using audiomentations to augment the training data and improve generalization.
- **Hyperparameter Tuning:** Experiment with different hyperparameters (learning rate, batch size, number of layers, etc.) to optimize model performance.
- **Advanced Architectures:** Consider using more sophisticated CNN architectures or other deep learning models for audio classification.
- **Larger Dataset:** Train the model on a larger and more diverse dataset to further improve its accuracy and robustness.

## Conclusion

This mini-project demonstrates the application of deep learning for audio classification using the UrbanSound8K dataset. The results show the potential of CNNs in extracting relevant features from audio spectrograms and achieving reasonable classification accuracy. Further improvements can be made through data augmentation, hyperparameter tuning, and exploring more advanced models and architectures.


## 👋 HellO There! Let's Dive Into the World of Ideas 🚀

Hey, folks! I'm **Himanshu Rajak**, your friendly neighborhood tech enthusiast. When I'm not busy solving DSA problems or training models that make computers *a tad bit smarter*, you’ll find me diving deep into the realms of **Data Science**, **Machine Learning**, and **Artificial Intelligence**.  

Here’s the fun part: I’m totally obsessed with exploring **Large Language Models (LLMs)**, **Generative AI** (yes, those mind-blowing AI that can create art, text, and maybe even jokes one day 🤖), and **Quantum Computing** (because who doesn’t love qubits doing magical things?).  

But wait, there's more! I’m also super passionate about publishing research papers and sharing my nerdy findings with the world. If you’re a fellow explorer or just someone who loves discussing tech, memes, or AI breakthroughs, let’s connect!

- **LinkedIn**: [Himanshu Rajak](https://www.linkedin.com/in/himanshu-rajak-22b98221b/) (Professional vibes only 😉)
- **Medium**: [Himanshu Rajak](https://himanshusurendrarajak.medium.com/) (Where I pen my thoughts and experiments 🖋️)

Let’s team up and create something epic. Whether it’s about **generative algorithms** or **quantum wizardry**, I’m all ears—and ideas!  
🎯 Ping me, let’s innovate, and maybe grab some virtual coffee. ☕✨
