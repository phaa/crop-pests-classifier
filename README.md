# Project: Pest detection in cotton crops using Transfer Learning

## Overview

This project utilizes convolutional neural networks (CNNs) to detect different pests in cotton crops. The primary goal is to build an effective model that can identify two specific pest species: Weevil and Whitefly, which significantly impact cotton production in Brazil.

The project explores both building a CNN from scratch and leveraging transfer learning with a pre-trained VGG16 model to compare performance and demonstrate the advantages of using pre-trained networks for image classification tasks with limited datasets.

## Objectives

- Analyze the distribution of instances in the dataset (training and validation).
- Pre-process and divide the dataset correctly into training, validation, and test sets.
- Apply normalization to the images.
- Apply data augmentation to expand the training set and prevent overfitting.
- Use transfer learning with a pre-trained VGG16 model to extract robust features.
- Test and validate the performance of the developed models.

## Dataset

The dataset contains images of two pest species found in cotton crops:

| Species   | Description                                                                                                | Image                                                                                                |
| --------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Weevil    | Insect with fine jaws that pierce the flower buds and bolls of cotton plants (Anthonomus grandis)          | <img src="/datasets/weevil/bicudo01.png" alt="Weevil Image" width="200" height="200">                  |
| Whitefly  | A sucking insect that steals nutrients from the plant it is on (*Bemisia tabaci*)                          | <img src="/datasets/whitefly/MoscaBranca01.png" alt="Whitefly Image" width="200" height="200">               |

The dataset is structured into directories for each class: `datasets/weevil` and `datasets/whitefly`.

## Project Structure

- **Notebook:** The analysis and model development are performed within a Google Colab notebook (`dio_transfer_learning.ipynb`).
- **Data:** The dataset is expected to be in a directory named `datasets/`.
- **Models:** Trained models, specifically the best performing one from scratch, are saved in a directory named `models/`.

## Methodology

1.  **Data Loading and Exploration:** The images from the dataset are loaded, and the class distribution is analyzed to ensure balance. Sample images are visualized.
2.  **Data Preprocessing:** Images are resized, converted to arrays, and normalized. The dataset is split into training, validation, and test sets. Labels are converted to one-hot encoding.
3.  **Model Development (from Scratch):** A sequential CNN model is built from scratch with multiple convolutional, pooling, batch normalization, and dropout layers.
4.  **Model Training (from Scratch):** The custom CNN model is compiled with categorical cross-entropy loss and AdamW optimizer. The model is trained, and training history (accuracy and loss) is plotted. Callbacks for model checkpointing and learning rate reduction are used.
5.  **Model Evaluation (from Scratch):** The trained model is evaluated on the test set, and a classification report and confusion matrix are generated to assess performance.
6.  **Model Development (Transfer Learning):** A pre-trained VGG16 model is loaded with ImageNet weights, excluding the top classification layer. A new dense layer with the number of classes is added. The pre-trained layers are frozen to leverage their learned features.
7.  **Model Training (Transfer Learning):** The transfer learning model is compiled with categorical cross-entropy loss and Adam optimizer. The model is trained with the unfrozen layers.
8.  **Model Evaluation (Transfer Learning):** The transfer learning model is evaluated on the test set, and a classification report and confusion matrix are generated for comparison.
9.  **Sample Image Testing:** Both models are tested on sample images to visualize predictions.

## Dependencies

The project requires the following libraries:

- `numpy`
- `scipy`
- `pandas`
- `seaborn`
- `matplotlib`
- `tensorflow` (with CUDA support for GPU acceleration)
- `pillow`
- `scikit-learn`

These dependencies are installed at the beginning of the notebook.

## How to Run

## How to Use  
1. Clone and access the repository:  
   ```bash
   git clone https://github.com/phaa/crop-pests-classifier.git
   cd crop-pests-classifier/
   ```
2. Activate the virtual environment (conda or venv):
   ```bash
   conda activate ibmenv
   ```
3. Run the notebooks in Jupyter lab:  
   ```bash
   jupyter lab
   ```
*The notebook has a cell to install the necessary dependencies.* 

## Results and Analysis

The project compares the performance of a CNN built from scratch with a model using transfer learning. The analysis of the training history, classification reports, and confusion matrices demonstrates the effectiveness of transfer learning in achieving better accuracy and avoiding overfitting, especially with a potentially limited dataset.

## Author

[Pedro Henrique Amorim de Azevedo](https://www.linkedin.com/in/pedro-henrique-amorim-de-azevedo/)
