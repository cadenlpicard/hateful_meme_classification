
# Multimodal Hateful Meme Classification

This project focuses on classifying hateful memes by using both text and image data (multimodal). The data used for training is the Hateful Memes Dataset from Facebook AI (https://ai.meta.com/tools/hatefulmemes/). It implements several models for text-only, image-only, and multimodal (feature-level and decision-level fusion) classification to explore how each modality contributes to identifying hateful content.

## Project Overview

This repository contains code to develop, train, and evaluate several models for hateful meme classification:
1. **Image-only Classifier**: Uses image features alone for classification.
2. **Text-only Classifier**: Uses text features alone for classification.
3. **Feature-level Fusion Model**: Concatenates image and text features before classification.
4. **Decision-level Fusion Model**: Trains separate classifiers for text and image features, then combines their outputs.

The code handles data preparation, model training, and evaluation, and provides methods to optimize batch sizes and memory usage for efficient GPU runtime in Google Colab.

## Installation

To run this code, ensure the following packages are installed:
```bash
pip install torch transformers pillow pytesseract tqdm
```

## Data Preparation

The dataset is expected to be in the format of `.jsonl` files containing image paths and text labels. Ensure the dataset is structured as follows:
- `train.jsonl` for training data
- `dev_seen.jsonl` for development data (validation)
- Each entry in the dataset should contain an `img` field (image path) and `text` field.

Images should be stored in an `img` folder within the dataset directory.

## Model Structure

### 1. Text Feature Extraction
The code uses a BERT-based model to extract features from text data. Text is tokenized and encoded on-the-fly using a custom `MemeDataset` class.

### 2. Image Feature Extraction
Image features are extracted using a pretrained ResNet-50 model. Images are resized and preprocessed before feature extraction, and any images larger than 375 KB are excluded to optimize memory usage.

### 3. Multimodal Fusion
Two fusion methods are implemented:
- **Feature-level Fusion**: Combines text and image features into a single vector for classification.
- **Decision-level Fusion**: Separately classifies text and image data, combining the results to make a final prediction.

## Usage

### 1. Setup Colab for GPU
Ensure Google Colab is set to GPU runtime:
- Go to `Runtime > Change runtime type` and select **GPU**.

### 2. Run the Code
Use the following steps:
- **Data Loading**: JSONL files are loaded and preprocessed with progress tracking.
- **Feature Extraction**: Extract text and image features using the models.
- **Training**: Models are trained, with batch sizes and memory usage optimized for GPU.
- **Evaluation**: Accuracy, precision, recall, F1 score, and AUC-ROC metrics are calculated on the development set.

### 3. Batching by Size
The code optimizes memory usage by batching data based on cumulative size rather than file count. Each batch size is capped at 150 MB, balancing GPU memory and efficiency.

## Results

The project includes various models trained and tested on hateful meme data. Results are stored in table format to track performance metrics across models. Final testing is conducted on a held-out test set, with metrics presented for each model.

| Model                         | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------------------------------|----------|-----------|--------|----------|---------|
| Image Classifier              | -        | -         | -      | -        | -       |
| Text Classifier               | -        | -         | -      | -        | -       |
| Feature-Level Fusion          | -        | -         | -      | -        | -       |
| Decision-Level Fusion         | -        | -         | -      | -        | -       |
| Final Model on Test Set       | -        | -         | -      | -        | -       |

## Key Learnings

This project highlights the importance of multimodal approaches for complex classification tasks like hateful meme detection. Text features tend to have a stronger signal, while image data alone may not suffice. Combining these features at the decision level or feature level improves overall performance, with feature-level fusion slightly outperforming decision-level fusion.

## License

This project is open-source and available for educational use.

