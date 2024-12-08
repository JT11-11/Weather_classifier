# Weather Image Classification

This repository provides an end-to-end pipeline for image classification using PyTorch. It includes functionalities for:

- Loading and preprocessing images.
- Splitting datasets into training, validation, and test sets.
- Defining custom PyTorch `Dataset` classes.
- Training a convolutional neural network (CNN) model.
## Features

- **Data Handling:**  
  Uses `PIL` and `torchvision.transforms` for image loading and preprocessing.  
  Splits data into train/validation/test sets using `sklearn.model_selection.train_test_split`.

- **Model Training and Evaluation:**  
  Implements training loops with `torch.optim` and `nn.CrossEntropyLoss`.  
  Utilizes progress bars via `tqdm` to monitor training progress.  
  Supports GPU acceleration if available.

### Process:
1. Visualizing given data using matplotlib
2. Splitting the data using sklearn train_test_split function
3. Train the data using a CNN with the usage of the Pytorch framework.


### Future plans:
With the current model running at an average of 50% accuracy rate, I hope to increase it to 80-90% accuracy consistently by enchancing and fine-tuning the model. 


### Prerequisites

- Python 3.7+
- PyTorch and TorchVision
- PIL (via Pillow)
- scikit-learn
- tqdm
- matplotlib
- pandas
- numpy

You can install these dependencies with:

```bash
pip install torch torchvision pillow scikit-learn tqdm matplotlib pandas numpy
