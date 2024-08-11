# Food Image Classification Using PCA and Randomized PCA

This project is part of a lab assignment that focuses on food classification using the Food-11 image dataset. The goal is to develop a machine learning model that can accurately categorize food items into 11 major categories. This project also explores the potential applications of such a model in the food industry, such as digital menu automation, nutritional analysis, and inventory management.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Data](#data)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling and Evaluation](#modeling-and-evaluation)
- [Business Applications](#business-applications)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Project Overview

The project explores various methods including Principal Component Analysis (PCA) and DAISY feature extraction, followed by K-Nearest Neighbor (KNN) classification. The project is structured to support both exploratory data analysis (EDA) in Jupyter Notebooks and script-based execution for production or further analysis.

## Directory Structure

```plaintext
NutriTake-Food-Classification/
│
├── notebooks/
│   └── EDA.ipynb                      # Exploratory Data Analysis notebook(lab assignment)
│
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── data_preparation.py            # Data loading and preparation
│   ├── pca_analysis.py                # PCA analysis and plotting
│   ├── daisy_features.py              # DAISY feature extraction
│   ├── knn_classification.py          # KNN classification logic
│   └── main.py                        # Main script to run the entire pipeline
│
│
├── README.md                          # Project overview and instructions
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Ignored files and directories
└── LICENSE                            # License for the project
```

## Getting Started

### Prerequisites

To run this project, you'll need to have Python installed. The specific version used in this project is Python 3.8 or later.

### Installation

#### 1. Clone the Repository:

```bash
git clone https://github.com/minhosong88/food-image-classification.git
cd food-image-classification
```

#### 2. Install the Required Packages:

Use the following command to install all the necessary dependencies:

```bash
pip install -r requirements.txt

```

### Usage

#### Running the Python Scripts

You can run the entire analysis pipeline using the main.py script:

```bash
python src/main.py
```

#### Exploratory Data Analysis

If you prefer interactive exploration, open the EDA.ipynb notebook:
jupyter notebook notebooks/EDA.ipynb

## Data

The dataset used in this project is the Food-11 image dataset from Kaggle. The dataset contains 16,643 images categorized into 11 major food categories such as bread, dairy, eggs, and meat. Due to the scope of this lab assignment, a random subset of 1,500 images is used for analysis.

## Exploratory Data Analysis

The EDA.ipynb notebook provides an exploratory analysis of the dataset, including:

Distribution of food categories
Visual inspection of sample images
Statistical analysis of pixel intensity
Initial observations and insights

## Modeling and Evaluation

The project includes several steps to process and model the data:

Data Preparation: Loading and resizing the images.
**1. PCA Analysis:** Applying Principal Component Analysis to reduce dimensionality.
**2. DAISY Feature Extraction:** Extracting DAISY descriptors for additional features.
**3. KNN Classification:** Using K-Nearest Neighbor for classification based on both PCA and DAISY features.
**4. Evaluation:** Comparing the performance of PCA and DAISY features using accuracy metrics.

## Business Applications

This classification model can be applied in various areas within the food industry:

1. **Digital Menu Automation**: Automate the digitization of restaurant menus by classifying dish images.
2. **Nutritional Analysis**: Assist consumers and dietitians in tracking nutritional intake by categorizing food items.
3. **Supply Chain Management**: Help food retailers manage inventory by classifying and tracking food items.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please fork the repository and submit a pull request.

## Contact Information

If you have any questions, suggestions, or would like to discuss the project further, feel free to contact me:

- **GitHub**: [minhosong88](https://github.com/minhosong88)
- **Email**: [hominsong@naver.com](mailto:hominsong@naver.com)

## References

- Kaggle. Food-11 image dataset URL: [https://www.kaggle.com/datasets/trolukovich/food11-image-dataset](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset)
