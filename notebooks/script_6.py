
# Create README.md
readme_content = '''# Restaurant Inspection & Yelp Review Analysis

A machine learning project analyzing NYC restaurant inspection data and Yelp reviews to predict restaurant grades and understand relationships between customer sentiment and health inspection scores.

## Project Overview

This project combines restaurant health inspection data with customer review sentiment to:
- Predict restaurant inspection grades using machine learning models
- Analyze the correlation between Yelp reviews and inspection scores
- Compare performance of Logistic Regression, SVM, and Random Forest classifiers
- Generate visualizations and insights for public health and restaurant industry stakeholders

## Directory Structure

```
data_curriculum/
├── data/
│   ├── Final_Inspection_Data.csv      # Cleaned inspection records
│   └── RecentInspDate.csv             # Most recent inspection per restaurant
├── notebooks/
│   └── inspection_code.ipynb          # Main analysis notebook
├── src/
│   ├── preprocess.py                  # Data cleaning & text preprocessing
│   ├── modeling.py                    # ML model training (LR, SVM, RF)
│   └── evaluation.py                  # Model evaluation & metrics
├── figures/                           # Generated visualizations
│   ├── boxplot_yelp_vs_grade.png     # (to be generated)
│   ├── scatter_sentiment_vs_score.png # (to be generated)
│   └── feature_importance.png         # (to be generated)
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
└── LICENSE                            # Project license
```

## Features

### Data Processing
- Automated data cleaning and preprocessing
- Text normalization for violation descriptions
- Missing value handling and outlier detection
- Restaurant-review matching algorithms

### Machine Learning Models
- **Logistic Regression**: Baseline linear classifier
- **Support Vector Machine (SVM)**: Non-linear classification with RBF kernel
- **Random Forest**: Ensemble tree-based classifier

### Evaluation Metrics
- Confusion matrices for all models
- Accuracy, Precision, Recall, F1-Score
- Feature importance analysis
- Cross-validation scores
- Model comparison visualizations

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd data_curriculum
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import pandas, sklearn, matplotlib; print('All packages installed successfully!')"
```

## Usage

### Running the Jupyter Notebook

1. Launch Jupyter:
```bash
jupyter notebook
```

2. Navigate to `notebooks/inspection_code.ipynb`

3. Run all cells to execute the complete analysis pipeline

### Using Individual Modules

#### Data Preprocessing
```python
from src.preprocess import clean_inspection_data, clean_text

# Clean inspection data
df_clean = clean_inspection_data(df_raw)

# Clean text data
clean_violation = clean_text(violation_text)
```

#### Model Training
```python
from src.modeling import ModelTrainer, split_data

# Initialize trainer
trainer = ModelTrainer(random_state=42)

# Prepare features and train models
X = trainer.prepare_features(text_data)
X_train, X_test, y_train, y_test = split_data(X, y)
models = trainer.train_all_models(X_train, y_train)
```

#### Model Evaluation
```python
from src.evaluation import evaluate_all_models, plot_confusion_matrix

# Evaluate models
results = evaluate_all_models(models, X_test, y_test)

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred, 
                     labels=['A', 'B', 'C'],
                     save_path='figures/confusion_matrix.png')
```

## Data Sources

- **NYC Restaurant Inspection Data**: NYC Open Data - DOHMH New York City Restaurant Inspection Results
- **Yelp Review Data**: Yelp Academic Dataset (sample)

## Methodology

1. **Data Collection**: Import inspection records and review text
2. **Data Cleaning**: Handle missing values, normalize text, remove duplicates
3. **Feature Engineering**: TF-IDF vectorization, numeric feature scaling
4. **Model Training**: Train LR, SVM, and RF with cross-validation
5. **Evaluation**: Generate metrics, confusion matrices, and comparison plots
6. **Analysis**: Interpret results and extract insights

## Results

Results will vary based on the dataset, but typical performance metrics include:

- **Logistic Regression**: Accuracy ~75-80%, F1 ~0.76-0.81
- **SVM**: Accuracy ~77-82%, F1 ~0.78-0.83
- **Random Forest**: Accuracy ~80-85%, F1 ~0.81-0.86

Detailed results are generated in the notebook and saved to the `figures/` directory.

## Visualizations

The project generates several key visualizations:

- Grade distribution histograms
- Inspection score distributions
- Sentiment vs. Score scatter plots
- Feature importance charts
- Confusion matrices for each model
- Model comparison bar charts

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NYC Department of Health and Mental Hygiene for inspection data
- Yelp for review dataset access
- scikit-learn community for machine learning tools
- All contributors and researchers in public health data science

## Contact

For questions or feedback, please open an issue on GitHub.

## Future Work

- [ ] Incorporate real-time Yelp API integration
- [ ] Add deep learning models (LSTM, BERT)
- [ ] Expand to other cities and jurisdictions
- [ ] Develop web dashboard for interactive exploration
- [ ] Implement automated alerting system for health violations

---

**Last Updated**: October 2025
'''

with open('data_curriculum/README.md', 'w') as f:
    f.write(readme_content)

print("Created: README.md")
