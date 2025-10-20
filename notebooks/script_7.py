
# Create requirements.txt
requirements_content = '''# Core Data Science Libraries
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0

# Machine Learning
scikit-learn>=1.2.0

# Text Processing
nltk>=3.8.0

# Jupyter
jupyter>=1.0.0
ipykernel>=6.19.0

# Utilities
tqdm>=4.64.0
openpyxl>=3.0.10

# Optional: Advanced NLP
# textblob>=0.17.0
# vaderSentiment>=3.3.2
'''

with open('data_curriculum/requirements.txt', 'w') as f:
    f.write(requirements_content)

print("Created: requirements.txt")

# Create LICENSE (MIT License)
license_content = '''MIT License

Copyright (c) 2025 Restaurant Inspection Analysis Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

with open('data_curriculum/LICENSE', 'w') as f:
    f.write(license_content)

print("Created: LICENSE")

# Create empty placeholder files in figures directory
placeholder_files = [
    'boxplot_yelp_vs_grade.png',
    'scatter_sentiment_vs_score.png',
    'feature_importance.png'
]

with open('data_curriculum/figures/.gitkeep', 'w') as f:
    f.write('# Placeholder to keep figures directory in version control\n')

print("Created: figures/.gitkeep (placeholder)")
