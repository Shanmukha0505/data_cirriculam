# Reality Check: Restaurant Reviews vs Health Inspections

**Author:** Shanmukha Sai Venkat Medisetti  
**Course:** Data Science Practicum  
**Instructor:** Dr. Christy Pearson  

---

## Abstract

This project explores whether **Yelp reviews** can be used to predict restaurant food safety outcomes in **New York City**.  
Since official inspection information is not always immediately public, online review platforms could provide early warning signals for customers and regulators.

The study merges **NYC Health Department inspection data** with **Yelp reviews**, applying feature engineering on both textual and numeric attributes.  
Predictive models such as **Logistic Regression**, **Support Vector Machines (SVM)**, and **Random Forest Classifiers** were built to estimate inspection outcomes.

Exploratory analysis revealed a small correlation between Yelp ratings and inspection grades. Predictive models achieved moderate accuracy, suggesting Yelp data may supplement—but not replace—official inspections.  
The findings have implications for public health monitoring, ethical data usage, and future AI-driven inspection systems.

---

## Introduction

Foodborne illness remains a major global public health concern. Consumers need reliable, **real-time information** about restaurant hygiene and food safety, but official inspection results are not always immediately available.

Websites like **Yelp** provide rich, user-generated insights into restaurant experiences, yet their accuracy in reflecting true safety conditions remains uncertain.  
This study asks two main questions:

1. How strongly do Yelp ratings correlate with official NYC inspection grades?  
2. Can we predict inspection outcomes using Yelp-derived features?

Understanding this relationship can help consumers make safer choices and give regulators early-warning indicators for potential food safety violations.

---

## Methods

### Data Sources

- **NYC Health Department inspections**: Provided grades (A/B/C), violation types, and inspection dates.  
- **Yelp reviews dataset**: Included restaurant names, ratings (1–5 stars), review text, and dates.

Together, these datasets enabled a comparison between **public perception** and **official safety results**.

---

### Data Cleaning and Matching

- Restaurant names and addresses were **normalized** to ensure proper alignment.  
- Duplicate or incomplete records were removed.  
- Yelp reviews were **time-aligned** with inspections within ±6 months.  
- This matching ensured that reviews and inspections reflected roughly the same operational period.

---

### Feature Engineering

- **Textual features:**  
  - Sentiment scores derived via **NLP**.  
  - Keyword counts related to cleanliness, hygiene, and food quality.

- **Numeric features:**  
  - Average star rating  
  - Review volume  
  - Recency relative to inspection dates  

These features served as predictors of inspection performance.

---

### Modeling Approach

Three predictive models were trained and evaluated:

- **Logistic Regression** → Interpretable probability-based classification  
- **Support Vector Machine (SVM)** → Captured non-linear relationships  
- **Random Forest** → Ranked feature importance and handled complex interactions  

Each model was tested using **accuracy**, **precision**, **recall**, **F1-score**, and **confusion matrices**.

---

## Results

### Exploratory Data Analysis

- Restaurants with **high Yelp ratings** often received **A grades** in official inspections — indicating a moderate positive relationship.  
- However, several **highly rated restaurants** had poor inspection results (B or C), showing inconsistencies between **public perception** and **regulatory assessment**.  
- The **Mismatch Index**, defined as the absolute difference between Yelp sentiment and inspection score, showed that mismatches, though not dominant, were **non-trivial**.

---

### Predictive Modeling

| Model | Key Traits | Performance Summary |
|-------|-------------|--------------------|
| **Logistic Regression** | Interpretable coefficients; strong correlation with mean rating | Moderate accuracy |
| **SVM** | Captured complex, non-linear relationships | Slightly higher accuracy |
| **Random Forest** | Best overall performer; ranked feature importance | Highest precision & recall |

**Top features:**  
- Average Yelp rating  
- Sentiment polarity score  
- Number of recent reviews  

While models achieved moderate predictive power, they confirm that **consumer sentiment holds useful—but incomplete—signals** about real inspection outcomes.

---

## Discussion

The results indicate that Yelp reviews **partially align** with official inspection results but are **not reliable substitutes**.  
High ratings correlate with better grades, yet **consumer opinions reflect taste and service** more than hygiene standards.

Regulators can still use Yelp data as **a supplementary monitoring tool**, identifying restaurants that may warrant earlier inspection.  
However, ethical issues arise when using user-generated content for regulatory purposes — since biases, false reviews, and exaggerations can misrepresent risk.

---

## Limitations & Future Work

### Current Limitations
- Yelp reviews may contain **inaccurate or biased information**.  
- **Time mismatches** between reviews and inspections reduce precision.  
- Limited to **NYC data**, restricting generalization across cuisines or regions.

### Future Enhancements

| Focus Area | Planned Enhancement | Goal / Expected Outcome |
|-------------|---------------------|--------------------------|
| **Temporal Modeling** | Implement ARIMA / LSTM models on Yelp & inspection trends | Detect seasonality and predict future outcomes |
| **Multi-City Comparison** | Extend dataset to cities like LA, Chicago, SF | Analyze regional variation in sentiment vs hygiene |
| **Dashboard Expansion** | Add cuisine, date, and severity filters | Improve interactive insights for users |
| **Advanced Text Analytics** | Use BERT / topic modeling for sentiment and keyword accuracy | Enhance text-based predictive power |
| **Policy Insights** | Provide data-driven suggestions for regulators | Strengthen public health and restaurant accountability |

---

## Conclusion

Yelp reviews **offer valuable early signals** about restaurant hygiene but cannot replace professional inspections.  
The analysis shows that prior performance (inspection history) and structured review data together can help **detect risk patterns** faster.  
For policymakers, these insights can improve **resource allocation** and **public transparency** — bridging the gap between consumer experience and food safety.

---

## References

Anderson, M., & Magruder, J. (2012). *Learning from the crowd: Regression discontinuity estimates of the effects of an online review database.* The Economic Journal, 122(563), 957–989.  
Luca, M. (2016). *Reviews, reputation, and revenue: The case of Yelp.com.* Harvard Business School Working Paper.  
NYC Health Department. (2023). *Restaurant inspection data.* Retrieved from [NYC Open Data](https://data.cityofnewyork.us/)  
Xu, A., Liu, L., Guo, Y., Sinha, V., & Akkiraju, R. (2017). *Mining consumer reviews for restaurant service quality: A sentiment analysis approach.* International Journal of Information Management, 37(2), 144–157.  

---
