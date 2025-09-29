# Data Curriculam Project

## Project Overview
This project explores the relationship between NYC Department of Health inspection results and public perception from Yelp reviews.  
The ultimate goal is to integrate both datasets, identify mismatches (e.g., restaurants with great reviews but poor inspection scores), and visualize the findings.

## Work Completed So Far

### 1. Data Collection
- Downloaded the **NYC DOHMH Restaurant Inspection Results** dataset.
- Gathered Yelp dataset (filtered to include NYC restaurants only).

### 2. Data Cleaning (Inspection Data)
- Converted `INSPECTION DATE` to proper datetime format.
- Converted `SCORE` column to numeric for filtering and analysis.
- Trimmed whitespace and standardized text fields such as `INSPECTION TYPE` and `GRADE`.
- Defined inspection type groups:
  - **Re-inspection**  
  - **Initial inspections with passing score (≤ 13)**  
  - **Reopening inspections**
- Filtered records to keep only valid grades (`A, B, C, P, Z`).
- Extracted **most recent inspection date per restaurant (CAMIS ID)**.
- Saved cleaned results into `RecentInspDate.csv`.

### Next Steps
- Merge Yelp and DOHMH datasets using:
  - Exact matches (phone/address),
  - Fuzzy string matching (name/address similarity),
  - Geospatial matching (latitude/longitude).
- Perform exploratory data analysis to identify trends and mismatches.
