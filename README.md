# Kaggle Playground Series S4E10 - Loan Approval Prediction

Welcome to the repository for our solution to the Kaggle Playground Series (Season 4, Episode 10) competition. In this challenge, we aim to predict whether a loan application will be approved using various machine learning models and feature engineering techniques. Our current ranking is **377th** out of **2,419** participants, and we continue to refine our approach.

## Competition Overview

**Goal**: The objective of the competition is to predict the likelihood of a loan being approved (`loan_status`) based on several applicant features like income, age, loan intent, and credit history.

Competition Link: [Kaggle Competition - Playground Series S4E10](https://www.kaggle.com/competitions/playground-series-s4e10)

## Team Members

- **Kartik Garg** (GitHub: [kartikgarg74](https://github.com/kartikgarg74))
- **Deepanshu** (GitHub: [I-Deepanshu](https://github.com/I-Deepanshu))

We worked together on this competition and collaborated on feature engineering, model optimization, and stacking models to improve our predictions.

## Repository Structure

- `loan_93745.ipynb`: The main Jupyter notebook containing all code related to preprocessing, model training, and submission generation.
- `submission.csv`: The final submission file that we generated from our model's predictions.

## Approach

### 1. Data Preprocessing
We engineered several features to capture important relationships:
- **Debt-to-income ratio (`dti_ratio`)**: Calculated as `monthly_debt / monthly_income`.
- **Income to age ratio**: Added to capture how applicant income varies with age.
- **Loan amount to income ratio**: To assess the relative size of the loan request compared to income.

We applied the following preprocessing steps:
- Numerical columns were imputed with the median and standardized.
- Categorical columns were imputed with the most frequent value and one-hot encoded.

### 2. Models Used
We utilized a stacking ensemble to combine the predictions from multiple models:
- **XGBoost**: Hyperparameter optimization was done using Optuna.
- **LightGBM**
- **CatBoost**

The final estimator for stacking was a **Logistic Regression** model, which aggregated predictions from the individual models.

### 3. Stacking & Model Optimization
We experimented with a stacking classifier to leverage the strengths of different models. Our final submission was generated using the following steps:
- Stacking model combining XGBoost, LightGBM, and CatBoost.
- Final predictions were generated using a logistic regression as the meta-model.

### 4. Submission
After preprocessing the test data, we generated the predicted probabilities and saved the results in a CSV format for submission.

## Results

As of now, our model ranks **377th out of 2,419 participants**. We are continuously working on improving the model by fine-tuning hyperparameters and exploring additional feature engineering techniques.

## How to Use This Repository

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/loan-approval-prediction.git
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   - Open the `loan_93745.ipynb` notebook.
   - Follow the steps to preprocess the data, train the model, and generate predictions.

4. Generate predictions:
   - Once the model is trained, you can generate predictions using the `test_data_processed` dataset.
   - Save the results as a CSV file using:
     ```python
     submission.to_csv('submission.csv', index=False)
     ```

## Next Steps
- **Improvement in Feature Engineering**: We are exploring additional features like interaction terms and advanced techniques such as feature selection using SHAP values.
- **Model Tuning**: We'll continue to optimize the hyperparameters for the individual models within the ensemble.
- **Ranking Improvement**: We aim to move up in the leaderboard by experimenting with different stacking approaches.

## Contributions
We welcome contributions! If you have any ideas or improvements, feel free to create an issue or open a pull request.
