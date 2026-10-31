# AI Usage

## Tools I Used and How

- **ChatGPT:** Used as a coding and learning assistant to understand the task requirements, discuss implementation approaches, review code, and troubleshoot errors.
- **VS Code:** Used to edit Python files, organize the project, and run commands in the integrated terminal.
- **Python and pytest:** Used to run the implementation and verify that the functions behave as expected.
- **pandas and NumPy:** Used for data manipulation and customer-history feature engineering.
- **scikit-learn:** Used to build the preprocessing and classification pipeline, train the models, and evaluate their performance.

I was involved in setting up the environment, implementing and reviewing the code, running tests, checking model metrics, and making implementation decisions with AI assistance.

## AI-Suggested Code I Reviewed or Fixed

I reviewed a potential issue with calculating customer history row by row: parcels belonging to the same customer with identical timestamps could incorrectly count as prior parcels for one another.

To avoid this, I used timestamp-level grouping so parcels with the same timestamp share the same history, and only strictly earlier parcels are counted.

I verified this behavior using a test with two parcels belonging to the same customer at the same timestamp and a third parcel at a later timestamp. The first two parcels had zero prior parcels, while the third had two prior parcels and a prior failure rate of 0.5.

## How I Checked That My Model Has No Data Leakage

I excluded parcel identifiers, the target label, and post-delivery information from the model features. Customer-history features use only strictly earlier parcels, and model evaluation uses a chronological train/test split to ensure that future parcels are not used to predict earlier ones.

On the seed 0 test set, the model achieved a ROC AUC of 0.6470, below the 0.9 leakage warning threshold. I also reviewed the selected features to ensure they represent information available at prediction time.