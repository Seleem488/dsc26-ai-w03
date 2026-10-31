# Results: Tawseel failed-delivery baseline

## Baseline table

Evaluation on the last 20% of parcels by time:
`make_parcels(4000, seed=0)`, with 800 test parcels.

| Model | ROC AUC | PR-AUC | Brier | Base rate |
|---|---:|---:|---:|---:|
| Dummy (prior) | 0.5000 | 0.2200 | 0.1717 | 0.2200 |
| Rule: Not tried | N/A | N/A | N/A | N/A |
| My model | 0.6470 | 0.3820 | 0.1634 | 0.2200 |

## Three sentences for Sherif

The model estimates the risk of a COD parcel failing using information available at prediction time, including parcel details and customer history.

On the seed 0 time-based test set, it achieved a PR-AUC of 0.3820 compared with 0.2200 for the dummy baseline, and a lower Brier score of 0.1634 compared with 0.1717.

These results are promising, but the model is evaluated on synthetic data and does not guarantee delivery outcomes or prove that its predictions will generalize to real-world deliveries.