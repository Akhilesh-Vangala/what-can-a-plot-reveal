# Which plot for which failure

From the evidential map at the strongest severity. A plot is listed only where both data sources (synthetic, MagicTelescope) agree. *Diagnoses* = detects the failure and tells it apart from the others (recall >= 0.50); *detects* = shows that something is wrong, not what; *cannot show* = no benchmark reaches AUC 0.70.

| failure | best plot | also diagnoses | detects only | cannot show | depends on the data |
|---|---|---|---|---|---|
| overfitting | loss curves (recall 0.81) | accuracy curves, reliability diagram, reliability diagram + bin sizes | validation vs test score histogram | confusion matrix, confusion matrix (row %), ROC curve, precision-recall curve | – |
| miscalibration | none diagnoses it | – | reliability diagram, reliability diagram + bin sizes, validation vs test score histogram | loss curves, accuracy curves (proven), confusion matrix (proven), confusion matrix (row %) (proven), ROC curve, precision-recall curve | – |
| label noise | reliability diagram (recall 0.76) | reliability diagram + bin sizes, precision-recall curve, validation vs test score histogram, ROC curve | loss curves | – | accuracy curves, confusion matrix, confusion matrix (row %) |
| class imbalance | confusion matrix (recall 0.87) | reliability diagram + bin sizes, validation vs test score histogram, precision-recall curve, ROC curve, confusion matrix (row %) | – | loss curves | accuracy curves, reliability diagram |
| data leakage | precision-recall curve (recall 0.76) | confusion matrix, ROC curve, accuracy curves, loss curves, reliability diagram + bin sizes, confusion matrix (row %) | reliability diagram, validation vs test score histogram | – | – |
| distribution shift | reliability diagram + bin sizes (recall 0.76) | validation vs test score histogram, reliability diagram, confusion matrix, confusion matrix (row %) | ROC curve | loss curves (proven), accuracy curves (proven) | precision-recall curve |

## The usual dashboard: loss curves + confusion matrix + ROC

- **miscalibration**: the dashboard cannot show it. No plot diagnoses it; the reliability diagram + bin sizes at least detects it.
