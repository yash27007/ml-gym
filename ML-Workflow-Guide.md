# A Practical Guide to Solving Machine Learning Problems

A reusable, model-agnostic workflow for core machine learning and deep learning. It covers the questions to ask, the decisions to make, and the checks that prevent common mistakes across tabular, text, image, audio, and time-series projects.

No checklist can replace domain knowledge. Treat each recommendation as a default to validate against the prediction goal, data collection process, and cost of errors.

---

## The one-screen workflow

```text
Define the decision and prediction time
        ↓
Inspect data, labels, and collection process
        ↓
Choose a valid split that matches deployment
        ↓
Build a simple baseline
        ↓
Fit preprocessing on training data only
        ↓
Train candidate models / architectures
        ↓
Select with the right metric and validation
        ↓
Inspect errors, leakage, robustness, and calibration
        ↓
Evaluate once on untouched test data
        ↓
Plan deployment, monitoring, and retraining
```

Keep asking:

> What decision will this prediction support, what information is available when it is made, and what kind of error is costly?

---

## 1. Frame the problem before opening the notebook

Write a short problem statement:

- **Goal:** What decision, estimate, ranking, or grouping should the system support?
- **Unit of prediction:** What does one example represent: a person, image, document, event, or time window?
- **Target/output:** What exactly should the model predict? How is it measured or labeled?
- **Prediction time:** At what moment must the prediction be made?
- **Available inputs:** Which information exists at that moment?
- **Success measure:** Which metric reflects useful performance? What baseline must be beaten?
- **Constraints:** Latency, memory, interpretability, privacy, data volume, update frequency, and inference cost.
- **Error costs:** What happens after false positives, false negatives, or large numeric errors?

Choose the problem family:

| Problem | Output | Typical examples |
|---|---|---|
| Regression | A number | price, demand, duration |
| Binary classification | One of two classes / probability | fraud, churn |
| Multiclass classification | One of several classes | topic, species |
| Multilabel classification | Any subset of labels | tags, multiple findings |
| Forecasting | Future value(s) | sales next week |
| Ranking | Ordered candidates | search results, recommendations |
| Clustering | Groups without known labels | customer segments |
| Representation / dimensionality reduction | Compact features | visualization, compression |

For clustering and other unsupervised tasks, there may be no ground-truth target. Define what makes a result useful and how a domain expert or downstream task will validate it.

---

## 2. Understand the data and its provenance

Before modeling, establish:

- What does one row, file, sequence, or example mean?
- How were examples sampled? Which population and time period do they represent?
- How were labels produced? Are they noisy, delayed, subjective, or inconsistent?
- Are there repeated entities, near-duplicates, multiple views, or repeated measurements?
- Which features exist at prediction time? Could any be caused by, derived from, or recorded after the outcome?
- Does the dataset reflect deployment, or is it a benchmark / curated sample?
- Are there privacy, consent, or sensitive-attribute considerations?

Create a data inventory: feature name, meaning, type, units, valid values, source, timestamp, missingness, and whether it is available at inference.

For a first inspection, record dataset size and shape, schema, target counts or range, missingness, duplicates, class balance, and examples of malformed records. Check train and test sources for overlap where possible.

### Leakage check

For every input, ask:

> Would I know this value at the exact moment I need to make this prediction?

Leakage can be direct (target copied into a feature), temporal (future information included in the past), or indirect (a post-outcome process reveals the answer). Also watch for duplicate or related examples crossing splits, preprocessing fit on the full dataset, and repeated hyperparameter selection against the test set.

---

## 3. EDA checklist: the 12 questions

Use these questions to guide exploratory data analysis (EDA). EDA is the process of checking what the data contains, finding quality issues and patterns, and turning those findings into decisions before and during modeling. You do not need every plot for every dataset; choose a check that answers a question.

1. **What is the prediction target?** Confirm its meaning, label source, type, units, valid range, and timing. For unsupervised work, state what structure or outcome you want to discover.
2. **What does each feature actually mean?** Use a data dictionary or domain source; record how and when it was collected and whether it is available at prediction time.
3. **What are the feature types?** Distinguish continuous values, counts, nominal/ordinal categories, IDs, dates, text, images, and sequences. Do not infer meaning from storage type alone.
4. **What are the units and valid ranges?** Check summary statistics and domain constraints for impossible values, inconsistent units, and encoding mistakes.
5. **Are there missing values?** Count them and inspect where/when they occur; ask why values are absent before choosing how to handle them.
6. **Are there duplicates?** Check exact and near duplicates, repeated entities, and repeated measurements. Decide whether each represents an error or a valid observation.
7. **Is the target balanced?** For classification, count each class and compare proportions; for regression, inspect range, skew, heavy tails, and extreme outcomes.
8. **What is the distribution of each important feature?** Use histograms/quantiles for numeric data and count bars for categories; note skew, rare categories, constant fields, and unusual shapes.
9. **Which features are related to each other?** Use correlations and scatter plots for numeric pairs, contingency/proportion tables for categories, and domain knowledge to spot redundancy. Association does not imply causation.
10. **Which features appear related to the target?** Compare numeric distributions or residual patterns and inspect target rates by category. Treat these as hypotheses to validate, not proof of predictive value.
11. **Are there outliers or suspicious values?** Investigate against source records and domain limits. Do not delete an extreme value until you know whether it is an error or a valid rare case.
12. **Is there domain knowledge that should influence feature construction?** Look for meaningful transformations, time windows, thresholds, interactions, and known measurement conventions; test them using leakage-safe validation.

### Quick plot map

| Question | Useful first checks |
|---|---|
| One numeric feature: distribution or extremes? | Histogram / quantiles; boxplot for screening |
| One categorical feature: what values and frequencies? | Count bar chart; inspect rare levels |
| Numeric feature vs numeric feature? | Scatter plot; correlation for linear association |
| Numeric feature vs classification target? | Per-class box/violin plots or distributions; binned event rate when meaningful |
| Categorical feature vs target? | Cross-tabulation and within-category target proportions |
| Feature vs time? | Time plot; check trend, seasonality, gaps, and changes in data source |

For every EDA finding, write: **observation → why it matters → decision to try → validation check**. EDA helps form modeling choices; validation determines whether those choices improve generalization.

---

### Explore with questions, not a plot quota

First classify each input by meaning, not just storage type:

- Continuous numeric, count, ordinal, nominal category, identifier, timestamp, text, image, audio, sequence, or structured object.
- A number may be a category or ID. A string may be a date, number, or category.
- Check units, impossible values, inconsistent labels, unexpected categories, and inconsistent formats.

Then investigate:

### Target

- Classification: class counts and proportions; rare classes; label quality.
- Regression: range, units, skew, heavy tails, multimodality, impossible values, and extreme outcomes.
- Forecasting: coverage over time, seasonality, missing intervals, and forecast horizon.

### Inputs

- Missing values and whether their pattern differs by target or group.
- Distributions, rare categories, outliers, and constant or near-constant fields.
- Redundant features, correlations, duplicated records, and group structure.
- Feature-to-target relationships, using plots and summaries appropriate to variable type.
- Distribution differences across time, source, geography, or other important subgroups.

Use visualizations to answer a question: histograms for numeric distributions; count bars for categories; scatter plots for numeric pairs; box/violin plots or grouped summaries for numeric values by class; event rates for categorical features against a binary target; time plots for temporal structure. Correlation is a clue about linear association, not proof of causation, usefulness, or independence.

Treat outliers as evidence to investigate, not automatic deletion candidates. They may be errors, valid rare events, or the cases the model most needs to handle.

Finish EDA by writing each finding as:

```text
Observation → why it matters → decision to try → validation check
```

Example: “A few measurements are missing; missingness may reflect the collection process → add train-fitted imputation and optionally a missing indicator → compare validation performance and inspect subgroup errors.”

---

## 4. Split data to imitate real use

Choose the split before learning any preprocessing statistics or selecting features.

- **Random split:** independent, identically sampled examples with no meaningful grouping or time order.
- **Stratified split:** classification when preserving class proportions is important, especially for rare classes.
- **Group split:** keep all records from a person, device, household, document, or other entity in one partition.
- **Time split:** train on the past and validate/test on later periods when predicting the future. Use a gap if labels or features overlap across windows.
- **Spatial/source split:** hold out regions, sites, cameras, or data sources when generalization to new ones matters.

Typical setup: training data for fitting; validation data or cross-validation for model and threshold choices; a final test set held untouched until decisions are complete. With limited data, use cross-validation inside the training portion. Keep preprocessing, feature selection, resampling, and augmentation within each training fold.

A random split is not automatically valid. The split should prevent a model from seeing near-duplicates, the same entity, or future information during training if those will not be available in deployment.

---

## 5. Preprocessing: reliable defaults

**Core rule:** anything that learns from data must be fitted on the training portion only, then applied unchanged to validation, test, and production inputs. Put these steps in a reproducible pipeline.

### Tabular data

- **Missing numeric values:** median is a robust starting point; mean can suit roughly symmetric data. Consider a missingness indicator when absence itself may carry information. Do not impute using the target or values from held-out rows.
- **Missing categories:** use an explicit “unknown/missing” category or a train-fitted most-frequent category. Handle categories appearing only at inference.
- **Categorical encoding:** one-hot encoding is a sound default for nominal features with manageable cardinality. Ordinal encoding is appropriate only when order is meaningful. High-cardinality categories may need regularized target encoding, hashing, or embeddings; fit any target-based encoding within folds to prevent leakage.
- **Scaling:** standardization (subtract training mean, divide by training standard deviation) often helps linear models, SVMs, KNN, and neural networks. Min-max scaling can be useful when bounded inputs are desired. Robust scaling can reduce sensitivity to extreme values. Tree models usually need no scaling.
- **Skew and outliers:** verify units and data errors first. Transformations such as log1p may help positive, right-skewed values. Clip/winsorize only with a justified rule learned from training data; trees often tolerate raw scales and outliers better than distance-based or linear models.
- **Dates:** derive only information available at prediction time (e.g. age of record, calendar parts, elapsed time). Encode cyclic values such as hour-of-day with sine/cosine when useful. Never let future timestamps or post-outcome dates leak in.
- **IDs:** usually exclude arbitrary identifiers as predictive features, but retain them for joins, grouped splitting, auditing, and error analysis. An ID can encode useful structure; test that hypothesis against a deployment-matched split.
- **Class imbalance:** first choose appropriate metrics and thresholds. If using class weights or resampling, apply them only to training folds. Oversampling before splitting leaks near-duplicate information.

### Text

- Normalize only what the task allows; preserve casing, punctuation, or formatting if it carries signal.
- Use a tokenizer and vocabulary fitted on training data for learned-from-scratch models. Pretrained tokenizers/encoders should match the chosen model.
- Truncate or chunk long inputs deliberately; record how this affects labels and context.
- For pretrained models, use the model's expected input normalization and special tokens. Do not fit a vocabulary on validation/test text.

### Images

- Check dimensions, channels, color order, bit depth, corrupted files, label alignment, and duplicate/near-duplicate images.
- Normalize pixels consistently; use the pretrained model's expected normalization when applicable.
- Augment training images only, and keep transformations label-preserving. Do not apply random augmentation to validation/test evaluation.
- Split by subject, source, scene, or video when related images could otherwise cross partitions.

### Audio and sequences

- Standardize sample rate, channels, units, padding/masking, and windowing consistently.
- Preserve temporal order when it matters; define sequence length and truncation/window rules.
- Fit any normalization statistics on training data only. Split by speaker, recording, device, or entity when needed.
- Avoid windows from the same underlying recording crossing splits.

### Targets

- Encode labels consistently; preserve a mapping from encoded classes back to names.
- For regression transforms (such as log target), fit/define the transform using training data and invert predictions before reporting in original units. Evaluate in the unit that matches the decision.
- Never use target information to construct inputs unless the setup explicitly supports that information at inference.

---

## 6. Build baselines before complexity

A baseline is a reference point and a debugging tool.

- Regression: predict the training mean or median; for time series, try the last observed value or seasonal naive forecast.
- Classification: predict the majority class; also consider a simple probabilistic baseline.
- Text/image: start with a simple established model or a frozen pretrained representation where suitable.
- Unsupervised: compare with a simple clustering/representation and inspect stability and downstream usefulness.

Then train a modest, reproducible model. Examples: regularized linear/logistic regression, a small tree or random forest/boosted tree for tabular data, and a standard pretrained or small neural architecture for unstructured inputs. Do not assume the largest architecture is best. Match model complexity to data volume, structure, latency, and available compute.

### Model-family cues

| Model family | Often a good fit | Pay attention to |
|---|---|---|
| Linear / logistic models | Strong baseline; approximately additive signal; interpretability | Scaling, encoding, nonlinear terms, collinearity, regularization |
| Trees / random forests / gradient boosting | Tabular nonlinearities and interactions | Leakage, overfitting, categorical handling, calibration |
| SVM / KNN | Smaller datasets; meaningful distance or margin structure | Scaling, feature dimension, memory and inference cost |
| Neural networks | Large or unstructured data; learned representations; flexible interactions | Data volume, normalization, initialization, regularization, training stability |
| CNNs / vision transformers | Images and spatial patterns | Resolution, augmentation, pretrained normalization, split by source |
| Sequence models / temporal convolutions / transformers | Ordered or sequential inputs | Temporal split, masks, context length, forecast horizon, future leakage |

These are starting points, not rules. Compare candidates using the same valid splits and metric.

---

## 7. Train and select without fooling yourself

- Fix random seeds where practical and record data version, split, preprocessing, model configuration, and metric.
- Fit preprocessing and model parameters inside each training fold. For small datasets, cross-validation can reduce dependence on one split.
- Tune only on training/validation data. Keep a test set for a final, limited evaluation.
- Use early stopping based on validation data when appropriate. Restore the best validation checkpoint.
- Compare against the baseline and report variation across folds or seeds when meaningful.
- Tune the decision threshold separately from ranking quality. A default threshold of 0.5 is not automatically optimal.
- Keep an experiment log so comparisons are reproducible.

For neural networks, monitor training and validation loss/metrics together. Training improves while validation worsens: likely overfitting. Both remain poor: investigate data, labels, optimization, model capacity, or preprocessing. Unstable loss: inspect learning rate, scaling, initialization, batch composition, and numerical issues.

---

## 8. Quick Formula Reference

### Basic statistics

```text
Mean:       x̄ = (1/n) Σᵢ xᵢ
Variance:   s² = (1/(n−1)) Σᵢ (xᵢ − x̄)²
Std dev:    s = √s²
Z-score:    z = (x − μ) / σ
```

### Regression losses / metrics

```text
MAE  = (1/n) Σᵢ |yᵢ − ŷᵢ|
MSE  = (1/n) Σᵢ (yᵢ − ŷᵢ)²
RMSE = √MSE
R²   = 1 − [Σᵢ (yᵢ − ŷᵢ)² / Σᵢ (yᵢ − ȳ)²]
```

R² compares squared error to predicting the evaluation-set mean; 1 is perfect, 0 matches that reference, and it can be negative. It is not “percent correct” and does not show error magnitude in the target's units. Pair it with MAE or RMSE.

### Classification

```text
Precision = TP / (TP + FP)      Of predicted positives, how many were correct?
Recall    = TP / (TP + FN)      Of actual positives, how many were found?
Specificity = TN / (TN + FP)    Of actual negatives, how many were rejected?
F1        = 2 × (Precision × Recall) / (Precision + Recall)
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
```

```text
Log loss (binary) = −(1/n) Σᵢ [yᵢ log(pᵢ) + (1−yᵢ) log(1−pᵢ)]
```

Here TP/FP/TN/FN are true/false positives/negatives. Precision and recall require a chosen threshold; PR-AUC and ROC-AUC summarize ranking over thresholds.

### Deep learning / optimization

```text
Gradient descent:       θ ← θ − η ∇θ L(θ)
L2 regularization:       L_total = L_data + λ Σⱼ θⱼ²
Softmax:                 pₖ = exp(zₖ) / Σⱼ exp(zⱼ)
Sigmoid:                 σ(z) = 1 / (1 + exp(−z))
```

`η` is the learning rate; `λ` controls regularization. Cross-entropy is a common classification loss. The training loss is an optimization objective; the evaluation metric should reflect the actual goal.

---

## 9. Which evaluation metric should I use?

Pick metrics from the consequence of errors, target type, and data balance. Set the primary metric before comparing models.

| Situation | Useful primary metrics | Why / caution |
|---|---|---|
| Balanced classification; errors similarly costly | Accuracy; macro-F1 if class-wise balance matters | Accuracy is intuitive but can hide minority-class failures |
| Rare positive class; finding positives matters | Recall, PR-AUC; Fβ with β > 1 when a threshold is needed | PR-AUC focuses on positive-class retrieval; inspect precision at a useful recall |
| False alarms are costly | Precision; precision at required recall | A high precision model may miss many positives |
| Need balance of precision and recall | F1; PR-AUC for threshold-independent ranking | F1 ignores true negatives and does not include error costs directly |
| Compare ranking across thresholds | ROC-AUC; PR-AUC for rare positives | AUC measures ranking, not calibration or performance at one operating threshold |
| Probabilities must be reliable | Log loss, Brier score, calibration curve | Good ranking does not guarantee trustworthy probabilities |
| Regression; typical absolute error matters | MAE | Same units as target; less sensitive to large errors |
| Regression; large errors should cost more | RMSE / MSE | Strongly penalizes large misses; sensitive to outliers |
| Regression; compare to mean predictor | R², alongside MAE/RMSE | Relative squared-error reference; can be negative, depends on target variation |
| Forecasting | MAE/RMSE plus a seasonal-naive comparison; MASE where appropriate | Respect time order; avoid percentage metrics near zero |
| Multiclass with imbalanced classes | Macro-F1, balanced accuracy, per-class recall; weighted-F1 as a summary | Macro averages classes equally; weighted averages by support |
| Clustering | Silhouette or stability plus domain/downstream validation | No single internal score proves clusters are useful |

**Metric reminders**

- Accuracy = correct predictions / all predictions. A majority-class predictor can score highly on an imbalanced dataset.
- F1 is the harmonic mean of precision and recall. Fβ gives recall more weight when β > 1 and precision more weight when β < 1.
- ROC-AUC measures how often a random positive is ranked above a random negative. PR-AUC is often more informative for rare positives; its baseline depends on positive prevalence.
- A probability model can rank well yet be poorly calibrated. Check calibration if decisions use probability values.
- For multiclass tasks, state averaging explicitly: macro (equal class weight), micro (aggregate decisions), or weighted (class-frequency weighted).
- Report per-class metrics and a confusion matrix when errors differ by class.
- For any metric, include uncertainty or fold-to-fold variation when the decision warrants it.

---

## 10. Evaluate errors and decide what to improve

Look beyond one aggregate score:

- Inspect false positives, false negatives, largest residuals, and representative correct predictions.
- Slice metrics by important group, time period, source, class, or operating condition. Check sample counts; small slices are uncertain.
- For regression, plot residuals against predictions and key inputs; look for bias, changing variance, and systematic misses.
- For classification, inspect confusion matrix, threshold behavior, precision-recall tradeoffs, and calibration if probabilities drive decisions.
- Check train-versus-validation performance, data drift between splits, and performance under plausible input corruption or missingness.
- Identify label errors, ambiguity, and examples outside the training distribution.

Use a disciplined loop:

```text
Observed failure → likely cause → one targeted change → compare on the same validation scheme
```

Possible changes include fixing data quality, improving labels, adding domain-informed features, changing loss/weights, adjusting threshold, using augmentation, regularizing, or trying another model family. Change one major factor at a time when possible.

Feature importance and explanations are diagnostic aids, not proof of causality. Correlation, SHAP-style attributions, and attention weights do not establish what would happen under intervention.

---

## 11. Final test, deployment, and monitoring

After model and threshold decisions are complete:

1. Fit the chosen workflow on all permitted training data.
2. Evaluate once on the untouched test set, using the preselected metrics and slices.
3. Record the model, preprocessing, label mapping, threshold, data period, split strategy, and results.
4. Confirm inference uses the same transformations and feature definitions as training.
5. Check latency, resource needs, failure behavior, and input validation before release.
6. Monitor input distributions, missingness, data/label delay, prediction rates, calibration, and outcome metrics when labels arrive.
7. Define when to investigate, retrain, or roll back. Re-evaluate after material changes to data, population, or process.

A strong offline score is evidence about the tested sample and split. It is not a guarantee of future performance.

---

## 12. The reusable checklist

### Before modeling
- [ ] State the decision, prediction unit, target, prediction time, constraints, and error costs.
- [ ] Understand sampling, labels, feature meaning, units, and data provenance.
- [ ] Identify leakage, duplicates, entity/group structure, and time order.
- [ ] Choose a split that represents deployment.

### During modeling
- [ ] Inspect target balance/range and data quality.
- [ ] Fit all learned preprocessing on training data only.
- [ ] Start with a baseline and a simple model.
- [ ] Choose a metric that matches the decision; use validation/CV for selection.
- [ ] Log experiments and inspect train/validation gaps.

### Before trusting results
- [ ] Inspect errors, subgroup performance, calibration if needed, and uncertainty.
- [ ] Compare to baseline and confirm the test set stayed untouched.
- [ ] Verify production inputs and preprocessing match training.
- [ ] Plan monitoring and reassessment as data changes.

## The 12 questions to remember

1. What decision should the prediction support?
2. What does one example represent?
3. What exactly is the target, and how trustworthy is it?
4. What information exists at prediction time?
5. How were examples sampled, grouped, and ordered?
6. What are the feature meanings, types, units, and valid values?
7. What is missing, duplicated, imbalanced, or anomalous?
8. What split best simulates deployment?
9. What simple baseline should I beat?
10. Which metric matches the cost of errors?
11. Where does the model fail, and for whom?
12. Can I reproduce, deploy, and monitor this workflow?
