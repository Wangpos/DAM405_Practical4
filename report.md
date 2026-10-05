# DAM405 Machine Learning Operations – Practical 4 Report

## Automated Training & Tuning with an Open-Source AutoML Tool (FLAML)

| | |
|---|---|
| **Module** | DAM405 – Machine Learning Operations |
| **Module tutor** | Kamal Acharya |
| **Student name** | [Your Name] |
| **Student ID** | [Your Student ID] |
| **Date** | 5 October 2026 |
| **Tool used** | FLAML (`flaml[automl]`), scikit-learn, LightGBM, XGBoost, pandas, joblib |
| **Environment** | Google Colab (Python 3.13) |
| **Notebook** | `DAM405 Practical 4 AutoML.ipynb` |

---

## 1. Introduction

Building a machine-learning model normally involves many repetitive decisions: which algorithm to try, which hyperparameters to set, and how long to keep tuning before stopping. Doing this by hand is slow and depends heavily on the experience of the person doing it. **Automated Machine Learning (AutoML)** tries to automate these repetitive parts. Given a dataset, a task, a metric and a time limit, an AutoML tool searches over many candidate models and hyperparameter settings and returns the best one it found.

In this practical I used **FLAML** (Fast and Lightweight AutoML), an open-source AutoML library developed by Microsoft. FLAML is designed to find good models cheaply. It starts with low-cost configurations and only moves to more expensive ones when they look promising. The aim was not just to "run AutoML". The practical also asked whether AutoML actually improves on a simple, hand-built model. For this reason the practical follows a disciplined MLOps workflow:

1. Train a simple **baseline** model first and record its score.
2. Run **AutoML** with a fixed time budget.
3. **Inspect** the best model AutoML found (algorithm, hyperparameters, validation score).
4. **Evaluate** both models on the same held-out test set and compare them fairly.
5. **Save** the AutoML model so it can be reused without repeating the search.

### Learning outcomes

- Use an open-source AutoML tool to train and tune a model on a given dataset.
- Compare an AutoML result against a simple baseline model.
- Inspect the best model, its hyperparameters and its score.
- Save the resulting model for later use.

### Dataset

The built-in **Breast Cancer Wisconsin (Diagnostic)** dataset from scikit-learn was used. It is a binary classification problem: each sample is a tumour described by 30 numeric features computed from a digitised image of a cell nucleus (radius, texture, smoothness, concavity, etc.). The target is whether the tumour is malignant or benign.

| Property | Value |
|---|---|
| Samples | 569 |
| Features | 30 (all numeric) |
| Classes | 2 (malignant / benign) |
| Training set | 455 samples (80%) |
| Test set | 114 samples (20%) |

---

## 2. What Was Done

The work was carried out in a Jupyter notebook on Google Colab. Each task in the practical handout corresponds to a section of the notebook.

### Task 1 – Install the tools and load the data

The required libraries were installed with `%pip install "flaml[automl]" scikit-learn joblib pandas`. The `[automl]` extra also installs LightGBM and XGBoost, which FLAML uses as candidate learners. The dataset was then loaded and split 80/20 into training and test sets.

I made two small, intentional improvements to the handout code:

- **`stratify=y`** was added to `train_test_split`. This keeps the proportion of malignant and benign cases the same in the training and test sets. With only 114 test samples, an unlucky split could otherwise distort the comparison.
- **A fixed random seed (`RANDOM_STATE = 42`)** was used for the split, the baseline and FLAML, so the experiment is as reproducible as possible.

### Task 2 – Build a baseline

A **Logistic Regression** model (`max_iter=2000`) was trained on the training set and evaluated on the test set. Besides accuracy, I also recorded the **ROC AUC** and the **training time** so the cost of AutoML could be compared with something concrete.

| Baseline metric | Result |
|---|---|
| Test accuracy | **0.9649** (4 wrong out of 114) |
| Test ROC AUC | 0.9954 |
| Training time | 0.57 s |

Logistic regression is already very strong on this dataset, which made it a demanding baseline for AutoML to beat.

### Task 3 – Run AutoML

FLAML was run with `task="classification"`, `metric="accuracy"` and `time_budget=60` seconds. From the FLAML log:

- FLAML used **cross-validation** on the training data only, minimising `1 − accuracy`. The test set was never seen during the search.
- It searched over **seven learners**: LightGBM (`lgbm`), Random Forest (`rf`), XGBoost (`xgboost`), Extra Trees (`extra_tree`), depth-limited XGBoost (`xgb_limitdepth`), SGD classifier (`sgd`) and L1-regularised logistic regression (`lrl1`).
- In 60 seconds it completed **188 trials**. Most were spent on the most promising learners: LightGBM (64 trials) and XGBoost (52), followed by Random Forest (27), Extra Trees (21) and SGD (21).
- The best model was found after only **18.2 seconds**. The remaining ~42 seconds did not find anything better.
- FLAML itself estimated that the "necessary" budget was about 40 s and a "sufficient" budget about 1,736 s (≈29 minutes).

### Task 4 – Inspect the best model

| Item | Value |
|---|---|
| **Best estimator** | LightGBM (`lgbm`) |
| `n_estimators` | 118 |
| `num_leaves` | 7 |
| `min_child_samples` | 11 |
| `learning_rate` | 0.514 |
| `log_max_bin` | 5 (i.e. `max_bin = 31`) |
| `colsample_bytree` | 0.978 |
| `reg_alpha` | 0.0149 |
| `reg_lambda` | 0.452 |
| **Best cross-validation accuracy** | **0.9780** |

The configuration describes a small gradient-boosted tree ensemble: shallow trees (7 leaves), a relatively high learning rate and light regularisation. The cross-validation score is computed as `1 − best_loss`, because FLAML stores the loss rather than the accuracy.

### Task 5 – Evaluate on the test set

Both models were evaluated on the same 114 held-out test samples:

| Model | Test accuracy | Errors | Test ROC AUC |
|---|---|---|---|
| Baseline (Logistic Regression) | **0.9649** | 4 / 114 | **0.9954** |
| FLAML AutoML (LightGBM, 60 s) | 0.9561 | 5 / 114 | 0.9947 |
| **Difference (AutoML − baseline)** | **−0.0088 (−0.88 pp)** | +1 error | −0.0007 |

**AutoML did not beat the baseline.** It made exactly one more mistake on the test set. Note that on 114 samples a single prediction is worth 0.88 percentage points, so the two models are practically equal.

### Task 6 – Save the best model

The full FLAML object was saved with `joblib.dump(automl, "automl_model.joblib")`. To check that the saved file actually works, it was reloaded with `joblib.load` and evaluated again. The reloaded model gave the same test accuracy (**0.9561**), confirming that the model can be reused later without repeating the search.

### Optional experiment – larger time budget and a different metric

To study how the time budget and the metric affect the result, FLAML was re-run with 120 s and 300 s budgets, and once with `metric="roc_auc"`:

| Run | Budget | Elapsed | Best estimator | Best CV score | Test accuracy | Test ROC AUC |
|---|---|---|---|---|---|---|
| Baseline (LogReg) | – | 0.6 s | logistic regression | – | **0.9649** | **0.9954** |
| AutoML, accuracy | 60 s | ≈60 s | lgbm | 0.9780 | 0.9561 | 0.9947 |
| AutoML, accuracy | 120 s | 120.1 s | xgboost | 0.9802 | **0.9649** | 0.9944 |
| AutoML, accuracy | 300 s | 300.1 s | xgboost | **0.9824** | 0.9474 | 0.9894 |
| AutoML, roc_auc | 60 s | 61.2 s | xgboost | 0.9977 (AUC) | **0.9649** | 0.9917 |

*(The "Best CV score" of the `roc_auc` run is a cross-validated ROC AUC, not an accuracy, so it cannot be compared directly with the other rows.)*

Key observations:

- **Cross-validation score improved steadily with more time** (0.9780 → 0.9802 → 0.9824), and the best learner changed from LightGBM to XGBoost.
- **Test accuracy did not follow.** The 120 s run matched the baseline (0.9649), but the 300 s run was the *worst* of all (0.9474, 6 errors).
- Changing the metric to ROC AUC produced a different model (XGBoost). It matched the baseline's accuracy but had a lower test ROC AUC than the baseline.
- Across every run, the simple logistic regression kept the **highest test ROC AUC (0.9954)** while costing well under a second to train.

---

## 3. What Was Learned

### 3.1 Answers to the reflection questions

**Q1. Did AutoML beat the baseline? By how much, and was the extra compute time worth it?**

No. With the required 60-second budget, AutoML (LightGBM) scored **0.9561** test accuracy against the baseline's **0.9649**, which is **0.88 percentage points lower** (one extra misclassification out of 114). With 120 s it only *tied* the baseline, and with 300 s it was worse. The baseline trained in about **0.6 seconds**, while each AutoML run took 60–300 seconds, roughly **100–500 times more compute** for no improvement. For this particular dataset the extra compute time was **not worth it**. The data is small, clean, numeric and close to linearly separable, which is exactly the kind of problem where logistic regression is already near the best achievable. AutoML would be more valuable on larger, messier datasets with non-linear patterns, where a simple model leaves more room for improvement.

**Q2. What did AutoML automate, and which decisions still required a human?**

*Automated by FLAML:*

- **Algorithm selection:** choosing between LightGBM, XGBoost, Random Forest, Extra Trees, SGD and L1 logistic regression.
- **Hyperparameter tuning:** e.g. number of trees, number of leaves, learning rate and regularisation strengths.
- **Search strategy and time allocation:** spending more trials on promising learners (LightGBM and XGBoost) and fewer on weak ones.
- **Validation:** running cross-validation internally to score each configuration.
- **Retraining** the best configuration on the full training set.

*Still required a human:*

- Choosing and understanding the **dataset** and the **prediction target**.
- Defining the **task type** (classification) and the **evaluation metric** (accuracy vs. ROC AUC). In a medical setting a metric such as recall may actually matter more, since a missed malignant tumour is worse than a false alarm.
- Setting the **time budget**, i.e. deciding how much compute to spend.
- Designing a **fair evaluation**: a held-out test set, stratification, fixed seeds and, most importantly, building a **baseline** to compare against.
- **Interpreting the results**: recognising that a higher CV score did not mean a better model, and deciding which model to actually deploy.

**Q3. How does increasing the time budget affect the result and the cost?**

Cost grows **linearly** with the budget. FLAML used almost exactly the time it was given (60 s, 120.1 s, 300.1 s). The *result*, however, did not improve reliably:

- The **cross-validation score rose** with more time (0.9780 → 0.9802 → 0.9824), because FLAML had more trials to find configurations that fit the validation folds better.
- The **test accuracy did not rise** (0.9561 → 0.9649 → 0.9474). The 300 s run, with the best CV score, had the worst test score.

This shows **diminishing returns** and a risk of **overfitting to the validation folds**: a longer search keeps finding configurations that look slightly better on cross-validation by chance, and these do not generalise to new data. On this small dataset, the differences are only one or two test samples, so they are within noise. In the 60 s run the best model was already found at 18.2 s, so most of the budget was spent without any gain. Larger budgets make sense for large or complex datasets, but here a short budget, or simply the baseline, was the more efficient choice.

### 3.2 Additional lessons

1. **Always build a baseline first.** Without the logistic-regression baseline, the AutoML score of 0.9561 would have looked impressive. The baseline showed that AutoML added no value on this problem.
2. **Validation score ≠ test score.** FLAML's best CV score (0.9780) was higher than its test score (0.9561). The CV score is the best of many attempts, so it is optimistically biased. Only a separate, untouched test set gives an honest estimate.
3. **Small test sets are noisy.** With 114 test samples, one prediction changes accuracy by 0.88 pp. Differences of one or two errors should not be treated as meaningful. A larger test set or repeated cross-validation would give more reliable comparisons.
4. **The choice of metric changes the chosen model.** Optimising ROC AUC instead of accuracy led FLAML to pick XGBoost instead of LightGBM. The metric must reflect what matters for the real application.
5. **Reproducibility needs effort.** Fixed seeds and a stratified split make the experiment repeatable. However, FLAML's search is time-based, so results can still differ slightly between machines (for example Colab vs. a laptop).
6. **Saving and reloading should be verified.** Reloading `automl_model.joblib` and getting the same accuracy (0.9561) confirmed that the saved artefact is usable. This is an important MLOps habit before handing a model to deployment.
7. **Warnings are worth reading.** The notebook printed `X does not have valid feature names, but LGBMClassifier was fitted with feature names`. This is harmless here, because FLAML internally converted the NumPy array to a DataFrame. In production, however, inconsistent input formats between training and prediction can cause real bugs.

---

## 4. Conclusion

In this practical I used the open-source AutoML library **FLAML** to automatically train and tune a classifier on the breast-cancer dataset, and compared it with a hand-built logistic-regression baseline. All tasks in the handout were completed: the data was loaded and split, a baseline was trained, FLAML was run with a 60-second budget, the best model (LightGBM) and its hyperparameters were inspected, both models were evaluated on the same test set, and the AutoML model was saved and successfully reloaded. The optional experiment with 120 s and 300 s budgets and the ROC AUC metric was also carried out.

The main finding is that **AutoML did not outperform the simple baseline** on this dataset. The baseline reached 0.9649 test accuracy and 0.9954 ROC AUC in under a second. FLAML's best 60-second model reached 0.9561 accuracy, and longer searches raised the cross-validation score without improving test performance. This does not mean AutoML is useless. FLAML explored 188 configurations across seven algorithms in one minute with almost no manual effort, which would take a person far longer. It does mean that AutoML is a tool, not a replacement for judgement. Its value depends on the problem, and its results must always be checked against a baseline on unseen data.

Overall, the practical showed that AutoML is good at automating algorithm selection and hyperparameter tuning. The human practitioner is still responsible for defining the problem, choosing the metric and the budget, designing a fair evaluation, and deciding whether the model is good enough to use.
