# Progress Presentation Outline
**Project**: Predictive Maintenance for U.S. Bridge Infrastructure
**Date**: April 7
**Time Frame**: 5-Minute Presentation + 3-Minute Q&A

> [!TIP]
> **Presentation Strategy**  
> Since you have 5 minutes max, you must focus on the *novelty* of your dataset size (5+ million records) and your targeted approach to the class imbalance (Predicting "Critical" bridges before they fail). Keep pacing brisk—roughly 45-50 seconds per slide. All four team members (Akash, Pramod, Shruthi, Shriram) must speak.

---

## Slide 1: Problem & Dataset (Speaker: Shriram)
**Title:** The Challenge of Aging Infrastructure
- **The Problem**: There are ~600,000 public bridges in the U.S. Traditional maintenance is reactive. We propose a proactive, predictive model to identify bridges that are likely to degrade to a "Critical" state.
- **The Dataset**: We collected a massive **5+ Million Record Dataset** from the FHWA National Bridge Inventory (NBI), aggregating data from 2014–2025.
- **Why It's Unique**: Leveraging over a decade of data across all 50 states gives our models unprecedented temporal and geographic robustness compared to standard single-year models.

## Slide 2: EDA Findings & Challenges (Speaker: Pramod)
**Title:** Decoding the NBI Data
- **Class Imbalance**: Show a bar chart illustrating the extreme imbalance. Our target labels (0-9) are mapped to 4 severity classes. The "Critical (0-3)" class represents a fraction of a percent of the data but is the most important to predict.
- **Feature Relationships**: Briefly mention significant predictors discovered during EDA (e.g., Bridge Age vs. Scour Rating or Traffic Volume).
- **Data Reality**: Mention handling missing values using median/mode imputation due to inconsistencies across state reporting standards over the years.

## Slide 3: Model Plan & Innovation (Speaker: Shruthi)
**Title:** A Three-Pillar Modeling Strategy
- **Baselines**: Implementing Logistic Regression & k-NN to establish performance floors.
- **Tree-Based Models (Our Core)**: XGBoost & Random Forests to handle nonlinear interactions and complex structured tabular data effectively.
- **Margin Systems**: SVMs with RBF kernels to detect difficult decision boundaries between "Poor" and "Fair" conditions.
- **The Innovation**: Using **SMOTETomek** to synthesize minority "Critical" samples and applying Class Probability Calibration (Platt Scaling) so the models output reliable failure *probabilities*, not just hard classifications.

## Slide 4: Preliminary Results (Speaker: Akash)
**Title:** Baselines vs Initial Trees
- **Current Standing**: We have successfully loaded and processed the 5M dataset and run initial baseline models.
- **Metric Selection**: We are optimizing for **Macro F1** and **ROC-AUC (OvR)** instead of accuracy, because a 99% accuracy model predicting all bridges as "Good" is useless for safety.
- **Early Wins**: Share a quick mock/early metric (e.g., "XGBoost is already capturing a 65% recall on the 'Poor' class compared to 12% on Logistic Regression.")

## Slide 5: Next Steps (Speaker: Akash)
**Title:** Bridging the Gap to the Final Milestone
- **Interpretability Integration**: Implementing SHAP (SHapley Additive exPlanations) to provide structural engineers with feature-by-feature reasons *why* a specific bridge is flagged as Critical.
- **Hyperparameter Tuning**: Scaling our pipeline using PySpark or distributed CV due to the sheer size of the 5M dataset.
- **Final Evaluation**: Conducting deep error analysis focused on costly misclassifications (e.g., when the model predicts "Good" but the bridge is actually "Critical").

## Slide 6: Q&A / Thank You (All Team Members)
**Title:** Questions?
- Leave this slide up during the 3-minute Q&A.
- Include a small footer with team names: *Akashkumar S., Pramod S.D., Shriram D., Shruthi T.*

---

## Preparation Checklist for Tomorrow
- [ ] Practice the hand-offs between speakers to ensure you stay under the 5-minute strict limit.
- [ ] Dr. Masum is looking for *Novelty*. During Q&A, strongly emphasize your handling of class-imbalance and your 10-year, 5M+ row aggregated dataset approach.
- [ ] Ensure everyone is ready to answer questions about their specific component (e.g., Akash for Tree Models/Calibration, Shruthi for SVM/Baselines).
