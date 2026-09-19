\# 5G Network Quality Prediction \& Intelligent Handover System



A machine learning project for \*\*5G network quality prediction and intelligent handover analysis\*\* using UE measurement data, serving-cell information, neighboring-cell measurements, and actual handover events.



The project has two primary machine-learning objectives:



1\. \*\*5G Network Quality Prediction\*\* — predict future RSRP, RSRQ, and SINR from the current network state.

2\. \*\*Handover Prediction\*\* — predict whether a UE will experience a handover within the next \*\*3 seconds\*\*.



The eventual goal is to use these predictions, together with neighboring-cell conditions and decision rules, to build an \*\*intelligent handover decision system\*\*.



\---



\# 1. Project Objectives



\## A. 5G Network Quality Prediction



Given the current radio conditions of a UE, predict its network quality a few seconds into the future.



The current prediction targets are:



\* Future RSRP

\* Future RSRQ

\* Future SINR



\## B. Handover Prediction



Given the current serving-cell and neighboring-cell conditions, predict:



> \*\*Will this UE experience a handover within the next 3 seconds?\*\*



The future intelligent handover system will use this prediction together with neighbor-cell information to support target-cell selection and handover decisions.



\---



\# 2. Dataset



The project started with two types of raw TXT files.



\## Measurement Data



```text

neigh\_measurements\_1.txt

neigh\_measurements\_2.txt

...

neigh\_measurements\_6.txt

```



These contain periodic radio measurements from UEs.



\## Handover Data



```text

handover\_events\_1.txt

handover\_events\_2.txt

...

handover\_events\_6.txt

```



These contain actual handover events.



\### Dataset Statistics



| Data                       | Records |

| -------------------------- | ------: |

| Measurements               | 467,799 |

| Handover events            |   1,822 |

| Unique UEs in measurements |     187 |

| Unique UEs in handovers    |      76 |



The raw and processed datasets are \*\*not stored in this Git repository\*\* because they are large. The `data/` directory is excluded through `.gitignore`.



\---



\# 3. Understanding the Raw Data Structure



The raw TXT files contained deeply nested JSON-like structures. Before building the ML pipeline, the structure of the measurement and handover records was investigated.



The relevant measurement structure was identified as:



```text

UE

Timestamp

ServingCellInfo

&#x20;   ├── Serving Cell ID

&#x20;   ├── RSRP

&#x20;   ├── RSRQ

&#x20;   └── SINR



Neighbor Cell Info

&#x20;   ├── Neighbor 1

&#x20;   ├── Neighbor 2

&#x20;   └── Neighbor 3

```



A critical finding was that the actual serving-cell identifier was located at:



```text

ServingCellInfo → NrCgi → NrCellId

```



rather than relying on the top-level `CELL` field.



For neighboring cells, `NrCellId` was consistently `0` and therefore could not be used as a meaningful neighbor identifier.



Instead, the useful neighbor identifier was the:



```text

PCI / PhyCellId

```



This cell-identification work was important because incorrect serving/neighbor mapping would affect the entire downstream handover analysis.



\---



\# 4. Data Cleaning



The nested raw data was converted into flat tabular datasets.



The cleaned datasets are:



```text

measurements\_clean.csv

handovers\_clean.csv

```



The measurement dataset contains \*\*42 columns\*\*.



Important measurement fields include:



```text

serving\_cell

serving\_rsrp\_raw

serving\_rsrq\_raw

serving\_sinr\_raw

```



and neighbor fields such as:



```text

neighbor\_1\_pci

neighbor\_1\_rsrp\_raw

neighbor\_1\_rsrq\_raw

neighbor\_1\_sinr\_raw

...

```



The cleaned handover dataset contains \*\*16 columns\*\*.



\---



\# 5. Timestamp Investigation and Normalization



The dataset contained timestamps using different units.



We identified:



\* Measurement timestamp → Unix \*\*microseconds\*\*

\* TL publish timestamp → Unix \*\*microseconds\*\*

\* Publish timestamp → Unix \*\*nanoseconds\*\*



The timestamps were therefore normalized before performing temporal analysis.



The selected temporal references were:



```text

Measurement → measurement\_timestamp

Handover → tl\_publish\_timestamp

```



The approximate recording durations were:



```text

Measurement data → 51.42 hours

Handover data    → 51.17 hours

```



\---



\# 6. Radio Measurement Analysis



We investigated the RSRP, RSRQ, and SINR fields.



An important discovery was that the values were stored as \*\*encoded/raw values\*\*, rather than directly representing physical units such as:



```text

\-95 dBm

\-10 dB

15 dB

```



For example, RSRP values were approximately in the range 11–104, with some values occurring very frequently.



Because the necessary encoding metadata was unavailable, we deliberately did \*\*not\*\* assume a conversion such as:



```text

RSRP = raw value - 140

```



Instead, the raw values were retained consistently as model features.



\---



\# 7. Handover Analysis



The handover data was analyzed to understand:



\* Source cells

\* Target cells

\* Target PCI

\* Handover frequency

\* Source → target relationships



Observed transitions included examples such as:



```text

2 → 3

3 → 2

3 → 5

4 → 5

5 → 4

5 → 3

```



This established the actual handover event structure before attempting to build prediction models.



\---



\# 8. Measurement–Handover Association



For each handover, we searched for the \*\*latest measurement from the same UE occurring before the handover event\*\*.



Conceptually:



```text

Measurement

&#x20;     ↓

Same UE

&#x20;     ↓

Measurement time < handover time

&#x20;     ↓

Latest available measurement

&#x20;     ↓

Handover event

```



The resulting associations were stored in:



```text

handover\_associations.csv

```



\### Association Results



We successfully associated:



```text

1,822 / 1,822 handovers

```



with a preceding measurement from the same UE.



We then checked whether the actual handover target appeared among the neighboring cells recorded in that measurement.



It appeared in:



```text

1,820 / 1,822 handovers

= 99.89%

```



This provided strong validation for the neighbor PCI interpretation.



The median time between the selected measurement and the handover was approximately:



```text

0.488 seconds

```



\---



\# 9. Prediction Horizon Selection



The project predicts handover events into the future.



Measurement intervals were investigated first.



The typical measurement interval was approximately:



```text

1 second

```



Future-measurement coverage was checked for several horizons:



| Horizon       |   Coverage |

| ------------- | ---------: |

| 1 second      |     99.89% |

| 2 seconds     |     99.84% |

| \*\*3 seconds\*\* | \*\*99.79%\*\* |

| 4 seconds     |     99.75% |

| 5 seconds     |     99.70% |



Based on this analysis, the primary prediction horizon was selected as:



> \*\*3 seconds\*\*



\---



\# 10. Machine-Learning Targets



The target-generation process produces:



```text

future\_rsrp\_raw

future\_rsrq\_raw

future\_sinr\_raw

```



and:



```text

handover\_within\_3s

```



Each measurement row therefore represents:



```text

CURRENT STATE

&#x20;   │

&#x20;   ├── Serving-cell quality

&#x20;   ├── Neighbor-cell conditions

&#x20;   └── Current network state

&#x20;           │

&#x20;           ▼

&#x20;      ML prediction

&#x20;           │

&#x20;           ▼

FUTURE STATE / EVENT

&#x20;   │

&#x20;   ├── Future RSRP

&#x20;   ├── Future RSRQ

&#x20;   ├── Future SINR

&#x20;   └── Handover within 3 seconds?

```



The resulting target dataset contains:



```text

467,799 samples

```



There are:



```text

4,619 positive handover-prediction rows

```



These \*\*4,619 positive rows are not 4,619 actual handovers\*\*.



A single real handover can generate several positive measurement rows because every measurement occurring within the 3-second prediction window may receive a positive label.



\---



\# 11. Chronological Train / Validation / Test Split



The dataset was \*\*not randomly split\*\*.



Because this is temporal network data, a random split could allow information from future conditions to influence training or model selection.



The recordings contained large gaps, so a simple 70/15/15 timestamp split was not appropriate.



Seven recording segments were identified and divided chronologically:



```text

TRAIN

Segment 1 + Segment 2



VALIDATION

Segment 3



TEST

Segment 4 + Segment 5 + Segment 6 + Segment 7

```



A small \*\*3-second purge\*\* was also applied around split boundaries to reduce temporal leakage.



\### Final Split



| Dataset    |    Rows | Positive Handover Rows |

| ---------- | ------: | ---------------------: |

| Train      | 413,632 |                    503 |

| Validation |  19,540 |                  1,962 |

| Test       |  34,594 |                  2,154 |



The \*\*test dataset has not been used for model training or model selection\*\*.



This distinction must be preserved during the remaining evaluation work.



\---



\# 12. Feature Engineering



The current network state was converted into ML features.



\## Serving-cell features



```text

serving\_cell

serving\_rsrp

serving\_rsrq

serving\_sinr

```



\## Neighbor-cell features



For up to three neighbors:



```text

neighbor\_1\_pci

neighbor\_1\_rsrp

neighbor\_1\_rsrq

neighbor\_1\_sinr

...

```



\## Neighbor-count features



```text

number\_of\_neighbors

has\_neighbor

```



\## Relative / Delta Features



Relative features are particularly important for handover prediction.



For example:



```text

neighbor RSRP - serving RSRP

```



This allows the model to capture whether a neighboring cell is stronger or weaker than the current serving cell.



Additional engineered information includes:



\* Best neighbor

\* Best-neighbor PCI

\* Best-neighbor radio metrics

\* Maximum neighbor RSRP

\* Minimum neighbor RSRP

\* Mean neighbor measurements

\* Neighbor ranges

\* Neighbor-versus-serving differences



The final feature dataset contains:



```text

71 columns

```



with approximately:



```text

57 predictor features

```



after excluding targets and metadata.



\---



\# 13. Handover Baseline — Logistic Regression



Before using a stronger model, a Logistic Regression baseline was implemented.



The baseline uses current-state information and does \*\*not\*\* use:



\* Future measurements

\* Future handover information

\* Future target cell

\* Future target PCI



Validation results:



| Metric    | Logistic Regression |

| --------- | ------------------: |

| Precision |               0.167 |

| Recall    |               0.981 |

| F1        |               0.286 |

| PR-AUC    |               0.294 |

| ROC-AUC   |               0.809 |



The baseline achieves very high recall but produces many false alarms.



This is expected to some extent because the handover-prediction problem is highly imbalanced.



\---



\# 14. Future-Quality Baseline — Persistence



For future network-quality prediction, a simple persistence baseline was created.



The assumption is:



```text

future RSRP ≈ current RSRP

future RSRQ ≈ current RSRQ

future SINR ≈ current SINR

```



Validation results:



| Metric |  RSRP |  RSRQ |  SINR |

| ------ | ----: | ----: | ----: |

| MAE    | 2.980 | 1.180 | 6.453 |

| RMSE   | 4.205 | 2.305 | 9.820 |

| R²     | 0.749 | 0.102 | 0.652 |



This provides a meaningful baseline for evaluating ML-based future-quality prediction.



\---



\# 15. XGBoost — Future Quality Prediction



XGBoost models were trained for:



```text

Future RSRP

Future RSRQ

Future SINR

```



Validation R² values were:



| Target | Persistence R² | XGBoost R² |

| ------ | -------------: | ---------: |

| RSRP   |          0.749 |  \*\*0.754\*\* |

| RSRQ   |          0.102 |  \*\*0.188\*\* |

| SINR   |          0.652 |  \*\*0.677\*\* |



XGBoost provides improvements over the persistence baseline, particularly for:



\* RSRQ

\* SINR



The improvement is relatively modest, indicating that future radio conditions contain a strong persistence component and that additional temporal/contextual features may be required for larger gains.



\---



\# 16. XGBoost — Handover Prediction



An XGBoost classifier was trained for the 3-second handover-prediction task.



The training data was highly imbalanced:



```text

Negative = 413,129

Positive = 503

```



Class imbalance was handled using approximately:



```text

scale\_pos\_weight = 821.33

```



Validation results:



| Metric          | Logistic Regression |   XGBoost |

| --------------- | ------------------: | --------: |

| Precision       |               0.167 | \*\*0.269\*\* |

| Recall          |           \*\*0.981\*\* |     0.811 |

| F1              |               0.286 | \*\*0.404\*\* |

| PR-AUC          |               0.294 | \*\*0.363\*\* |

| ROC-AUC         |               0.809 | \*\*0.856\*\* |

| False positives |               9,583 | \*\*4,321\*\* |

| False negatives |                  37 |       370 |



The models have different operating characteristics.



\### Logistic Regression



```text

Very high recall

&#x20;      ↓

Catches almost all positive cases

&#x20;      ↓

Produces many false alarms

```



\### XGBoost



```text

Better precision / discrimination

&#x20;      ↓

Fewer false alarms

&#x20;      ↓

Misses more positive cases

```



Therefore, model selection should not be based on a single metric. Threshold analysis and application-specific evaluation are required before designing the final handover decision logic.



\---



\# 17. Completed Pipeline



The work completed so far can be summarized as:



```text

RAW 5G TXT FILES

&#x20;       │

&#x20;       ▼

Understand nested JSON structure

&#x20;       │

&#x20;       ▼

Clean measurements + handovers

&#x20;       │

&#x20;       ▼

Normalize timestamps

&#x20;       │

&#x20;       ▼

Analyze radio measurements

&#x20;       │

&#x20;       ▼

Analyze handover events

&#x20;       │

&#x20;       ▼

Associate measurements with handovers

&#x20;       │

&#x20;       ▼

Select 3-second prediction horizon

&#x20;       │

&#x20;       ▼

Create ML targets

&#x20;       │

&#x20;       ▼

Chronological train / validation / test split

&#x20;       │

&#x20;       ▼

Feature engineering

&#x20;       │

&#x20;       ▼

Logistic Regression baseline

&#x20;       │

&#x20;       ▼

Persistence future-quality baseline

&#x20;       │

&#x20;       ▼

XGBoost future-quality models

&#x20;       │

&#x20;       ▼

XGBoost handover model

&#x20;       │

&#x20;       ▼

&#x20;         CURRENT END

```



\---



\# 18. Remaining Work



The remaining work begins with \*\*model evaluation and continues toward the intelligent handover system\*\*.



\## Step 13 — Model Evaluation



Perform detailed evaluation of the trained models.



Possible evaluation tasks include:



\* Confusion matrices

\* Precision / Recall analysis

\* PR curves

\* ROC curves

\* Threshold analysis

\* Future-quality prediction errors

\* Per-target error analysis

\* Validation-versus-test comparison

\* Final test-set evaluation



The test set must remain untouched until the modeling and threshold decisions are finalized.



\---



\## Step 14 — Intelligent Handover Decision Engine



The ML predictions should eventually be combined with current network information and decision rules.



Conceptually:



```text

Predicted Future Quality

&#x20;         +

Handover Probability

&#x20;         +

Neighbor Conditions

&#x20;         +

Decision Rules / Constraints

&#x20;         │

&#x20;         ▼

Target Neighbor Selection

&#x20;         │

&#x20;         ▼

Handover Decision

```



This stage is different from simply predicting whether a handover will occur. The objective is to develop a system that can support an \*\*intelligent handover decision\*\*.



\---



\## Step 15 — Final Test Evaluation



Once the model, features, and decision thresholds have been finalized:



```text

TRAIN

&#x20; ↓

Model development

&#x20; ↓

VALIDATION

&#x20; ↓

Feature / model / threshold decisions

&#x20; ↓

TEST

&#x20; ↓

Final unbiased evaluation

```



The test set should only be used for the final evaluation.



\---



\## Step 16 — Dashboard / Application



A dashboard or application can eventually display:



```text

Current serving cell

&#x20;       │

Current RSRP / RSRQ / SINR

&#x20;       │

Neighbor cells

&#x20;       │

Predicted future quality

&#x20;       │

Handover probability

&#x20;       │

Recommended target

&#x20;       │

Handover decision

```



The `app/` directory is currently reserved for this stage.



\---



\## Step 17 — Final Deliverables



The final project should include:



\* Model evaluation

\* Graphs and visualizations

\* Results and reports

\* Intelligent handover decision logic

\* Dashboard / demo

\* Final project report

\* Final presentation



\---



\# 19. Repository Structure



```text

5G\_Network\_Quality/

│

├── .gitignore

├── README.md

├── requirements.txt

│

├── app/

│

├── data/

│   ├── raw/

│   │   ├── handovers/

│   │   └── measurements/

│   ├── processed/

│   ├── train/

│   ├── validation/

│   └── test/

│

├── models/

│

├── notebooks/

│

├── results/

│   ├── figures/

│   ├── predictions/

│   └── reports/

│

└── src/

&#x20;   ├── analysis/

&#x20;   ├── data\_processing/

&#x20;   ├── evaluation/

&#x20;   ├── feature\_engineering/

&#x20;   ├── handover/

&#x20;   ├── models/

&#x20;   └── target\_creation/

```



The `data/` directory exists in the local project but is excluded from GitHub because of its size.



\---



\# 20. Installation



The project uses \*\*Python 3.11\*\*.



Install the project dependencies using:



```bash

pip install -r requirements.txt

```



Current primary dependencies:



```text

numpy

pandas

scikit-learn

xgboost

```



\---



\# 21. Collaboration Notes



This repository is intended for the project team.



Before starting work:



```bash

git pull

```



For independent development, create a branch:



```bash

git checkout -b feature-name

```



After making changes:



```bash

git add .

git commit -m "Describe your changes"

git push -u origin feature-name

```



Then create a Pull Request on GitHub for review and merging into `main`.



\### Important



Do not commit:



\* The large dataset

\* Passwords

\* API keys

\* Tokens

\* `.env` files

\* Virtual environments

\* Large generated files



The dataset should be shared separately with project members.



\---



\# 22. Current Project Status



\### Completed



\* Raw dataset investigation

\* Nested JSON structure analysis

\* Serving-cell identification

\* Neighbor-cell identification

\* Measurement cleaning

\* Handover cleaning

\* Timestamp investigation

\* Timestamp normalization

\* Radio measurement analysis

\* Handover analysis

\* Measurement–handover association

\* 3-second prediction-horizon selection

\* ML target creation

\* Chronological train/validation/test splitting

\* Temporal leakage protection through split-boundary purge

\* Feature engineering

\* Logistic Regression handover baseline

\* Persistence future-quality baseline

\* XGBoost future-quality models

\* XGBoost handover-prediction model

\* Initial validation results



\### Current Handoff Point



> \*\*Model development is complete up to the initial XGBoost models.\*\*



\### Remaining



\* Detailed model evaluation

\* Threshold analysis

\* Final model/threshold decisions

\* Intelligent handover decision engine

\* Final untouched test evaluation

\* Dashboard/application

\* Final visualizations

\* Final reports

\* Final presentation



