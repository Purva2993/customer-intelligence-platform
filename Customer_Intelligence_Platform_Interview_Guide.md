# Customer Intelligence Platform — Interview Guide
### Purwa Mugdiya | AI/ML Portfolio Project

---

# PART 1 — PROJECT STORY

## The One-Sentence Pitch

> "I built an end-to-end customer intelligence platform on 400K+ retail transactions that predicts which customers will churn, segments the customer base, recommends products, and forecasts revenue — all automated through an Airflow pipeline and tracked with MLflow."

---

## The Business Problem

The dataset is from a UK-based online retailer (UCI Online Retail, Dec 2010–Dec 2011). The business problems are universal:

- You have thousands of customers but don't know who is about to leave
- You don't know who your most valuable customers are
- You are wasting marketing budget treating all customers the same
- You have no visibility into future revenue

This project solves all four problems using ML.

---

## The 2-Minute Interview Script

> "I built a customer intelligence platform on the UCI Online Retail dataset — about 400K transactions from a UK retailer. The goal was to answer four business questions: who will churn, who are the most valuable customers, what should we recommend to each customer, and what will revenue look like next quarter.
>
> The most interesting technical challenge was churn prediction. My first model had data leakage — I was predicting churn using recency as a feature, but I had also defined churn based on recency. The model was just reading the answer. I caught it because the accuracy was suspiciously high. The fix was a proper time-based split: features from the first 9 months, churn labels from the last 3 months.
>
> After fixing that and adding 13 engineered features — things like quarterly purchase trends, whether the customer was active recently, log-transformed spend — I used XGBoost with GridSearchCV across 540 hyperparameter combinations and got 73–74% ROC-AUC, which is the honest ceiling for this dataset.
>
> I layered on K-Means segmentation to classify customers as Champions, Loyal, At-Risk, or Lost — and used that segmentation as a fallback for the recommendation engine when collaborative filtering didn't have enough history. Prophet handled revenue forecasting.
>
> Everything is wrapped in an Airflow DAG so it runs weekly — new data in, updated predictions out. Experiments are tracked in MLflow so every model version is reproducible."

---

## The 7 Modules — What, Why, and How

### Module 1 — Data Exploration (`01_exploration.ipynb`)

**What:** Profiled the raw data to understand shape, quality, and quirks before building anything.

**Key findings:**
- 541,909 transactions but ~25% had no CustomerID — had to be dropped
- Negative quantities represented product returns — not purchase behavior
- Huge outliers in spend (one customer spent £280K)
- Dataset spans Dec 2010 to Dec 2011 — exactly 12 months

**Interview point:** "Before building any model, I profiled the data. I found ~25% of records had no customer ID — keeping them would have introduced noise into every downstream model, so I excluded them early."

---

### Module 2 — Data Cleaning (`02_cleaning.ipynb`)

**What:** Removed cancellations (InvoiceNo starting with 'C'), negative quantities, missing CustomerIDs. Computed `TotalAmount = Quantity × UnitPrice`.

**Interview point:** "Clean data is the foundation. In retail data, returns are encoded as negative quantities. I removed them because a return is not a purchase behavior — including them would distort RFM features."

---

### Module 3 — Feature Engineering (`03_feature_engineering.ipynb`)

**What:** Transformed 400K raw transactions into 13 customer-level behavioral features.

**Core framework — RFM:**

| Feature | Meaning |
|---|---|
| Recency | Days since last purchase — lower = more engaged |
| Frequency | Number of unique orders — higher = more loyal |
| Monetary | Total spend — higher = more valuable |

**Enriched features added:**

| Feature | Why it matters |
|---|---|
| LogMonetary | Monetary is right-skewed — log transform normalizes it |
| LogAvgOrderValue | Average order size, normalized |
| IsOneTimeBuyer | One-time buyers churn at a much higher rate |
| PurchaseSpread | Days between first and last purchase — captures tenure |
| AvgDaysBetweenPurchases | Regularity of buying behavior |
| Q1/Q2/Q3_Orders | Quarterly order counts — captures purchase trend over time |
| PurchaseTrend | Q3 minus Q1 orders — is the customer buying more or less recently? |
| ActiveInQ3 | Did the customer buy in the last quarter of the feature window? |

**Interview point:** "Raw RFM with 3 features gave ~72% ROC-AUC. Adding 10 more behavioral features — especially quarterly trends and whether a customer was active recently — pushed it to ~74% and gave the model richer signal about customer trajectory."

---

### Module 4 — Churn Prediction (`04_churn_prediction.ipynb`)

**What:** XGBoost classifier that predicts whether a customer will churn.

**Key decisions:**

**1. Churn definition — time-based split (most important decision in the project)**
- Feature window: Dec 2010 → Sep 2011 (9 months of behavior)
- Observation window: Oct → Dec 2011 (3 months to see if they return)
- If a customer did NOT purchase in Oct–Dec 2011 → Churn = 1

**2. Model:** XGBoost with GridSearchCV (540 fits, 5-fold cross-validation)
- Best params: max_depth=3, n_estimators=200, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8

**3. Results:**
- Test ROC-AUC: 0.7347
- CV ROC-AUC: 0.7389 (very close — model is not overfitting)

**4. Feature importance:**
- Frequency: 37% — how often they buy is the strongest signal
- ActiveInQ3: 13% — whether they bought recently matters a lot
- Recency was surprisingly low (~5%) — because ActiveInQ3 already captures the "recent purchase" signal

**5. MLflow tracking:** Every experiment was logged — parameters, metrics, and the model artifact — so any result can always be reproduced.

---

### Module 5 — Customer Segmentation (`05_segmentation.ipynb`)

**What:** K-Means clustering to group customers into 4 meaningful segments.

**How K=4 was chosen:**
- Elbow method: inertia drops sharply up to K=4, then flattens
- Silhouette score: local peak at K=4 (0.37) — clusters are reasonably well-separated

**Segments:**

| Segment | Size | Profile |
|---|---|---|
| Champions | 28 | Recent, frequent, high spend — your VIPs |
| Loyal | 1,223 | Regular buyers, moderate spend — your stable base |
| At-Risk | 1,395 | Used to buy, now slipping — prime retention targets |
| Lost | 958 | Low recency, low frequency — likely gone |

**Interview point:** "Segmentation tells you WHO the churn model is targeting. An At-Risk customer and a Lost customer both have high churn probability — but the business response is different. An At-Risk customer is worth a retention offer; a Lost customer may need a win-back campaign or might not be worth the spend at all."

---

### Module 6 — Product Recommendations (`06_recommendations.ipynb`)

**What:** Item-based collaborative filtering — recommends products to each customer.

**How it works:**
1. Build a customer × product matrix (rows = customers, columns = products, values = purchase count)
2. Compute cosine similarity between all product pairs
3. For each customer, find products similar to what they have bought but have not tried yet
4. Two-layer hybrid: collaborative filtering for customers with enough history (2,162), segment-based fallback for sparse customers (1,442)

**Interview point:** "Collaborative filtering says 'customers who bought product A also bought product B.' The cold-start problem is that new or sparse customers don't have enough history. I solved this with a segment fallback — if we cannot personalize, we recommend the top products from the customer's segment."

---

### Module 7 — Revenue Forecasting (`07_revenue_forecast.ipynb`)

**What:** Facebook Prophet model that forecasts monthly revenue 3 months ahead.

**Key findings:**
- Revenue at risk from churned customers: £1,382,321 (22.5% of customer base)
- Forecast: Dec 2011 £1.04M → Jan £1.09M → Feb £1.14M → Mar £1.19M

**Why Prophet:** Handles trend decomposition automatically, interpretable, and does not require stationarity unlike ARIMA.

**Key choice:** Excluded Dec 2011 from training — only 9 days of data would have created a false revenue drop signal.

---

### Module 8 — Pipeline Automation (`dags/customer_intelligence_dag.py`)

**What:** Apache Airflow DAG that runs all 6 steps in sequence every week.

```
ingest → clean → features → train → segment → recommend/forecast
```

**Two delivery options:**
- `run_pipeline.py` — standalone runner, no Airflow needed, for demo/testing
- Airflow DAG — production-grade, scheduled, with dependency management

**Interview point:** "The pipeline means this is not a one-time analysis. If you drop new transaction data in, the whole system re-runs: features are recomputed, the churn model retrains, segments are refreshed, recommendations are updated. That is what makes it a platform, not just a notebook."

---

# PART 2 — CHALLENGES AND RESOLUTIONS

## Challenge 1 — Data Leakage (Most Important)

**Problem:** The first version defined churn as `Recency >= 90 days` — and Recency was also a feature. The model was essentially learning "if Recency >= 90, predict Churn = 1." That is not machine learning, it is just reading the answer.

**How it was caught:** ROC-AUC was suspiciously high (>0.85) with only 3 features. That was the red flag — a simple model should not perform that well.

**Fix:** Time-based train/test split. Features from the past (Dec 2010–Sep 2011), labels from the future (Oct–Dec 2011). No feature can contain information about the label.

**Interview value:** "This is data leakage — one of the most common and dangerous mistakes in ML. I caught it by questioning why a simple model was performing so well, which is counterintuitive but important. A model that appears to work and does not is worse than a model that is transparent about its limitations."

---

## Challenge 2 — ROC-AUC Ceiling at ~73%

**Problem:** The dataset has a natural ceiling — 12 months of data, ~4K customers. There is only so much predictive signal available.

**What was tried:**
- Basic RFM (3 features) → ~72% AUC
- Enriched features (13 features) → ~73% AUC
- GridSearchCV (540 hyperparameter combinations) → 73.47% AUC

**Resolution:** Accepted the honest ceiling. Updated the resume from an aspirational "89% ROC-AUC" to an honest "~74% ROC-AUC."

**Interview value:** "73–74% ROC-AUC on a dataset with this much noise and only 12 months of history is actually a good result. More importantly, the methodology is sound — proper time-based split, cross-validation, no leakage. A reported 89% with leakage is worse than an honest 73%."

---

## Challenge 3 — Skewed Distributions

**Problem:** Monetary and AvgOrderValue had extreme right skew — a few customers spending £50K–£280K. This distorts distance-based calculations and gives outliers too much weight.

**Fix:** Log transformation using `np.log1p()` — compresses extreme values while preserving ordering. Used `log1p` (log of 1 + x) instead of `log` to safely handle any zero values.

---

## Challenge 4 — Prophet Negative Forecasts

**Problem:** First attempt used logistic growth with a cap and floor. With only 12 months of data, Prophet had insufficient history to fit the growth curve — it oscillated and produced zero or negative revenue forecasts.

**Fix:** Switched to linear growth with `changepoint_prior_scale=0.05` (conservative trend changes) and `yearly_seasonality=False` (not enough data for a full seasonal cycle). Clipped forecast values at zero as a safety net.

---

## Challenge 5 — Pipeline Bug (Tuple Unpacking)

**Problem:** `features.py` had `for period, col in [3-element tuples]` — two variables trying to unpack three values. This crashed the pipeline silently during execution.

**Fix:** Rewrote the loop as `for start, end, col in quarters` — explicit variable names for all three elements.

---

# PART 3 — CONCEPTS DEEP DIVE

## XGBoost

### What is it?
XGBoost stands for Extreme Gradient Boosting. It is an ensemble method — it builds many decision trees sequentially, where each tree learns from the mistakes of the previous one.

### How Gradient Boosting Works

Think of it like a team of analysts:
1. Analyst 1 makes predictions — gets some right, some wrong
2. Analyst 2 focuses only on the cases Analyst 1 got wrong
3. Analyst 3 focuses on what Analyst 2 still got wrong
4. Repeat for N trees
5. Final answer = weighted vote of all analysts

Each tree corrects the residual error of the ensemble so far. The "gradient" part means it uses gradient descent to figure out which direction to correct.

### XGBoost vs Random Forest

| | Random Forest | XGBoost |
|---|---|---|
| How trees are built | In parallel, independently | Sequentially, each corrects the last |
| Overfitting risk | Moderate | Lower (has regularization) |
| Speed | Faster to train | Slower but more accurate |
| Industry standard for tabular data | Common | Dominant |

### Key Hyperparameters

| Parameter | What it controls | Our best value |
|---|---|---|
| `max_depth` | How deep each tree grows — deeper = more complex, more overfitting risk | 3 (shallow = generalizes better) |
| `n_estimators` | How many trees to build | 200 |
| `learning_rate` | How much each tree contributes — lower = slower but more robust | 0.05 |
| `subsample` | Fraction of training rows used per tree — adds randomness, reduces overfitting | 0.9 |
| `colsample_bytree` | Fraction of features used per tree — like Random Forest's feature sampling | 0.8 |

### What is ROC-AUC?

**ROC** = Receiver Operating Characteristic curve
**AUC** = Area Under that Curve

The ROC curve plots:
- X-axis: False Positive Rate — how often we wrongly predicted churn
- Y-axis: True Positive Rate — how often we correctly predicted churn

AUC = probability that the model ranks a randomly chosen churned customer higher than a randomly chosen active customer.

| AUC | Meaning |
|---|---|
| 0.5 | Random guessing — useless |
| 0.7 | Acceptable |
| 0.8 | Good |
| 0.9+ | Excellent |
| 1.0 | Perfect (suspicious — likely leakage) |

**Why AUC over accuracy?** AUC measures ranking quality across all thresholds — not just at the default 0.5 cutoff. A business might want to target the top 20% highest-risk customers, not everyone above 50% probability. AUC captures that.

### Interview Answer Template
> "I chose XGBoost because our data is structured and tabular — exactly the domain where gradient boosting outperforms neural networks. It also has built-in regularization which helps prevent overfitting on a relatively small dataset of 3,600 customers. Shallow trees (max_depth=3) outperformed deep trees (max_depth=7) because deep trees were memorizing training patterns. We achieved 73.47% ROC-AUC with proper cross-validation."

---

## MLflow

### What is it?
MLflow is an experiment tracking and model management tool. Think of it as a lab notebook for machine learning — every experiment is recorded and reproducible.

### What Problem Does it Solve?

Without MLflow:
- You run 10 experiments, forget which parameters gave the best result
- You cannot reproduce the best model months later
- There is no audit trail for model decisions

With MLflow, every run logs:
- Parameters (max_depth, learning_rate, etc.)
- Metrics (ROC-AUC, accuracy)
- The actual model artifact (the saved model file)

### Four Core MLflow Components

| Component | What it does | Used in this project? |
|---|---|---|
| Tracking | Records parameters, metrics, tags per run | Yes |
| Projects | Packages code for reproducible runs | No |
| Models | Standard format to save and load any ML model | Yes |
| Registry | Version control for models (staging → production) | No |

### Interview Answer Template
> "MLflow meant that every time I tried a new hyperparameter combination, I had a complete record — what I tried, what result I got, and the actual model artifact. After GridSearchCV ran 540 fits, I could look at the MLflow UI and immediately see which run produced the best AUC. If someone asks 'how did you get that model?' the answer is always traceable. That is production-grade practice."

---

## K-Means Clustering

### How it Works (Step by Step)

1. Choose K (number of clusters)
2. Randomly place K centroids (cluster centers)
3. Assign each customer to the nearest centroid
4. Move each centroid to the average position of its assigned customers
5. Repeat steps 3–4 until centroids stop moving

### How K=4 Was Chosen

**Elbow Method:**
- Train K-Means for K=2 through K=10
- Plot inertia (sum of squared distances from each point to its centroid)
- Inertia always drops as K increases — look for where the drop slows sharply (the "elbow")
- Our elbow was at K=4

**Silhouette Score:**
- Measures how similar a customer is to their own cluster vs. other clusters
- Range: -1 (wrong cluster) to +1 (perfect fit)
- Our score peaked at 0.37 at K=4

### Features Used for Clustering

We clustered on: Recency, Frequency, LogMonetary, PurchaseSpread

- Recency and Frequency: the two strongest behavioral signals
- LogMonetary: spend level — log-transformed to prevent outliers dominating
- PurchaseSpread: tenure signal — a customer active over 9 months is structurally different from one who appeared briefly

### Business Strategy per Segment

| Segment | Strategy |
|---|---|
| Champions (28) | Reward them, ask for referrals, give early access to new products |
| Loyal (1,223) | Upsell, cross-sell, enroll in loyalty program |
| At-Risk (1,395) | Send retention offer NOW — these are your biggest recoverable churn risk |
| Lost (958) | Win-back campaign or calculate if ROI justifies the spend |

### Interview Answer Template
> "I used two methods to validate K=4 rather than just picking arbitrarily. The elbow told me the inertia gains flatten after K=4 — more clusters give diminishing returns. The silhouette score confirmed the clusters are actually meaningful — customers within a segment are more similar to each other than to customers in other segments. The result is four segments with clear, distinct business profiles and different recommended actions."

---

## Collaborative Filtering

### What is it?
The idea: customers who bought similar things in the past will likely want similar things in the future.

We used item-based collaborative filtering (not user-based).

### Item-based vs User-based

| | User-based | Item-based |
|---|---|---|
| Logic | Find similar users, recommend what they liked | Find similar items, recommend items similar to what you have bought |
| Scales better? | No — comparing all user pairs is expensive | Yes — product catalog is smaller than user base |
| More stable? | No — user behavior changes over time | Yes — item similarity is more stable |

### How We Built It

1. Build a customer × product matrix — rows = customers, columns = products, values = purchase count
2. Compute cosine similarity between all product pairs — products often bought by the same customers score high
3. For each customer, find products similar to what they have bought but have not tried yet
4. Return top 5 recommendations

### Cosine Similarity

Cosine similarity measures the angle between two vectors:
- 0 = products have nothing in common
- 1 = products are bought by identical sets of customers

It is preferred over Euclidean distance here because purchase counts vary widely — a customer who bought 20 units of one product should not dominate over one who bought 1.

### The Cold-Start Problem and Our Fix

**Problem:** Customers with only 1 purchase do not have enough signal for meaningful item similarity.

**Fix:** Two-layer hybrid
- Layer 1: Collaborative filtering for 2,162 customers with sufficient history
- Layer 2: Segment-based fallback — top products in their segment — for 1,442 sparse customers

### Interview Answer Template
> "Collaborative filtering says 'customers who bought product A also bought product B.' The cold-start problem is that sparse users do not have enough history to find meaningful similar products. I solved this with a graceful fallback — if we cannot personalize, we at least recommend products that customers in the same behavioral segment tend to buy. It is less precise but still more relevant than random recommendations."

---

## Facebook Prophet

### What is it?
Prophet is a time series forecasting library from Meta. It decomposes a time series into components:

```
Revenue = Trend + Seasonality + Holidays + Error
```

### Why Prophet over ARIMA?

| | ARIMA | Prophet |
|---|---|---|
| Requires stationarity? | Yes | No |
| Handles missing data? | No | Yes |
| Multiple seasonalities? | Hard | Easy |
| Interpretable components? | Hard | Built-in |
| Minimum data needed | High | Moderate |
| Tuning complexity | High | Minimal |

### Key Decisions Made

**Excluded December 2011:** Only 9 days of data. Including it would make November → December look like a massive revenue crash, skewing the forecast downward.

**No yearly seasonality:** You need at least 2 full years to reliably fit a yearly seasonal cycle. With only 12 months, forcing yearly seasonality overfits noise.

**Linear growth over logistic:** Logistic growth requires knowing the theoretical maximum revenue (a cap). We had no business basis for that number. With 12 months of data, logistic growth produced oscillating nonsense. Linear growth was more honest.

**changepoint_prior_scale=0.05:** Conservative setting — tells Prophet not to overreact to short-term fluctuations and trend changes.

### Interview Answer Template
> "I chose Prophet because our data only spans 12 months — not enough for ARIMA to reliably estimate seasonal patterns. Prophet's linear trend model with a conservative changepoint prior was the right balance between capturing real trends and not overfitting 12 data points. The key data quality decision was excluding December 2011, which only had 9 days of data — including a partial month would have falsely signaled a revenue crash."

---

## Apache Airflow

### What is it?
Airflow is a workflow orchestration tool. You define pipelines as DAGs — Directed Acyclic Graphs — a set of tasks with defined dependencies between them.

### Key Concepts

| Concept | Meaning |
|---|---|
| DAG | Directed Acyclic Graph — the pipeline definition |
| Task | One unit of work (e.g., clean data, train model) |
| Operator | The type of task (PythonOperator, BashOperator, etc.) |
| Schedule | When the DAG runs (daily, weekly, on trigger) |
| Dependency | `t1 >> t2` means t2 runs only after t1 succeeds |

### TaskFlow API (What We Used)

The modern way to write Airflow — uses Python decorators instead of verbose operator configurations:

```python
@dag(schedule='@weekly')
def customer_intelligence_pipeline():

    @task
    def ingest(): ...

    @task
    def clean(): ...

    ingest() >> clean() >> features() >> train() >> segment() >> recommend()
```

### Why Airflow Over Just Running a Script?

| Scenario | Script | Airflow |
|---|---|---|
| One-time run | Fine | Overkill |
| Weekly automated runs | Fragile | Designed for this |
| Step fails halfway | Reruns everything from scratch | Reruns from the failed step only |
| Monitoring | None | Built-in UI, logs, alerts |
| Team visibility | None | Full audit trail |

### Interview Answer Template
> "Airflow is what separates a data science project from a data science product. A script you run manually is an experiment. A scheduled DAG that re-runs every week, retries on failure, and has a monitoring UI is a system. That distinction matters in industry — you want stakeholders to be able to trust that the model is always up to date, not relying on someone remembering to run a script."

---

# PART 4 — THE FORMULA FOR ANY TECHNICAL QUESTION

When an interviewer asks "how does X work?" use this structure:

1. **What problem does it solve?** (1 sentence)
2. **How it works** (2–3 sentences, no jargon)
3. **Why you chose it over alternatives** (1 sentence)
4. **What result you got** (1 sentence with a number)

### Example — XGBoost

> "XGBoost solves the problem of learning from structured tabular data. It builds trees sequentially where each tree corrects the errors of the previous one — like a team where each person focuses on what the last person got wrong. I chose it over Random Forest because it has better regularization for smaller datasets, and over neural networks because our data is tabular not image or text. We achieved 73.47% ROC-AUC with proper cross-validation."

### Example — K-Means

> "K-Means solves the problem of grouping customers into meaningful segments without needing labeled data. It repeatedly assigns each customer to the nearest cluster center and then moves those centers until they stabilize. I chose K=4 by validating with both the Elbow Method and Silhouette Score rather than guessing. The result was four business-interpretable segments — Champions, Loyal, At-Risk, and Lost — each with a different recommended action."

### Example — MLflow

> "MLflow solves the problem of reproducibility in ML experiments. Every time I train a model, it logs the parameters, metrics, and model artifact automatically. I used it because after running 540 hyperparameter combinations with GridSearchCV, I needed a reliable way to identify and reproduce the best run. Without it, experiment tracking would have been a mess of notes and renamed files."

---

# PART 5 — LIKELY INTERVIEW QUESTIONS AND ANSWERS

**Q: What is data leakage and did you encounter it?**
A: Data leakage is when information about the target variable is present in your features, causing the model to cheat rather than learn. Yes — my first churn model defined churn as Recency >= 90 days but also used Recency as a feature. The model just learned to read the label directly from the feature. I caught it because the AUC was suspiciously high. The fix was a time-based split — features from the past, labels from the future.

**Q: Why is your ROC-AUC only 73%? Can you improve it?**
A: 73–74% is the honest ceiling for this dataset — 12 months of data, 3,600 customers, 13 features. I tried basic RFM, enriched features, and GridSearchCV — each helped marginally. I could potentially improve it with more data, external features like seasonality or marketing events, or a more complex ensemble. But more importantly, the methodology is sound. A reported 89% with data leakage is worse than an honest 73%.

**Q: Why K-Means? Did you consider other clustering algorithms?**
A: K-Means works well for RFM data — the features are numerical and the clusters are roughly convex. Alternatives include DBSCAN (good for arbitrary shapes, but our clusters are not arbitrarily shaped) and hierarchical clustering (good for visualization, computationally expensive at scale). K-Means gave interpretable, stable clusters that map directly to business segments.

**Q: How do your recommendations handle new customers?**
A: New customers with no purchase history cannot use collaborative filtering — this is the cold-start problem. I handled it with a segment fallback: once a new customer is assigned to a segment based on initial behavior, they receive the top products for that segment. It is less personalized but better than random.

**Q: Why Airflow? Could you have just used a cron job?**
A: A cron job can schedule a script, but it has no retry logic, no monitoring UI, no dependency management between steps, and no audit trail. If step 3 fails, a cron job gives you no visibility. Airflow reruns from the failed step, logs the failure, and lets you inspect what went wrong. For a production ML pipeline, that observability is essential.

**Q: How would you deploy this to production?**
A: The pipeline is already modular — each step is a separate Python function. Deployment would involve: containerizing with Docker, deploying the Airflow DAG to a managed service like AWS MWAA or Google Cloud Composer, serving the churn model via a REST API using FastAPI or Flask, and storing outputs in a data warehouse like BigQuery or Redshift for downstream consumption by dashboards or email campaigns.
