Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

# Environment Set up
* **Option 1: Using pip and venv (Recommended)**
    * Ensure you have Python 3.13 installed
    * Create virtual environment: `python3.13 -m venv .venv`
    * Activate environment: `source .venv/bin/activate` (On Windows: `.venv\Scripts\activate`)
    * Install dependencies: `pip install -r starter/requirements.txt`

* **Option 2: Using conda**
    * Download and install conda if you don't have it already.
    * conda create -n [envname] "python=3.13" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn pydantic httpx matplotlib seaborn -c conda-forge
    * Install git either through conda ("conda install git") or through your CLI, e.g. sudo apt-get git.

## Repositories
* Create a directory for the project and initialize git.
    * As you work on the code, continually commit changes. Trained models you want to use in production must be committed to GitHub.
* Connect your local git repo to GitHub.
* Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
    * Make sure you set up the GitHub Action to use Python 3.13 (same version as development).
    * Note: Add flake8 to requirements.txt if you want to use it for linting: `pip install flake8`

# Data
* Download census.csv and commit it to dvc.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.

# Model
* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
    * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.

# API Creation
*  Create a RESTful API using FastAPI this must implement:
    * GET on the root giving a welcome message.
    * POST that does model inference.
    * Type hinting must be used.
    * Use a Pydantic model to ingest the body from POST. This model should contain an example.
   	 * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

# API Deployment
* Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).
* Create a new app and have it deployed from your GitHub repository.
    * Enable automatic deployments that only deploy if your continuous integration passes.
    * Hint: think about how paths will differ in your local environment vs. on Heroku.
    * Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
    * Note: Install flake8 separately if needed: `pip install flake8`
* Write a script that uses the requests module to do one POST on your live API.

# Model Card

## Model Details

This project uses a supervised binary classification model to predict whether a person's income is `<=50K` or `>50K` from census-style demographic and employment features. The model was created by Shivam Sharma for the Udacity ML DevOps Engineer Nanodegree final project.

- Model type: `RandomForestClassifier`
- Library: scikit-learn
- Number of trees: `100`
- Random seed: `42`
- Parallelism: `n_jobs=-1`
- Cross-validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Training script: `starter/starter/train_model.py`
- Supporting preprocessing: `starter/starter/ml/data.py`

The preprocessing pipeline strips whitespace from CSV column names, one-hot encodes categorical features with `OneHotEncoder(handle_unknown="ignore")`, and binarizes the `salary` target with `LabelBinarizer`.

Reference: https://arxiv.org/pdf/1810.03993.pdf

## Intended Use

This model is intended for educational use as part of an end-to-end MLOps workflow covering training, evaluation, slice-based performance analysis, CI, and deployment. Intended users are Udacity reviewers, students, and developers evaluating the project pipeline.

This model is not intended for production decision-making in hiring, compensation, lending, insurance, immigration, or any other high-stakes domain.

## Data

The model is trained on the census income dataset stored in `starter/data/census.csv`. The label is `salary`, which is converted into a binary outcome representing income `<=50K` or `>50K`.

Training and evaluation use an 80/20 train-test split from the same dataset. The pipeline uses the following categorical features for one-hot encoding:

- `workclass`
- `education`
- `marital-status`
- `occupation`
- `relationship`
- `race`
- `sex`
- `native-country`

Continuous features such as age, fnlgt, education-num, capital-gain, capital-loss, and hours-per-week are passed through without scaling.

## Metrics

### Overall Performance

| Metric | Value |
| --- | ---: |
| Mean CV F1 | 0.6649 |
| CV F1 standard deviation | 0.0057 |
| Test precision | 0.7391 |
| Test recall | 0.6384 |
| Test F1 | 0.6851 |

![Overall model performance](starter/screenshots/overall_metrics.png)

### Key Slice Performance

The project computes slice metrics across categorical feature values. Selected examples are shown below.

| Feature | Slice | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| sex | Female | 0.726 | 0.511 | 0.599 |
| sex | Male | 0.741 | 0.661 | 0.699 |
| race | White | 0.737 | 0.637 | 0.683 |
| race | Black | 0.741 | 0.615 | 0.672 |
| race | Asian-Pac-Islander | 0.786 | 0.710 | 0.746 |
| education | HS-grad | 0.646 | 0.423 | 0.511 |
| education | Bachelors | 0.757 | 0.733 | 0.745 |
| education | Masters | 0.826 | 0.850 | 0.838 |

![Selected slice F1 scores](starter/screenshots/slice_metrics.png)

## Bias and Fairness Considerations

The model uses sensitive or demographic-adjacent attributes such as `sex`, `race`, `relationship`, and `native-country`. These features can encode or proxy protected characteristics and structural inequities present in the source data.

The slice metrics show uneven performance across groups. For example, the F1 score for `sex=Female` is lower than for `sex=Male`, and some small subgroups show unstable or extreme metric values because they contain very few examples. Those differences indicate that the model can reflect data imbalance and historical bias rather than neutral income prediction.

## Caveats

- The test set is drawn from the same source dataset as the training set, so this is not an external validation.
- Slice metrics for rare categories are noisy and can look artificially perfect or artificially poor because the support is very small.
- The model is appropriate for coursework and demonstration, not for real-world policy or business decisions.
- If the dataset, preprocessing, or model parameters change, the reported metrics and plots should be regenerated.
