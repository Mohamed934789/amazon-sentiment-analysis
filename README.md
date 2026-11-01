# Review Rating Classification — learning baseline

An extracted, runnable TF-IDF/logistic-regression baseline for five-level review Score.
This repairs the original notebook's use-before-definition and missing scikit-learn dependency.
Python 3.12: install `requirements.txt`, then Run All in the notebook from this directory.
It defaults to clearly labelled synthetic reviews for an offline smoke path, not an Amazon benchmark.
Set `AMAZON_REVIEWS_CSV` to your authorized real Text/Score CSV for actual data analysis.
Duplicate review text cannot cross the split; conflicting labels are rejected. A single split does
not prove user/product independence or production accuracy. No real-data metric is fabricated.
Tests: install `requirements-dev.txt`, then `python -m pytest -q` here.
Earlier credential-shaped matches in historical notebook output were sklearn HTML false positives;
no key rotation or history rewrite is justified by them. This remains secondary NLP learning work.
