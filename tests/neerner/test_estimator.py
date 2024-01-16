import pandas as pd
from faker import Faker
from pandera.typing import DataFrame

from ml_template.models.estimator import MyEstimator

Faker.seed(11)
fake = Faker()


def test_functional_english_model() -> None:
    """A functional test of the estimator that it is working"""

    x = []
    y = []

    for _ in range(150):  # Adjust the number of strings as needed
        word_count = fake.random_int(min=0, max=5000)  # Generate 0-500 words
        words = fake.words(nb=word_count, ext_word_list=None)
        x.append({MyEstimator.InputFrame.post: "|".join(words)})
        y.append({MyEstimator.LabelFrame.label: fake.random_int(min=0, max=1)})

    xdf = DataFrame[MyEstimator.InputFrame](pd.DataFrame.from_records(x))
    ydf = DataFrame[MyEstimator.LabelFrame](pd.DataFrame.from_records(y))

    clf = MyEstimator(min_tokens_required=300, max_words_from_end=3000, use_smooth_embedding=False)
    clf.fit(xdf, ydf)
    _ = clf.predict(xdf)
    _ = clf.predict_proba(xdf)


def test_functional_brazil_model() -> None:
    """Brazil uses advanced data model"""

    x = []
    y = []

    for _ in range(150):  # Adjust the number of strings as needed
        word_count = fake.random_int(min=0, max=5000)  # Generate 0-500 words
        words = fake.words(nb=word_count, ext_word_list=None)
        x.append({MyEstimator.InputFrame.post: "|".join(words)})
        y.append({MyEstimator.LabelFrame.label: fake.random_int(min=0, max=1)})

    xdf = DataFrame[MyEstimator.InputFrame](pd.DataFrame.from_records(x))
    ydf = DataFrame[MyEstimator.LabelFrame](pd.DataFrame.from_records(y))

    clf = MyEstimator(min_tokens_required=0, max_words_from_end=10000, use_smooth_embedding=True)
    clf.fit(xdf, ydf)
    _ = clf.predict(xdf)
    _ = clf.predict_proba(xdf)
