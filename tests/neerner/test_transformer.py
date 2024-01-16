import pandas as pd
from faker import Faker

from ml_template.models import transformers 
fake = Faker()


def test_clean_token_with_right_token_size() -> None:
    """Validate clean token transfomer is truncating tokens and keeping
    max available tokens"""

    num_records = 10
    max_words = 5

    # generate data
    data = []
    expected = []
    for _ in range(num_records):
        row = [fake.word() for _ in range(max_words + 5)]
        data.append(row)
        expected.append(max_words)

        row = [fake.word() for _ in range(max_words - 5)]
        data.append(row)
        expected.append(len(row))

    s = pd.Series(data=data)

    # train transformer
    t = CleanTokenTransformer(max_words_from_end=max_words)
    o = t.fit_transform(s)

    # validate
    assert len(o) == num_records * 2
    assert list(map(len, o.values)) == expected


def test_clean_token_removing_blank() -> None:
    max_words = 5
    data = [[fake.word() for _ in range(max_words)] + ["", "  "]]
    s = pd.Series(data=data)

    t = CleanTokenTransformer(max_words_from_end=5)
    o = t.fit_transform(s)

    assert len(o) == 1
    assert o.values[0][-1] not in ("", "  ")


def test_binary_token_value_frequency_transformer() -> None:
    data = [[fake.word() for _ in range(20)] for _ in range(10)]
    s = pd.Series(data=data)

    t = TokenValueDictTransformer(binary=True)
    o = t.fit_transform(s)
    assert len(o) == 10
    assert list(map(lambda x: x["isEmpty"], o.values)) == list(map(lambda _: 0, range(10)))


def test_binary_token_value_frequency_transformer_blank_document() -> None:
    data = [[""], ["", ""]]
    s = pd.Series(data=data)

    t = TokenValueDictTransformer(binary=True)
    o = t.fit_transform(s)
    assert len(o) == 2
    assert list(map(lambda x: x["isEmpty"], o.values)) == list(map(lambda _: 1, range(2)))
