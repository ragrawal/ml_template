from collections import Counter
from typing import Any, Self

import pandera as pa
import pandera.typing as pat
from pandera.typing import DataFrame, Series
from sklearn.base import TransformerMixin


class MyTransformer(TransformerMixin):
    pass