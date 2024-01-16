from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
import pandera as pa
from loguru import logger
from pandera.typing import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

from ml_templates.models import transformers


@dataclass
class MyEstimator(BaseEstimator):
    vec: DictVectorizer
    classifier: BaseEstimator
    tfidf_transformer: TfidfTransformer

    class InputFrame(pa.DataFrameModel):
        col1: Series[str] = pa.Field(
            description="column 1 containing text",
            nullable=False,
            coerce=True
        )
        
        col2: Series[int] = pa.Field(
            description="column2 c containing int",
            nullable=False,
            coerce=True
        )

    class LabelFrame(pa.DataFrameModel):
        label: Series[int] = pa.Field(description="binary label", coerce=True)

    def __init__(self, *args, **kwargs) -> None:
        """constructor
        """
        pass

    def add_features(self, x: DataFrame[InputFrame], fit: bool = False) -> None:
        """Generates features for the model
        WARNING: as a side effect it will add new columns to the input dataframe

        Args:
            x (DataFrame[InputFrame]): _description_
            fit (bool, optional): _description_. Defaults to False.

        """
        method = "transform" if not fit else "fit_transform"

        
        # apply transformers
        x["__t1"] = getattr(self.vec, method)(x["__token_dict"])
        x["__t2"] = getattr(self.tfidf_transformer, method)(x["__t1"])
        

    @pa.check_types
    def fit(
        self,
        x: DataFrame[InputFrame],
        y: DataFrame[LabelFrame],
    ) -> Self:
        """_summary_

        Args:
            x (DataFrame[InputFrame]): _description_
            y (DataFrame[LabelFrame]): _description_

        Returns:
            Self: _description_
        """
        x["__target"] = y.values

    
        # Initialize Transformers
        self.vec = DictVectorizer(sparse=False)
        self.tfidf_transformer = TfidfTransformer(smooth_idf=True)

        # Train Transformers
        self.add_features(x, fit=True)

    
        # Train classifier
        self.classifier = LogisticRegression()
        self.classifier.fit(x, y.values)

        return self

    def predict_proba(self, x: DataFrame[InputFrame]) -> pd.DataFrame:
        self.add_features(x, fit=False)
        output = self.classifier.predict_proba(x[...])
        return pd.DataFrame(output)

    def predict(self, x: DataFrame[InputFrame]) -> DataFrame[LabelFrame]:
        self.add_features(x)

        output = self.classifier.predict(x[...])
        return DataFrame[MyEstimator.LabelFrame](output)
