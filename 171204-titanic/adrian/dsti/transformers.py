from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer
import pandas as pd


class CategoricalImputer(TransformerMixin):
    def __init__(self, column, strategy='mean'):
        self.column = column
        self.strategy = strategy
        self.imputing_value = None

    def fit(self, X, y=None):
        vc = X[self.column].value_counts()
        if self.strategy == 'median':
            self.imputing_value = (vc - vc.median())\
                .abs().sort_values().index[0]
        elif self.strategy == 'mean':
            self.imputing_value = (vc - vc.mean())\
                .abs().sort_values().index[0]
        elif self.strategy == 'most_frequent':
            self.imputing_value = vc.sort_values(ascending=False).index[0]
        else:
            raise ValueError('Invalid strategy value: %s', self.strategy)
        return self

    def transform(self, X):
        X[self.column].fillna(self.imputing_value, inplace=True)
        return X


class PivotTableImputer(TransformerMixin):
    def __init__(self, values, index, aggfunc):
        self.values = values
        self.index = index
        self.aggfunc = aggfunc

    def _get_imputing_value(self, row):
        if pd.isnull(row.values.any()):
            return self.table[row[self.index]]
        else:
            return row.values.any()

    def fit(self, X, y=None):
        self.table = X.pivot_table(
            values=self.values, index=self.index, aggfunc=self.aggfunc)
        return self

    def transform(self, X):
        X[self.values] = X.apply(
            (lambda row: self._get_imputing_value(row)), axis=1)
        return X


class FunctionExtractor(TransformerMixin):
    def __init__(self, func, result_column,
                 source_column=None, validate=False):
        self.func = func
        self.source_column = source_column
        self.result_column = result_column
        self.validate = validate
        self.extractor = FunctionTransformer(self.func, validate=self.validate)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.source_column is not None:
            X[self.result_column] = self.extractor.transform(
                X[self.source_column])
        else:
            X[self.result_column] = self.extractor.transform(X)
        return X


class CrossTableTransformer(TransformerMixin):
    def __init__(self, index_column, result_column):
        self.index_column = index_column
        self.result_column = result_column

    def _get_transformed_value(self, row):
        if self.crosstable.loc[row, 1] == 0:
            return 0
        elif self.crosstable.loc[row, 1] <= self.crosstable.loc[row, 0]:
            return 1
        elif self.crosstable.loc[row, 0] == 0:
            return 3
        elif self.crosstable.loc[row, 1] > self.crosstable.loc[row, 0]:
            return 2
        else:
            return -1

    def fit(self, X, y):
        self.crosstable = pd.crosstab(index=X[self.index_column], columns=y)
        return self

    def transform(self, X):
        X[self.result_column] = X[self.index_column].apply(
            (lambda row: self._get_transformed_value(row)))
        return X


class GroupByTransformer(TransformerMixin):
    def __init__(self, by, func, result_column):
        self.by = by
        self.func = func
        self.result_column = result_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.result_column] = X.groupby(by=self.by)[self.by].\
            transform(func=self.func)
        return X


class DropColumnsTransformer(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.drop(self.columns, axis=1, inplace=True)
        return X


class DummiesTransformer(TransformerMixin):
    def __init__(self, columns, prefix=None):
        self.columns = columns
        self.prefix = prefix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dummies = pd.get_dummies(X, columns=self.columns, prefix=self.prefix)
        X = pd.concat([X, dummies], axis=1)
        return X


class DataFrameTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        return df
