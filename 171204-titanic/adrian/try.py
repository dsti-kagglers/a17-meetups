import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from dsti import transformers
from dsti.titanic import\
    get_title_from_name, get_rank_from_title, get_family_size, get_alphanumeric
# import dsti.transformers
# reload(dsti.transformers)
# import dsti.titanic
# reload(dsti.titanic)
pd.options.mode.chained_assignment = None

training_set = pd.read_csv('./input/train.csv', index_col='PassengerId')
training_target = training_set['Survived']

testing_set = pd.read_csv('./input/test.csv', index_col='PassengerId')

embarked_imputer = transformers.CategoricalImputer(
    column='Embarked', strategy='median')

title_extractor = transformers.FunctionExtractor(
    get_title_from_name, source_column='Name', result_column='Title')

rank_extractor = transformers.FunctionExtractor(
    get_rank_from_title, source_column='Title', result_column='Rank')

age_imputer = transformers.PivotTableImputer(
    values='Age', index=['Rank', 'Sex', 'Pclass'], aggfunc='median')

family_size_extractor = transformers.FunctionExtractor(
    get_family_size, result_column='FamilySize')

family_size_fate_extractor = transformers.CrossTableTransformer(
    index_column='FamilySize', result_column='FamilySizeFate')

ticket_transformer = transformers.FunctionExtractor(
    get_alphanumeric, source_column='Ticket', result_column='Ticket')

group_ticket_extractor = transformers.GroupByTransformer(
    by='Ticket', func='count', result_column='GroupTicket')

group_ticket_fate_extractor = transformers.CrossTableTransformer(
    index_column='GroupTicket', result_column='GroupTicketFate')

dummy_columns = ['Sex', 'Embarked', 'Rank', 'Pclass']
dummies_transformer = transformers.DummiesTransformer(
    columns=dummy_columns, prefix=dummy_columns)
features_dropper = transformers.DropColumnsTransformer(
    ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin',
     'Title', 'FamilySize', 'GroupTicket', 'Sex_male'] + dummy_columns)

pipeline = Pipeline([
    ('features', Pipeline([
        ('e', Pipeline([
            ('embarked', embarked_imputer)
        ])),
        ('t_r_a', Pipeline([
            ('title', title_extractor),
            ('rank', rank_extractor),
            ('age', age_imputer)
        ])),
        ('fs_fsf', Pipeline([
            ('family_size', family_size_extractor),
            ('family_size_fate', family_size_fate_extractor)
        ])),
        ('t_gt_gtf', Pipeline([
            ('ticket', ticket_transformer),
            ('group_ticket', group_ticket_extractor),
            ('group_ticket_fate', group_ticket_fate_extractor)
        ]))
    ])),
    ('d', Pipeline([
        ('dummies', dummies_transformer),
        ('drop', features_dropper)
    ]))
])

"""
pipeline = Pipeline([
    ('embarked', embarked_imputer),
    ('title', title_extractor),
    ('rank', rank_extractor),
    ('age', age_imputer),
    ('family_size', family_size_extractor),
    ('family_size_fate', family_size_fate_extractor),
    ('ticket', ticket_transformer),
    ('group_ticket', group_ticket_extractor),
    ('group_ticket_fate', group_ticket_fate_extractor),
    ('dummies', dummies_transformer),
    ('drop', features_dropper)
])
"""

pipeline.fit(training_set, training_target)
training_set = pipeline.transform(training_set)
training_set = training_set.T.drop_duplicates().T

X_train, X_test = train_test_split(
    training_set, test_size=0.90, random_state=42)
y_train = X_train['Survived']
y_test = X_test['Survived']

param_grid = dict()
classifier = GridSearchCV(SVC(), param_grid=param_grid)

classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
accuracy_score(y_test, y_predict)
