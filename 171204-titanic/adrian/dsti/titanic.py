def get_title_from_name(X):
    return X.str.extract(' ([A-Za-z]+)\.', expand=False)


def get_alphanumeric(X):
    return X.str.replace('[\/\.\s]+', '')


def get_rank_from_title(X):
    title_rank_dictionary = dict([
        ("Capt", "Officer"),
        ("Col", "Officer"),
        ("Major", "Officer"),
        ("Jonkheer", "Royalty"),
        ("Don", "Royalty"),
        ("Sir", "Royalty"),
        ("Dr", "Officer"),
        ("Rev", "Officer"),
        ("Countess", "Royalty"),
        ("Dona", "Royalty"),
        ("Mme", "Mrs"),
        ("Mlle", "Miss"),
        ("Ms", "Mrs"),
        ("Mr", "Mr"),
        ("Mrs", "Mrs"),
        ("Miss", "Miss"),
        ("Master", "Master"),
        ("Lady", "Royalty")
    ])
    return X.apply(title_rank_dictionary.get)


def get_family_size(X):
    return X.SibSp + X.Parch + 1
