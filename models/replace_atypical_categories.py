def replace_atypical_categories(df_train, df_test, columnname, pct = .01, base_df = "test"):
    """ Replace all categories in a categorical variable whenever the number of 
    observations in the test or train data set is lower than pct percetage of the
    total number of observations.
    The replaced categories are assigned to "other" category. 

    Input:
    ----------
    df_train: train DataFrame
    df_test: test DataFrame
    columnname:  name of the categorical variable whose categories will
    be replaced.
    pct: percetage of number of the observations that will be required for a categorical
    not to be labeled as atypical.
    base_df: The base DataFrame in which the analysis will be done.

    Output:
    ----------
    df_train and df_test with the columnname variable with the new labels.

    """

    if base_df == "test":
        limmit  = len(df_test) *pct
        vc = df_test[columnname].value_counts()
    else:
        limmit  = len(df_train) *pct
        vc = df_train[columnname].value_counts()
    
    common = vc > limit
    common = set(common.index[common].values)
    print("Set", sum(vc <= limit), columnname, "categories to 'other';", end=" ")
    
    df_train.loc[df_train[columnname].map(lambda x: x not in common), columnname] = 'other'
    df_test.loc[df_test[columnname].map(lambda x: x not in common), columnname] = 'other'
    print("now there are", df_train[columnname].nunique(), "categories in train")