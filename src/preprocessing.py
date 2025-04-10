from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

def encode_categorical_features(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    # 2. TODO Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.
    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.
    # Take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the OrdinalEncoder and
    #     OneHotEncoder classes, then use the fitted models to transform all the
    #     datasets.
    
    #1st get only the list of dtype columns 
    list_dtype_col = train_df.select_dtypes(include='object').columns.to_list()
    list_onehot = []
    list_ordinal = []
    #2nd. for each column, check how many distinct items they have.
    for col in list_dtype_col:
        amount_items = len(train_df[ col ].dropna().unique())   #IMPORTANT to drop NaN values seems it can be considered as categories
        if amount_items == 2:   #Apply OrdinalEncoder for those with 2 unique values
            list_ordinal.append(col)
        elif amount_items > 2:  #Apply OneHotEncoder for those with more than 2 values
            list_onehot.append(col)

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    #Assign values on the same columns. Do not add any additional column
    encoder.fit(train_df[list_ordinal])
    train_df[list_ordinal] = encoder.transform(train_df[ list_ordinal ])
    val_df[list_ordinal] = encoder.transform(val_df[ list_ordinal ])
    test_df[list_ordinal] = encoder.transform(test_df[ list_ordinal ])


    #Apply OneHot encoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_df[ list_onehot ])

    ohe_train = encoder.transform(train_df[ list_onehot ])
    ohe_val = encoder.transform(val_df[ list_onehot ])
    ohe_test = encoder.transform(test_df[ list_onehot ])

    df_ohe_train = pd.DataFrame(
            ohe_train,
            columns= encoder.get_feature_names_out(list_onehot),
            index = train_df.index
        )
    
    df_ohe_val = pd.DataFrame(
            ohe_val,
            columns= encoder.get_feature_names_out(list_onehot),
            index = val_df.index
        )
    
    df_ohe_test = pd.DataFrame(
            ohe_test,
            columns= encoder.get_feature_names_out(list_onehot),
            index = test_df.index
        )

    #Join new dataframes to the original dataframes
    train_df = pd.concat([train_df, df_ohe_train], axis=1)
    val_df = pd.concat([val_df, df_ohe_val], axis=1)
    test_df = pd.concat([test_df, df_ohe_test], axis=1)
    #drop old columns
    train_df.drop(columns=list_onehot, inplace=True)
    val_df.drop(columns=list_onehot, inplace=True)
    test_df.drop(columns=list_onehot, inplace=True)

    return train_df, val_df, test_df


def handle_missing_values(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #     model to transform all the datasets.

    imputer = SimpleImputer(strategy='median')
    df_train_imputed = imputer.fit_transform(train_df)  #This is the same as doing "fit" and then "transform", but in a single line.
    df_val_imputed = imputer.transform(val_df)
    df_test_imputed = imputer.transform(test_df)

    train_df = pd.DataFrame(df_train_imputed, columns=train_df.columns.tolist(), index=train_df.index)
    val_df = pd.DataFrame(df_val_imputed, columns = val_df.columns.tolist(), index=val_df.index)
    test_df = pd.DataFrame(df_test_imputed, columns= test_df.columns.tolist(), index= test_df.index)

    return train_df, val_df, test_df


def scale_features_min_max(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.

    min_max_scaler = MinMaxScaler()
    df_train_scaled = min_max_scaler.fit_transform(train_df)
    df_val_scaled = min_max_scaler.fit_transform(val_df)
    df_test_scaled = min_max_scaler.fit_transform(test_df)

    train_df = pd.DataFrame(df_train_scaled,
                            columns = train_df.columns.tolist(),
                            index = train_df.index)
    val_df = pd.DataFrame(df_val_scaled,
                          columns= val_df.columns.tolist(),
                          index= val_df.index)
    test_df = pd.DataFrame(df_test_scaled,
                           columns= test_df.columns.tolist(),
                           index= test_df.index)

    return train_df, val_df, test_df


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"] = working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan})
    working_val_df["DAYS_EMPLOYED"] = working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan})
    working_test_df["DAYS_EMPLOYED"] = working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan})

    #working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)      #this code line throws a warning
    #working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)        #this code line throws a warning
    #working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)       #this code line throws a warning

    working_train_df,working_val_df,working_test_df = encode_categorical_features(working_train_df, working_val_df, working_test_df)

    print("Input train data shape after encoding: ", working_train_df.shape)
    print("Input val data shape after encoding: ", working_val_df.shape)
    print("Input test data shape after encoding: ", working_test_df.shape, "\n")

    working_train_df,working_val_df,working_test_df = handle_missing_values(working_train_df, working_val_df, working_test_df)

    working_train_df,working_val_df,working_test_df = scale_features_min_max(working_train_df, working_val_df, working_test_df)

    #convert dataframes to numpy arrays
    return working_train_df.to_numpy(), working_val_df.to_numpy(), working_test_df.to_numpy()
