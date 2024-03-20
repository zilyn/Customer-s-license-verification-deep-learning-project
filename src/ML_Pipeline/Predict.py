import pandas as pd


def init(test_data, model, columns):
    # Preprocess test data to do categorical encoding and match columns expected
    # by the model
    new_cols = [x for x in columns if x not in test_data.columns]

    new_df = pd.DataFrame(columns=new_cols, index=range(test_data.shape[0]))
    new_df.fillna(0, inplace=True)
    test_data = pd.concat([test_data, new_df.reindex(test_data.index)], axis=1)
    test_data = test_data[columns]
    test_data = test_data.loc[:, ~test_data.columns.duplicated()]

    from ML_Pipeline import Utils
    for col in Utils.TARGET:
        try:
            test_data = test_data.drop(col, axis=1)
        except:
            continue

    x_test = test_data.values
    predict = model.predict(x_test.astype('float32'))
    return predict
