import sys
import joblib as jb
import pandas as pd
from sklearn.model_selection import train_test_split
from TrainModels import PredictPRinKP
from sklearn.metrics import accuracy_score, roc_auc_score,\
    balanced_accuracy_score, f1_score, \
    recall_score, precision_score

SEED = PredictPRinKP().SEED

def read_dataset(dataset, gwas):
    """Read datasets .csv ."""
    if gwas is True:
        df_train = pd.read_csv('../data/' + dataset + 'train.csv')

        df_test = pd.read_csv('../data/' + dataset + 'test.csv')

        df_full = pd.concat([df_test, df_train])

        metadata = df_full[['pbr_res']]
        df_full = df_full.drop(columns=['pbr_res'], axis=1)

    else:
        df_full = pd.read_csv('../data/' + dataset + 'full.csv')
        df_full.set_index('isolate', drop=True, inplace=True)

        metadata = df_full[['pbr_res']]
        df_full = df_full.drop(columns=['pbr_res'], axis=1)

    selected_features = pd.read_csv('../selected_features/' + dataset + '.csv')

    return df_full, metadata, selected_features

def get_test_set(dataset, metadata, selected_features):
    """Get train/test set with selected features."""
    dataset = dataset.loc[:, selected_features['Features']]
    X_train, X_test, y_train, y_test = \
        train_test_split(dataset.values,
                         metadata.values.ravel(),
                         test_size=0.25,
                         random_state=SEED,
                         stratify=metadata.values.ravel())

    return X_test, y_test

def run_prediction(X_test, y_test, model, dataset):
    """Run Model predictions and print metrics."""
    model_path = '../models/' + dataset + '/' + dataset + model + '.pkl'
    saved_model = jb.load(model_path)

    pred = saved_model.predict(X_test)
    proba = saved_model.predict_proba(X_test)[:, 1]

    print('Metrics:')
    print('Roc_auc ->', roc_auc_score(y_test, proba))
    print('BACC ->', balanced_accuracy_score(pred, y_test))
    print('Accuracy ->', accuracy_score(pred, y_test))
    print('f1 ->', f1_score(pred, y_test))
    print('Recall ->', recall_score(pred, y_test))
    print('Precision ->', precision_score(pred, y_test))

def main(args):
    """Main run prediction functions."""
    df, metadata, selected_features = read_dataset(args[0], args[1])

    X_test, y_test = get_test_set(df, metadata, selected_features)

    run_prediction(X_test, y_test, args[2], args[0])

    return 1

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
