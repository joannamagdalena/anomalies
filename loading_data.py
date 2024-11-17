# mrwellsdavid/unsw-nb15
import pandas as pd
import oracledb

# loading data from csv
def loading_data_from_csv():
    dataset_train_full = pd.read_csv("../unsw-nb15/UNSW_NB15_training-set.csv")
    dataset_test_full = pd.read_csv("../unsw-nb15/UNSW_NB15_testing-set.csv")

    dataset_train = dataset_train_full.drop(["attack_cat", "id"], axis=1)
    dataset_test = dataset_test_full.drop(["attack_cat", "id"], axis=1)

    return dataset_train, dataset_test


# loading data from Oracle db
def loading_data_from_oracle_db():
    connection_to_db = oracledb.connect(user="SYSTEM", password="12345", host="localhost", port=1521)

    cursor_db = connection_to_db.cursor()
    cursor_db.execute("select * from UNSW_NB15")
    dataset_train_from_oracle = pd.DataFrame(cursor_db.fetchall())
    dataset_train_from_oracle.columns = [d[0] for d in cursor_db.description]
    cursor_db.close()

    connection_to_db.close()

    return dataset_train_from_oracle