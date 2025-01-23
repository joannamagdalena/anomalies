# mrwellsdavid/unsw-nb15
import pandas as pd
import oracledb
from datetime import date

# loading data from csv
def loading_data_from_csv():
    dataset_train_full = pd.read_csv("../unsw-nb15/UNSW_NB15_training-set.csv")
    dataset_test_full = pd.read_csv("../unsw-nb15/UNSW_NB15_testing-set.csv")

    type_change = ["dur","proto","service","state","rate","sload","dload","sinpkt","dinpkt","sjit","djit","tcprtt",
                   "synack","ackdat","attack_cat","label"]
    dataset_train_full[type_change] = dataset_train_full[type_change].astype(str)
    dataset_test_full[type_change] = dataset_test_full[type_change].astype(str)

    dataset_train = dataset_train_full.drop(["attack_cat", "id"], axis=1)
    dataset_test = dataset_test_full.drop(["attack_cat", "id"], axis=1)

    return dataset_train_full, dataset_test_full


# loading data from Oracle db
def loading_data_from_oracle_db():
    current_year = date.today().year
    current_month = date.today().month
    current_date = str(date.today())

    connection_to_db = oracledb.connect(user="SYSTEM", password="12345", host="localhost", port=1521)

    cursor_db = connection_to_db.cursor()
    cursor_db.execute("select distinct extract(year from loading_date) as year, extract(month from loading_date) as month from UNSW_NB15 where loading_date = (select max(loading_date) from UNSW_NB15)")
    max_date = cursor_db.fetchall()[0]
    cursor_db.close()

    # checking the datasets---if data is missing, loading it
    if max_date[0] != current_year or max_date[1] != current_month:
        print("Data was not loaded.")

        cursor_db = connection_to_db.cursor()
        cursor_db.execute("select max(nrid) from UNSW_NB15")
        max_nrid = int(cursor_db.fetchall()[0][0])
        cursor_db.close()

        dataset_train_to_load, dataset_test_to_load = loading_data_from_csv()
        cursor_db = connection_to_db.cursor()
        for index, row in dataset_train_to_load.iterrows():
            inp = list(row)
            inp[0] += max_nrid
            command = "insert into UNSW_NB15 values " + "(" + str(inp)[1:-1] + ", TO_DATE('" + current_date + "','YYYY-MM-DD')" + ")"
            cursor_db.execute(command)
            connection_to_db.commit()
        cursor_db.close()

    cursor_db = connection_to_db.cursor()

    cursor_db.execute("select * from UNSW_NB15")
    dataset_train_from_oracle = pd.DataFrame(cursor_db.fetchall())
    dataset_train_from_oracle.columns = [d[0] for d in cursor_db.description]

    cursor_db.execute("select * from UNSW_NB15_TEST")
    dataset_test_from_oracle = pd.DataFrame(cursor_db.fetchall())
    dataset_test_from_oracle.columns = [d[0] for d in cursor_db.description]

    cursor_db.close()
    connection_to_db.close()

    return dataset_train_from_oracle, dataset_test_from_oracle


loading_data_from_oracle_db()