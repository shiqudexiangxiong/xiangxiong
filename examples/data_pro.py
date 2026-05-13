from data.universal_csv_processor import process_csv_for_privpgd
filename="E:\\All_Download\\Black Friday Dataset.csv"


if __name__=="__main__":
    process_csv_for_privpgd(
        file_path=filename,
        label_column="Purchase",
        columns_to_drop=["User_ID", "Product_ID"]
    )