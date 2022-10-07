import os
import pandas as pd



base = r'C:\Users\Huang Lab\Desktop\汇总Confocal'


dir_names = os.listdir(base)

for dir in dir_names:
    csv_list = []
    file_path = base + "/" + dir
    file_names = os.listdir(file_path)
    for file in file_names:
        file_name = file_path + "/" + file
        print(file_name)
        # if file.split(".")[1] == "csv":
        df = pd.read_csv(file_name)
        csv_list.append(df)

    result = pd.concat(csv_list)
    result.to_csv(base + "/" + f"{dir}.csv", mode='a', index=False, encoding='utf_8_sig')


