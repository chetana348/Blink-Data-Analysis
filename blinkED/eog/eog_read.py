import pandas as pd
import re
import chardet
import numpy as np

match = np.vectorize(re.match)
def eog_read(file_path: str, readings_header: str = 'CH1\tCH2\tCH3\t', include_header = False):
    """
    Personalized funtion to read and interpret our eog data files. \n

    full path to the files must be listed
    """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        file_encoding = result['encoding']
    f = open(file_path, 'r',encoding=file_encoding)
    file_aslist = [line.rstrip('\n') for line in f]
    table_start = np.where(match('CH\d', file_aslist) != None)[0][0]
    df_eog = pd.read_csv(file_path, sep ="\t", skiprows = table_start - 1)
    for i in df_eog.columns:
        if 'CH' not in i:
            df_eog = df_eog.drop(i, axis = 1)
    df_eog.set_axis(['CH1', 'CH2'], axis = 1)
    if include_header: return df_eog, file_aslist[:table_start]
    return df_eog
