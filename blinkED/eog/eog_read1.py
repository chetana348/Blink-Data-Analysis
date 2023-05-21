import re
import numpy as np

match = np.vectorize(re.match)

def eog_read1(file_path: str, readings_header: str = 'CH1\tCH2\tCH3\t', include_header=False):
    
    """
    Personalized function to read and interpret our eog data files.

    Full path to the files must be provided.
    """
    with open(file_path, 'r') as f:
        file_aslist = f.readlines()
    table_start = np.where(match('CH\d', file_aslist) != None)[0][0]
    df_eog = []
    for line in file_aslist[table_start:]:
        if not line.startswith('-'):
            break
        data = line.strip().split()
        df_eog.append([float(x) for x in data])
    df_eog = pd.DataFrame(df_eog, columns=['CH1', 'CH2'])
    if include_header:
        return df_eog, file_aslist[:table_start]
    return df_eog
