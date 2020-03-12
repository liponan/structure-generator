import numpy as np
import pandas as pd


def parse_pdb(text):
    data = list()
    for line in text:
#         print(line[0:4])
        if line[0:4] == "ATOM":
            record = line[0:4]
            atom_no = int(line[6:11])
            atom_name = line[12:16]
            alt = line[16]
            res_name = line[17:20]
            chain = line[21]
            res_no = int(line[22:26])
            insertion = line[26]
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            occupancy = float(line[54:60])
            factor = float(line[60:66])
            data.append([record, atom_no, atom_name, alt, res_name, chain, res_no, insertion, x, y, z, occupancy, factor])
        else:
            continue
        
    df = pd.DataFrame(data, columns=["record", "atom_no", "atom_name", "alt", "res_name", "chain", "res_no",
                                     "insertion", "x", "y", "z", "occupancy", "factor"])
    return df


def extract_seq(df):
    df = df[["chain", "res_name", "res_no"]].drop_duplicates().reset_index(drop=True)
    return df
        
