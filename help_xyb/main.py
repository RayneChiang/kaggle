import os
import pandas as pd
import seaborn as sns


if __name__ == '__main__':
    df = pd.read_csv('1.csv')
    # file = file.values
    target = df[df['E2'] == ' ']
    E1_frame = target[['E1_a', 'E1_b', 'E1_c', 'E1_d', 'E1_e', 'E1_f', 'E1_g', 'E1_h']]

    cols = ['E1_a', 'E1_b', 'E1_c', 'E1_d', 'E1_e', 'E1_f', 'E1_g', 'E1_h']
    for index, col in enumerate(cols):
        index_list = E1_frame[col][E1_frame[col] == 1].index
        for index in index_list:
            df.loc[index, 'E2'] = col

    left_frame = df[df['E2'] == ' ']
    print(len(left_frame))

    df.to_csv('1_1.csv')