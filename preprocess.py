import numpy as np
import pandas as pd


def split_timedeltas(seq):
    cond = ((seq['MINUTE'] == 0) | (seq['MINUTE'] == 30)) \
    & ((seq['TIMEDELTA'] == 0) | (seq['TIMEDELTA'] == 30) | (seq['TIMEDELTA'].shift(-1) != 0))
    
    delta30 = seq[cond].copy()
    delta1 = seq[~cond].copy()
    
    delta30 = delta30.reindex(columns=[*delta30.columns.tolist(), 'MIXA_PASTEUR_SECTEMP', 'MIXB_PASTEUR_SECTEMP'], fill_value=np.nan)
    delta1.rename(columns={'MIXA_PASTEUR_TEMP':'MIXA_PASTEUR_SECTEMP', 'MIXB_PASTEUR_TEMP':'MIXB_PASTEUR_SECTEMP'}, inplace=True)
    delta1 = delta1.reindex(columns=[*delta1.columns.tolist(), 'MIXA_PASTEUR_TEMP', 'MIXB_PASTEUR_TEMP'], fill_value=np.nan)
    
    cols = ['STD_DT', 'DATE', 'TIME', 'HOUR', 'MINUTE', 'TIMEDELTA',
            'MIXA_PASTEUR_STATE_0', 'MIXA_PASTEUR_STATE_1', 'MIXA_PASTEUR_STATE_nan',
            'MIXB_PASTEUR_STATE_0', 'MIXB_PASTEUR_STATE_1', 'MIXB_PASTEUR_STATE_nan',
            'MIXA_PASTEUR_TEMP', 'MIXB_PASTEUR_TEMP', 'MIXA_PASTEUR_SECTEMP', 'MIXB_PASTEUR_SECTEMP', 'INSP']
    delta1 = delta1.loc[:, cols]
    delta30 = delta30.loc[:, cols]
    
    merged = pd.concat([
        delta1, delta30
    ], axis=0)
    merged.sort_values(by=['STD_DT', 'TIMEDELTA'], ascending=[True, False], inplace=True)
    return merged


if __name__ == '__main__':
    data = pd.read_csv('data/data.csv')

    data['STD_DT'] = pd.to_datetime(data['STD_DT'], format='%Y-%m-%d %H:%M')
    data['DATE'] = data['STD_DT'].dt.strftime('%Y-%m-%d')
    data['TIME'] = data['STD_DT'].dt.strftime('%H:%M')
    data['HOUR'] = data['STD_DT'].dt.hour
    data['MINUTE'] = data['STD_DT'].dt.minute
    data['TIMEDELTA'] = data['STD_DT'].diff().astype('timedelta64[m]')
    data.insert(1, 'DATE', data.pop('DATE'))
    data.insert(2, 'TIME', data.pop('TIME'))
    data.insert(3, 'HOUR', data.pop('HOUR'))
    data.insert(4, 'MINUTE', data.pop('MINUTE'))
    data.insert(5, 'TIMEDELTA', data.pop('TIMEDELTA'))
    data['INSP'] = data['INSP'].apply(lambda x: ['OK', 'NG'].index(x))
    data.loc[data['TIMEDELTA'] > 60, 'TIMEDELTA'] = 0
    data.loc[data['MIXA_PASTEUR_STATE'] > 1000, 'MIXA_PASTEUR_STATE'] = np.nan
    data.loc[data['MIXB_PASTEUR_STATE'] > 1000, 'MIXB_PASTEUR_STATE'] = np.nan
    data[['MIXA_PASTEUR_STATE', 'MIXB_PASTEUR_STATE', 'INSP']] = data[['MIXA_PASTEUR_STATE', 'MIXB_PASTEUR_STATE', 'INSP']].astype('category')
    data.loc[data['MIXB_PASTEUR_TEMP'] > 1000, 'MIXB_PASTEUR_TEMP'] = np.nan
    
    data_X = data.iloc[:, :-1][['TIMEDELTA', 'MIXA_PASTEUR_TEMP', 'MIXB_PASTEUR_TEMP', 'MIXA_PASTEUR_SECTEMP', 'MIXB_PASTEUR_SECTEMP']]
    data_X = data_X.interpolate('linear', limit_direction='forward').fillna(0).to_numpy()
    data_y = data['INSP'].to_numpy()
    
    with open('data/original_sectemp_X.npy', 'wb') as f:
        np.save(f, data_X)
    with open('data/data_y.npy', 'wb') as f:
        np.save(f, data_y)