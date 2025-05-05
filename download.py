import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def clear_data(path2df):
    mat_df = pd.read_csv(path2df, delimiter = ';')

    Mjob_Fjob_dict = {'teacher': 0,
                      'health': 1,
                      'services': 2,
                      'at_home': 3,
                      'other': 4
                      }
    Reason_dict = {'home': 0,
                   'reputation': 1,
                   'course': 2,
                   'other': 3
                   }
    Guardian_dict = {'mother': 0,
                     'father': 1,
                     'other': 2
                     }
    mat_df['Mjob'] = mat_df['Mjob'].map(Mjob_Fjob_dict)
    mat_df['Fjob'] = mat_df['Fjob'].map(Mjob_Fjob_dict)
    mat_df['reason'] = mat_df['reason'].map(Reason_dict)
    mat_df['guardian'] = mat_df['guardian'].map(Guardian_dict)

    mat_df['school'] = mat_df['school'].apply(lambda x: 0 if x == 'GP' else 1)
    mat_df['sex'] = mat_df['sex'].apply(lambda x: 0 if x == 'F' else 1)
    mat_df['address'] = mat_df['address'].apply(lambda x: 0 if x == 'U' else 1)
    mat_df['famsize'] = mat_df['famsize'].apply(lambda x: 0 if x == 'LE3' else 1)
    mat_df['Pstatus'] = mat_df['Pstatus'].apply(lambda x: 0 if x == 'T' else 1)
    mat_df['schoolsup'] = mat_df['schoolsup'].apply(lambda x: 0 if x == 'yes' else 1)
    mat_df['famsup'] = mat_df['famsup'].apply(lambda x: 0 if x == 'yes' else 1)
    mat_df['paid'] = mat_df['paid'].apply(lambda x: 0 if x == 'yes' else 1)
    mat_df['activities'] = mat_df['activities'].apply(lambda x: 0 if x == 'yes' else 1)
    mat_df['nursery'] = mat_df['nursery'].apply(lambda x: 0 if x == 'yes' else 1)
    mat_df['higher'] = mat_df['higher'].apply(lambda x: 0 if x == 'yes' else 1)
    mat_df['internet'] = mat_df['internet'].apply(lambda x: 0 if x == 'yes' else 1)
    mat_df['romantic'] = mat_df['romantic'].apply(lambda x: 0 if x == 'yes' else 1)
    mat_df.to_csv('mat_df_clear.csv')
    return True



clear_data('student-mat.csv')
