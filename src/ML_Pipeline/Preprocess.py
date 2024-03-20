import numpy as np
import pandas as pd


# Function to cleanup data, convert to numerical variables and impute wherever required
def cleanup(data):
    data['LEGAL_BUSINESS_NAME_MATCH'] = data \
        .apply(lambda x: 1 if str(x['LEGAL_NAME'].upper()) in str(x['DOING_BUSINESS_AS_NAME']).upper()
                              or str(x['DOING_BUSINESS_AS_NAME']).upper() in str(x['LEGAL_NAME']).upper() else 0,
               axis=1)

    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Motor Vehicle Repair : Engine Only (Class II)',
                                                                      'Motor Vehicle Repair')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Motor Vehicle Repair: Engine/Body(Class III)',
                                                                      'Motor Vehicle Repair')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Motor Vehicle Repair; Specialty(Class I)',
                                                                      'Motor Vehicle Repair')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Day Care Center Under 2 Years',
                                                                      'Day Care Center')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Day Care Center 2 - 6 Years', 'Day Care Center')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Day Care Center Under 2 and 2 - 6 Years',
                                                                      'Day Care Center')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Peddler, non-food', 'Peddler')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Peddler, non-food, special', 'Peddler')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Peddler, food (fruits and vegtables only)',
                                                                      'Peddler')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace(
        'Peddler,food - (fruits and vegetables only) - special', 'Peddler')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Tire Facilty Class I (100 - 1,000 Tires)',
                                                                      'Tire Facilty')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Tire Facility Class II (1,001 - 5,000 Tires)',
                                                                      'Tire Facilty')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Tire Facility Class III (5,001 - More Tires)',
                                                                      'Tire Facilty')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Repossessor Class A', 'Repossessor')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Repossessor Class B', 'Repossessor')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Repossessor Class B Employee', 'Repossessor')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Expediter - Class B', 'Expediter')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Expediter - Class A', 'Expediter')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Expediter - Class B Employee', 'Expediter')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Itinerant Merchant, Class II',
                                                                      'Itinerant Merchant')
    data['LICENSE_DESCRIPTION'] = data['LICENSE_DESCRIPTION'].replace('Itinerant Merchant, Class I',
                                                                      'Itinerant Merchant')

    data['LEGAL_NAME'] = data['LEGAL_NAME'].str.replace('.', '', regex=False)
    data['DOING_BUSINESS_AS_NAME'] = data['DOING_BUSINESS_AS_NAME'].str.replace('.', '', regex=False)

    # Impute business type
    data['BUSINESS_TYPE'] = 'PVT'

    data['BUSINESS_TYPE'] = np.where(data['LEGAL_NAME'].str.contains('INC'), 'INC', data['BUSINESS_TYPE'])
    data['BUSINESS_TYPE'] = np.where(data['LEGAL_NAME'].str.contains('INCORPORATED'), 'INC', data['BUSINESS_TYPE'])
    data['BUSINESS_TYPE'] = np.where(data['DOING_BUSINESS_AS_NAME'].str.contains('INC'), 'INC', data['BUSINESS_TYPE'])
    data['BUSINESS_TYPE'] = np.where(data['DOING_BUSINESS_AS_NAME'].str.contains('INCORPORATED'), 'INC',
                                     data['BUSINESS_TYPE'])
    data['BUSINESS_TYPE'] = np.where(data['LEGAL_NAME'].str.contains('LLC'), 'LLC', data['BUSINESS_TYPE'])
    data['BUSINESS_TYPE'] = np.where(data['DOING_BUSINESS_AS_NAME'].str.contains('LLC'), 'LLC', data['BUSINESS_TYPE'])
    data['BUSINESS_TYPE'] = np.where(data['LEGAL_NAME'].str.contains('CO'), 'CORP', data['BUSINESS_TYPE'])
    data['BUSINESS_TYPE'] = np.where(data['LEGAL_NAME'].str.contains('CORP'), 'CORP', data['BUSINESS_TYPE'])
    data['BUSINESS_TYPE'] = np.where(data['LEGAL_NAME'].str.contains('CORPORATION'), 'CORP', data['BUSINESS_TYPE'])
    data['BUSINESS_TYPE'] = np.where(data['DOING_BUSINESS_AS_NAME'].str.contains('CO'), 'CORP', data['BUSINESS_TYPE'])
    data['BUSINESS_TYPE'] = np.where(data['DOING_BUSINESS_AS_NAME'].str.contains('CORP'), 'CORP', data['BUSINESS_TYPE'])
    data['BUSINESS_TYPE'] = np.where(data['DOING_BUSINESS_AS_NAME'].str.contains('CORPORATION'), 'CORP',
                                     data['BUSINESS_TYPE'])
    data['BUSINESS_TYPE'] = np.where(data['LEGAL_NAME'].str.contains('LTD'), 'LTD', data['BUSINESS_TYPE'])
    data['BUSINESS_TYPE'] = np.where(data['LEGAL_NAME'].str.contains('LIMITED'), 'LTD', data['BUSINESS_TYPE'])
    data['BUSINESS_TYPE'] = np.where(data['DOING_BUSINESS_AS_NAME'].str.contains('LTD'), 'LTD', data['BUSINESS_TYPE'])
    data['BUSINESS_TYPE'] = np.where(data['DOING_BUSINESS_AS_NAME'].str.contains('LIMITED'), 'LTD',
                                     data['BUSINESS_TYPE'])

    data['ZIP_CODE'].fillna(-1, inplace=True)
    data['ZIP_CODE_MISSING'] = data.apply(lambda x: 1 if x['ZIP_CODE'] == -1 else 0, axis=1)
    data['SSA'].fillna(-1, inplace=True)

    data['APPLICATION_REQUIREMENTS_COMPLETE'].fillna(-1, inplace=True)
    data['APPLICATION_REQUIREMENTS_COMPLETE'] = data\
        .apply(lambda x: 0 if x['APPLICATION_REQUIREMENTS_COMPLETE'] == -1 else 1, axis=1)

    return data


# Function to encode categorical variables
def categorical_encode(data):
    try:
        from ML_Pipeline.Utils import PREDICTORS
        final_df = data[PREDICTORS + ["LICENSE_STATUS"]]
        final_df = pd.get_dummies(final_df, columns=['APPLICATION_TYPE', 'CONDITIONAL_APPROVAL', 'LICENSE_CODE',
                                                     'LICENSE_DESCRIPTION', 'BUSINESS_TYPE', 'LICENSE_STATUS'])
    except:
        # For test data
        final_df = data[PREDICTORS]
        final_df = pd.get_dummies(final_df, columns=['APPLICATION_TYPE', 'CONDITIONAL_APPROVAL', 'LICENSE_CODE',
                                                     'LICENSE_DESCRIPTION', 'BUSINESS_TYPE'])

    return final_df


# Function to call dependent functions
def apply(data):
    print("Preprocessing started....")

    data = cleanup(data)
    print("Data cleanup completed....")

    data = categorical_encode(data)
    print("Categorical encoding completed....")

    data = data.loc[:, ~data.columns.duplicated()]

    print("Preprocessing completed....")
    return data
