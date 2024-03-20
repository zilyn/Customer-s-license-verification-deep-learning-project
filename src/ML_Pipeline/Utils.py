import keras
import pickle

PREDICTORS = ['APPLICATION_TYPE', 'CONDITIONAL_APPROVAL', 'LICENSE_CODE', 'SSA', 'LEGAL_BUSINESS_NAME_MATCH',
              'ZIP_CODE_MISSING', 'SSA', 'APPLICATION_REQUIREMENTS_COMPLETE', 'LICENSE_DESCRIPTION', 'BUSINESS_TYPE']

TARGET = ["LICENSE_STATUS_AAC", "LICENSE_STATUS_AAI", "LICENSE_STATUS_REV"]


def save_model(model, columns):
    model.save("../original/output/dnn-model")

    file = open("../original/output/columns.mapping", "wb")
    pickle.dump(columns, file)
    file.close()

    return True


def load_model(model_path):
    model = None
    try:
        model = keras.models.load_model(model_path)
    except:
        print("Please enter correct path")
        exit(0)

    file = open("../original/output/columns.mapping", "rb")
    columns = pickle.load(file)
    file.close()

    return model, columns
