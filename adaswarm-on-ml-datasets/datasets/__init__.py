import pandas as pd

def load_sonar():
    df = pd.read_csv("data/sonar.csv", header=None)
    df[60][df[60] == 'R'] =0
    df[60][df[60] == 'M'] = 1
    data_numpy = df.to_numpy()
    return data_numpy[:, :-1], data_numpy[:, -1]

def load_ionosphere():
    df = pd.read_csv("data/ionosphere.csv", header=None)
    df[34][df[34] == 'g'] = 1
    df[34][df[34] == 'b'] = 0
    data_numpy = df.to_numpy()
    return data_numpy[:, :-1], data_numpy[:, -1]
def load_seeds():
    df = pd.read_csv("data/seeds_dataset.csv", header=None)
    df[7][df[7] == 1.0] = 0
    df[7][df[7] == 2.0] = 1
    df[7][df[7] == 3.0] = 2
    data_numpy = df.to_numpy()
    return data_numpy[:, :-1], data_numpy[:, -1]
def load_breast_cancer():
    print("Breast Cancer Wisconsin (Diagnostic) Data Set")
    df = pd.read_csv("data/breast-cancer-wisconsin.csv", header=None)
    df = df.drop(0, 1)
    df[10][df[10] == 2] = 0
    df[10][df[10] == 4] = 1
    data_numpy = df.to_numpy()
    return data_numpy[:, :-1], data_numpy[:, -1]

def load_diabetes():
    print("Pima Indians Diabetes Dataset")
    df = pd.read_csv("data/pima-indians-diabetes.csv", header=None)
    data_numpy = df.to_numpy()
    return data_numpy[:, :-1], data_numpy[:, -1]

def load_heart_disease():
    print("Cleveland Heart Disease Dataset")
    df = pd.read_csv("data/heart-disease.csv", header=None)
    for column in df.columns:
        df = df[df[column]!='?']
    #Zero no heart disease
    #One heart disease
    df[13][df[13] > 0] = 1
    data_numpy = df.to_numpy()
    return data_numpy[:, :-1].astype('float64'), data_numpy[:, -1]

def load_census_income():
    print("Census Income Dataset")
    df = pd.read_csv("data/adult.csv", header=None)
    str_cols = [1, 3, 5, 6, 7, 8, 9, 13, 14]
    df[str_cols] = df[str_cols].apply(lambda col:pd.Categorical(col).codes)
    for column in df.columns:
        df[column] = df[column].astype('category')
        df = df[df[column]!='?']
    #<=50k 0
    #>50 1
    data_numpy = df.to_numpy()
    return data_numpy[:, :-1].astype('float64'), data_numpy[:, -1]

def load_haberman_survival():
    print("Haberman’s Survival Dataset")
    df = pd.read_csv("data/haberman.csv", header=None)
    #1 0
    #2 1
    df[3][df[3] == 1] = 0
    df[3][df[3] == 2] = 1
    data_numpy = df.to_numpy()
    return data_numpy[:, :-1].astype('float64'), data_numpy[:, -1]

def load_banknote_authentication():
    print("Banknote Authentication Dataset")
    df = pd.read_csv("data/banknote.csv", header=None)
    #0 (authentic) or 1 (forgery)
    data_numpy = df.to_numpy()
    return data_numpy[:, :-1].astype('float64'), data_numpy[:, -1].astype('int')

def load_wine():
    print("Wine Dataset")
    df = pd.read_csv("data/wine.csv", header=None)
    #1 0
    #2 1
    df[0][df[0] == 1] = 0
    df[0][df[0] == 2] = 1
    df[0][df[0] == 3] = 2
    data_numpy = df.to_numpy()
    return data_numpy[:, 1: ].astype('float64'), data_numpy[:, :1]

def load_car_evaluation():
    print("Car Evaluation Dataset")
    df = pd.read_csv("data/car.csv", header=None)
    str_cols = [0, 1, 4 , 5, 6]
    df[df.columns] = df[df.columns].apply(lambda col:pd.Categorical(col).codes)
    for column in df.columns:
        df[column] = df[column].astype('category')
        df = df[df[column]!='?']
    #<=50k 0
    #>50 1
    data_numpy = df.to_numpy()
    return data_numpy[:, :-1].astype('float64'), data_numpy[:, -1]

    
