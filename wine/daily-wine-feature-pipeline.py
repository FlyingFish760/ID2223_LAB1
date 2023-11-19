import os
import modal
import pandas as pd

LOCAL=False
FEATURE_NAMES = ['fixed_acidity','volatile_acidity','citric_acid',
        'residual_sugar','chlorides','free_sulfur_dioxide',
        'total_sulfur_dioxide','density','ph',
        'sulphates','alcohol']


if LOCAL==False:
    stub = modal.Stub("wine_daily")
    image = modal.Image.debian_slim(python_version='3.9').pip_install(['hopsworks'])

    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()


def generate_wine(wine_quality):
    import pandas as pd
    import random

    csv_path = r'E:\ID2223\LAB1\ID2223_LAB1\wine\feature_ranges.csv'
    feature_ranges = pd.read_csv(csv_path)
    df = pd.DataFrame()
    for feature_name in FEATURE_NAMES:
        if feature_name=='ph': feature_name='pH'   # feature names were lowered during backfiiling
        feature_min = float(feature_ranges[feature_name].loc[wine_quality-2])
        feature_max = float(feature_ranges[feature_name+'.1'].loc[wine_quality-2])
        df[feature_name] = [random.uniform(feature_min, feature_max)]
    df['quality'] = wine_quality
    return df


def get_random_wine():
    import pandas as pd
    import random

    # randomly plick wine quality to generate
    pick_wine_quality = random.randint(3, 9)
    wine_df = generate_wine(pick_wine_quality)
    print("Wine of quality {} added".format(pick_wine_quality))
    return wine_df

def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_df = get_random_wine()

    wine_fg = fs.get_feature_group(name='wine',version=1)
    wine_fg.insert(wine_df)


if __name__=='__main__':
    if LOCAL==True:
        g()
    else:
        with stub.run():
            f.local()





