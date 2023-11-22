import os 
import modal

LOCAL=False

if LOCAL==False:
    stub = modal.Stub()
    hopsworks_image = modal.Image.debian_slim(python_version='3.9').pip_install(["hopsworks","joblib","seaborn","scikit-learn==1.1.1","dataframe-image"])
    @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()

    # Get the model from hopsworks
    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=2)
    model_dir = model.download()
    model = joblib.load(model_dir+'/wine_model.pkl')

    # Read features from hopsworks (feature view)
    feature_view = fs.get_feature_view(name="wine_quality", version=1)
    batch_data = feature_view.get_batch_data()

    
    # Make predictions (and only choose the latest prediction to upload)
    y_pred = model.predict(batch_data)
    print("y_pred:", y_pred)
    offset=1
    wine = y_pred[y_pred.size-offset]
    # num = 100
    # wine = y_pred[-num::1]
    print("Wine predicted:", wine)
    dataset_api = project.get_dataset_api()
    dataset_api.upload(wine)

    # Get the true label
    wine_fg = fs.get_feature_group(name="wine", version=1)
    df = wine_fg.read()
    label = df.iloc[-offset]["quality"]
    # label = df.iloc[-num::1]["quality"]
    print("Wine quality actual: ", label)

    # Get the monitor and upload the latest prediction
    monitor_fg = fs.get_or_create_feature_group(
        name="wine_predictions",
        version=2,
        primary_key=["datetime"],
        description="Wine quality prediction monitor"
    )
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [wine],
        'label': [label],
        'datetime': [now]
    }
    # print("wine, label, datatime:", len(wine), len(list(label)), len([now]*num))
    # data = {
    #     'prediction': wine,
    #     'label': list(label),
    #     'datetime': [now]*num
    # }
    monitor_df = pd.DataFrame(data)
    print("monitor_df:", monitor_df)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})

    # Get the recent predictions (insertion will take appro. 1 min), make a plot
    # out of it and upload it to dataset api
    history_df = monitor_fg.read()
    history_df = pd.concat([history_df, monitor_df])

    df_recent = history_df.tail(2)
    dfi.export(df_recent, '/df_recent.png', table_conversion="matplotlib")
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)

    # Create a confusion matrix out of recent predictions (only do this when wine of quality
    # 3-9 were all newly added to feature group)
    predictions = history_df[['prediction']]
    labels = history_df[['label']]
    
    if labels.value_counts().count() == 6:
        print("predictions:", predictions)
        print("labels:", labels)
        res = confusion_matrix(labels, predictions)
        df_cm = pd.DataFrame(res, 
                            ['True 3', 'True 4', 'True 5', 'True 6', 'True 7', 'True 8'],
                            ['Pred 3', 'Pred 4', 'Pred 5', 'Pred 6', 'Pred 7', 'Pred 8'])
        cm = sns.heatmap(df_cm, annot = True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix.png")
        dataset_api.upload("./confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("You need 7 different wine predictions to generate a confusion matrix")
    
    print("Finished {} times*******************\n".format(offset))

if __name__=='__main__':
    if LOCAL==True:
        g()
    else:
        with stub.run():
            f.local()