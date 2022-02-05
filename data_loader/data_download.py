import os
import gdown
from tensorflow import keras
from zipfile import ZipFile

def download_data(url, output_dir):
    os.makedirs(output_dir)

    output = output_dir + "/data.zip"
    gdown.download(url, output, quiet=True)

    with ZipFile(output, "r") as zipobj:
        zipobj.extractall(output_dir)

def create_dataset(path):
    ds_train = keras.utils.image_dataset_from_directory(
        path, label_mode=None, image_size=(64, 64), batch_size=32
    )
    ds_train = ds_train.map(lambda x: x / 255.0)

    return ds_train