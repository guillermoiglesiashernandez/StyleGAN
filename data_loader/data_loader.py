import tensorflow as tf
import numpy as np
from functools import partial

# we use different batch size for different resolution, so larger image size
# could fit into GPU memory. The keys is image resolution in log2
batch_sizes = {4: 16, 8: 16, 16: 16, 32: 16, 64: 16, 128: 8, 256: 4, 512: 2, 1024: 1}

def resize_image(res, image):
    # only donwsampling, so use nearest neighbor that is faster to run
    image = tf.image.resize(
        image, (res, res), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return image


def create_dataloader(res, ds_train):
    batch_size = batch_sizes[res]
    # NOTE: we unbatch the dataset so we can `batch()` it again with the `drop_remainder=True` option
    # since the model only supports a single batch size
    dl = ds_train.map(
        partial(resize_image, res), num_parallel_calls=tf.data.AUTOTUNE
    ).unbatch()
    dl = dl.shuffle(200).batch(batch_size, drop_remainder=True).prefetch(1).repeat()
    return dl