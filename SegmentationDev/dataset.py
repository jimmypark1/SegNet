import tensorflow as tf
import numpy as np
from PIL import Image
import numpy as np
import skimage.io as io
import utils


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def generate_dataset():
    tfrecords_filename = 'data.tfrecords'

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    # Let's collect the real images to later on compare
    # to the reconstructed ones
    original_images = []

    filename_pairs = None

    inputs = utils.get_files('input_dev')
    targets = utils.get_files('target_dev')

    filename_pairs = zip(inputs, targets)

    for img_path, annotation_path in filename_pairs:
        # img = np.array(Image.open(img_path))

        img = utils.get_img(img_path, img_size=[48, 48])
        img = np.array(img)
        # img = np.reshape(img, [-1, 48, 48, 3])

        # annotation = np.array(Image.open(annotation_path))
        annotation = utils.get_img(annotation_path, img_size=[48, 48])
        annotation = np.array(Image.fromarray(annotation.astype(np.uint8)).convert('L'))
        #annotation = annotation / 255.0;
        #print('annotation',annotation)
        for i in range(48):
            for j in range(48):
                #print(annotation[i, j])

                if annotation[i, j] <= 10:
                    annotation[i, j] = 0
                else:
                    annotation[i, j] = 255

        annotation = np.array(annotation)

        # The reason to store image sizes was demonstrated
        # in the previous example -- we have to know sizes
        # of images to later read raw serialized string,
        # convert to 1d array and convert to respective
        # shape that image used to have.
        height = img.shape[0]
        width = img.shape[1]
        print('width', width)
        # Put in the original images into array
        # Just for future check for correctness
        original_images.append((img, annotation))

        img_raw = img.tostring()
        annotation_raw = annotation.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
            'mask_raw': _bytes_feature(annotation_raw)}))

        writer.write(example.SerializeToString())

    writer.close()


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string)
        })

    print('features',features)
    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    annotation = tf.decode_raw(features['mask_raw'], tf.uint8)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    #print('width',width)
    #image_shape = tf.pack([height, width, 3])
    #annotation_shape = tf.pack([height, width, 1])

    image = tf.reshape(image, shape=[height,width,3])
    annotation = tf.reshape(annotation, shape=[height,width,1])

    image_size_const = tf.constant((48, 48, 3), dtype=tf.int32)
    annotation_size_const = tf.constant((48, 48, 1), dtype=tf.int32)

    # Random transformations can be put here: right before you crop images
    # to predefined size. To get more information look at the stackoverflow
    # question linked above.

    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                           target_height=48,
                                                           target_width=48)

    resized_annotation = tf.image.resize_image_with_crop_or_pad(image=annotation,
                                                                target_height=48,
                                                                target_width=48)

    images, annotations = tf.train.shuffle_batch([resized_image, resized_annotation],
                                                 batch_size=10,
                                                 capacity=30,
                                                 num_threads=2,
                                                 min_after_dequeue=10)

    return images, annotations


generate_dataset()