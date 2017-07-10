#import dataset
import tensorflow as tf
import skimage.io as io
import model
import numpy as np

#def read_and_decode(filename_queue):
tfrecords_filename = 'data.tfrecords'


batch_size = 1

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

    image = tf.reshape(image, [48,48,3])
    annotation = tf.reshape(annotation, [48,48,1])
    print('image',image)
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

    #batch_size = 1

    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    images, annotations = tf.train.batch([image, annotation],
                                                 batch_size=batch_size
                                                 )

    return images, annotations
########
############ model define ###########
x = tf.placeholder(tf.float32, shape=[batch_size, 48, 48, 3], name="input")

y = tf.placeholder(tf.float32, shape=[batch_size, 48 * 48], name="output_placeholder")

prob = tf.placeholder(tf.float32)

weight_kener_size = 5

conv1_weight_shape = [weight_kener_size, weight_kener_size, 3, 64]
conv2_weight_shape = [weight_kener_size, weight_kener_size, 64, 64]
conv3_weight_shape = [3, 3, 64, 64]

fc4_weight_shape = [12 * 12 * 64, 100]
fc5_weight_shape = [100, 400]
fc6_weight_shape = [400, 48 * 48]

print('process')
conv1_weight = model.weights_variables(conv1_weight_shape, "conv1_weight")
conv1_bias = model.bias_variables([64], "conv1_bias")
conv2_weight = model.weights_variables(conv2_weight_shape, "conv2_weight")
conv2_bias = model.bias_variables([64], "conv2_bias")
conv3_weight = model.weights_variables(conv3_weight_shape, "conv3_weight")
conv3_bias = model.bias_variables([64], "conv3_bias")
fc4_weight = model.weights_variables(fc4_weight_shape, "fc4_weight")
fc4_bias = model.bias_variables([100], "fc4_bias")
fc5_weight = model.weights_variables(fc5_weight_shape, "fc5_weight")
fc5_bias = model.bias_variables([400], "fc5_bias")
fc6_weight = model.weights_variables(fc6_weight_shape, "fc6_weight")
fc6_bias = model.bias_variables([48 * 48], "fc6_bias")

#for i in range(7):
#    with tf.device('/gpu:%d' % i):
#for d in ['/gpu:2', '/gpu:3']:
#  with tf.device(d):

conv1 = tf.nn.conv2d(x, conv1_weight, strides=[1, 1, 1, 1], padding='SAME', name='conv')
pool1 = tf.nn.relu(conv1+conv1_bias)
pool1 = tf.nn.max_pool(pool1, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')
#pool1 = tf.nn.dropout(pool1,keep_prob=prob)

kenel2 = conv2_weight# + conv2_bias
conv2 = tf.nn.conv2d(pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME', name='conv2')
pool2 = tf.nn.relu(conv2+conv2_bias)
pool2 = tf.nn.max_pool(pool2, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')
#pool2 = tf.nn.dropout(pool2,keep_prob=prob)

conv3 = tf.nn.conv2d(pool2, conv3_weight, strides=[1, 1, 1, 1], padding='SAME', name='conv2')
conv3 = tf.nn.relu(conv3+conv3_bias)
#conv3 = tf.nn.dropout(conv3,keep_prob=prob)

h_flat = tf.reshape(conv3, [batch_size, 12 * 12 * 64])  # -1은 batch size를 유지하는 것.

fc4 = (tf.matmul(h_flat, fc4_weight) + fc4_bias)
fc4 = tf.nn.sigmoid(fc4)
fc4 = tf.nn.dropout(fc4,keep_prob=prob)
#print('fc4', fc4)

fc5 = (tf.matmul(fc4, fc5_weight) + fc5_bias)
fc5 = tf.nn.sigmoid(fc5)
fc5 = tf.nn.dropout(fc5,keep_prob=prob)
#fc5 = tf.nn.dropout(fc5,keep_prob=prob)

#print('fc5', fc5)

# h_flat = tf.reshape(fc5, [-1, 12 * 12 * 64])  # -1은 batch size를 유지하는 것.
h = tf.matmul(fc5, fc6_weight) + fc6_bias
# fc6 = tf.nn.sigmoid(h, name="fc6")
output_node = tf.nn.sigmoid(h, name="output")

model_directory = './weights/'

####### output 48*48
#y = tf.constant(img_target)
regularizer = tf.nn.l2_loss(conv1_weight) +  tf.nn.l2_loss(conv2_weight)+ tf.nn.l2_loss(conv3_weight)+  tf.nn.l2_loss(fc4_weight)+  \
              tf.nn.l2_loss(fc5_weight)+  tf.nn.l2_loss(fc6_weight) + tf.nn.l2_loss(conv1_bias) + tf.nn.l2_loss(conv2_bias) + \
              tf.nn.l2_loss(conv3_bias)+  tf.nn.l2_loss(fc4_bias)+  tf.nn.l2_loss(fc5_bias)+  tf.nn.l2_loss(fc6_bias)


#loss = (tf.nn.l2_loss(y - output_node))#+ 0.001*regularizer#+ 0.01*regularizer#+0.0001*regularizer)
#loss = tf.reduce_mean(tf.nn.l2_loss(y - output_node))#+ 0.001*regularizer#+ 0.01*regularizer#+0.0001*regularizer)
#loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum((y-output_node)*(y-output_node),reduction_indices=[1])))
#loss = tf.sqrt(tf.reduce_sum((y-output_node)*(y-output_node)))#+ 0.01*regularizer#+0.0001*regularizer)
loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(((y-output_node)*(y-output_node)))))


learning_rate =  0.2
decay_rate = 0.85

#loss = tf.reduce_mean(tf.sqrt(y-output_node))#tf.nn.l2_loss(y - fc6))

lr = tf.constant(.003, name='learning_rate')
global_step = tf.Variable(0, trainable=False)

correct = tf.equal(tf.argmax((y),1),tf.argmax((output_node),1))
accuracy = tf.reduce_mean((tf.cast(correct,tf.float32)))


global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.0001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 0.95, staircase=True)

# Optimizer.
#loss = tf.reduce_mean(((output_node - y)*(output_node - y)))

train_op = tf.train.MomentumOptimizer(learning_rate = 0.0035,momentum = 0.85,use_nesterov = True).minimize(loss)
#mean_cost = 0
#train_op = tf.train.AdamOptimizer(0.06).minimize(loss)
#train_op = tf.train.GradientDescentOptimizer(0.6).minimize(loss)
saver = tf.train.Saver()
tf.summary.scalar('loss',loss)
tf.summary.scalar('learning_rate',learning_rate)


probability = 1#0.6#ß0.995#0.95




#########
tfrecords_filename = 'data.tfrecords'

filename_queue = tf.train.string_input_producer(["data.tfrecords"],num_epochs=2000000)
image, annotation = read_and_decode(filename_queue)


init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session(config=tf.ConfigProto( log_device_placement=True))  as sess:
    sess.run(init_op)
    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter('train/train_log',
                                         sess.graph)

    #    sess.run(tf.local_variables_initializer())
#    sess.run(tf.global_variables_initializer())

    #coord = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(coord=coord)

    #print('current batch0')

    #img, anno = sess.run([image, annotation])
    #print('current batch1')


    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners( coord=coord)
    iter = 0
    i=0
    #try:
        #while not coord.should_stop():
    for iter in range(20000):
            # Run training steps or whatever
            total_batch = int(4744 / batch_size)
            cost_history = np.zeros(shape=[1], dtype=float)

            for i in range(total_batch):

                #while i * 50 < 4877:
                    #offset = (iter * batch_size) % (total_batch_size - batch_size)

                    #print('current batch0')
                img, anno = sess.run([image, annotation])


                anno = np.reshape(anno,[-1,48*48])
                anno = anno/255.0
                #print('anno',anno)
                out, _loss, summary = sess.run([train_op, loss,merged], feed_dict={x: img, y: anno, prob: probability})
                #i = i+1
                cost_history = np.append(cost_history, _loss)

                #print('i', i, 'loss', _loss)
            mean_cost = np.mean(cost_history)

            #if iter % 10 == 0:
            saver.save(sess, 'model/model.ckpt')
            train_writer.add_summary(summary, iter)

            #summary = sess.run(merged, feed_dict={x: img, y: anno, prob: probability})

            print('step', iter, 'loss', mean_cost, 'img shape', img.shape)

            #iter = iter+1
    coord.request_stop()


            # print(img[0, :, :, :].shape)

                #print('current batch')

                # We selected the batch size of two
                # So we should get two image pairs in each batch
                # Let's make sure it is random

                #io.imshow(img[0, :, :, :])
                #io.show()

                #io.imshow(anno[0, :, :, 0])
                #io.show()

                #io.imshow(img[1, :, :, :])
                #io.show()

                #io.imshow(anno[1, :, :, 0])
                #io.show()

#    except tf.errors.OutOfRangeError:
#        print('Done training -- epoch limit reached')
#    finally:
        # When done, ask the threads to stop.

    # Wait for threads to finish.
    coord.join(threads)

    # Let's read off 3 batches just for example


    #coord.request_stop()
    #coord.join(threads)

