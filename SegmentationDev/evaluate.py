import tensorflow as tf
import numpy as np
from PIL import Image
import scipy.misc
import utils
import model
import segmentizer
import net

#test_path = 'learning/00001.jpg'
#test_path = 'learning2/57.jpg'
#test_path = 'input.jpg'
#learning_list = utils.get_files('resized_img/learning')
learning_list = utils.get_files('input_dev')



#input0 = utils.get_img(test_path,img_size=[48,48])
#input0 = np.array(Image.fromarray(input0.astype(np.uint8)))#.convert('L'))

#input_org = input0
#input1 = np.reshape(input0, [-1, input0.shape[0], input0.shape[1], 3])
#input = input_#np.zeros([1,  48, 48,3])
#input = np.zeros([1,  48, 48,3])

def imsave(path, img):
    #img = np.clip(img, 0, 255).astype(np.uint8)
    #print('out img data', img)
    Image.fromarray(img).save(path, quality=100)


device_ = '/cpu:0'

x =tf.placeholder(tf.float32, shape=[1,48,48,3],name="input")

#y = tf.placeholder(tf.float32, shape = [None,48*48],name="output_placeholder")
#

#segNet = net.SegNet()
#mask_pred = segNet(x)
# Simple SegNet

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


conv1 = tf.nn.conv2d(x, conv1_weight, strides=[1, 1, 1, 1], padding='SAME', name='conv')
pool1 = tf.nn.relu(conv1+conv1_bias)
pool1 = tf.nn.max_pool(pool1, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')

kenel2 = conv2_weight# + conv2_bias
conv2 = tf.nn.conv2d(pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME', name='conv2')
pool2 = tf.nn.relu(conv2+conv2_bias)
pool2 = tf.nn.max_pool(pool2, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')

conv3 = tf.nn.conv2d(pool2, conv3_weight, strides=[1, 1, 1, 1], padding='SAME', name='conv2')
conv3 = tf.nn.relu(conv3+conv3_bias)


h_flat = tf.reshape(conv3, [-1, 12 * 12 * 64])  # -1은 batch size를 유지하는 것.
fc4 = (tf.matmul(h_flat, fc4_weight) + fc4_bias)
fc4 = tf.nn.sigmoid(fc4)
print('fc4', fc4)

fc5 = (tf.matmul(fc4, fc5_weight) + fc5_bias)
fc5 = tf.nn.sigmoid(fc5)
print('fc5', fc5)

# h_flat = tf.reshape(fc5, [-1, 12 * 12 * 64])  # -1은 batch size를 유지하는 것.
h = tf.matmul(fc5, fc6_weight) + fc6_bias
# fc6 = tf.nn.sigmoid(h, name="fc6")
output_node = tf.nn.sigmoid(h, name="output")
# mask_pred = segmentizer.inference(x)


model_directory = './weights/'

saver = tf.train.Saver()

init = tf.global_variables_initializer()

with tf.Session() as sess:
#
    sess.run(init)

    print('evaluate...')

    #saver = tf.train.import_meta_graph('model/model.ckpt.meta')
    saver.restore(sess,"model_test2/model.ckpt")
    #saver.restore(sess,tf.train.latest_checkpoint('model'))

    with open(model_directory + "conv1_W.bin", 'wb') as f:
        W_conv1_p = tf.transpose(conv1_weight, perm=[3, 0, 1, 2])
        f.write(sess.run(W_conv1_p).tobytes())
    with open(model_directory + "conv2_W.bin", 'wb') as f:
        W_conv2_p = tf.transpose(conv2_weight, perm=[3, 0, 1, 2])
        f.write(sess.run(W_conv2_p).tobytes())
    with open(model_directory + "conv3_W.bin", 'wb') as f:
        W_conv3_p = tf.transpose(conv3_weight, perm=[3, 0, 1, 2])
        f.write(sess.run(W_conv3_p).tobytes())

    with open(model_directory + "conv1_b.bin", 'wb') as f:
        f.write(sess.run(conv1_bias).tobytes())
    with open(model_directory + "conv2_b.bin", 'wb') as f:
        f.write(sess.run(conv2_bias).tobytes())
    with open(model_directory + "conv3_b.bin", 'wb') as f:
        f.write(sess.run(conv3_bias).tobytes())

    with open(model_directory + "fc4_W.bin", 'wb') as f:
        W_fc4_shp = tf.reshape(fc4_weight, [12, 12, 64, 100])
        W_fc4_p = tf.transpose(W_fc4_shp, perm=[3, 0, 1, 2])
        f.write(sess.run(W_fc4_p).tobytes())
    with open(model_directory + "fc5_W.bin", 'wb') as f:
        W_fc5_shp = tf.reshape(fc5_weight, [1, 1, 100, 400])
        W_fc5_p = tf.transpose(W_fc5_shp, perm=[3, 0, 1, 2])
        f.write(sess.run(W_fc5_p).tobytes())
    with open(model_directory + "fc6_W.bin", 'wb') as f:
        W_fc6_shp = tf.reshape(fc6_weight, [1, 1, 400, 48 * 48])
        W_fc6_p = tf.transpose(W_fc6_shp, perm=[3, 0, 1, 2])
        f.write(sess.run(W_fc6_p).tobytes())

    with open(model_directory + "fc4_b.bin", 'wb') as f:
        f.write(sess.run(fc4_bias).tobytes())
    with open(model_directory + "fc5_b.bin", 'wb') as f:
        f.write(sess.run(fc5_bias).tobytes())
    with open(model_directory + "fc6_b.bin", 'wb') as f:
        f.write(sess.run(fc6_bias).tobytes())

    len  = len(learning_list)
    for i in range(len):
        test_path = 'learning2/'+str(i+1)+".jpg"
        #test_path = 'input.jpg'
        image_q = utils.get_img(learning_list[i], img_size=[48, 48])
        # img = scipy.misc.imread("learning2/"+str(i+1)+".jpg", mode='RGB')
        # image_q = scipy.misc.imresize(img, [48,48],interp="lanczos")
        image_q = np.array(Image.fromarray(image_q.astype(np.uint8)))#.convert('L'))

        #image_q = np.array(Image.fromarray(image_q.astype(np.uint8)))

        image_q = np.reshape(image_q, [-1, 48, 48, 3])

        output = sess.run(output_node,feed_dict={x:image_q})

        output = np.reshape(output, [48, 48])
        array1 = np.array(output)
        array = 255 * array1

        image2 = Image.fromarray(array.astype(np.uint8))  # Image.fromarray(convert.astype(np.uint8))
        original_yuv = np.array(image2)  # .convert('YCbCr'))
        #imsave('eval/restore'+str(i)+".jpg", original_yuv)
        imsave('eval/'+learning_list[i], original_yuv)

        #image2 =np.array(Image.fromarray(input0.astype(np.uint8)))
        #original_yuv = np.array(image2)  # .convert('YCbCr'))
        #imsave('org_developing.jpg', original_yuv)

#print('input',input)