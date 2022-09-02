import os
import numpy as np
import tensorflow as tf
import input_data
import model
import argparse
tf.reset_default_graph()

# you need to change the directories to yours.
train_logs_dir = '/output/ckpt/'
val_logs_dir = '/output/val'


# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='tensorflow Training Example')
parser.add_argument('--train_dataset',required=True, help='layer of resnet') #resnet的结构选择
parser.add_argument('--trainval_dataset',required=True, help='layer of resnet') #resnet的结构选择
parser.add_argument('--output', default='/output', help='folder to output images and model checkpoints') #输出结果保存路径
args = parser.parse_args()

# 获取命令行参数
train_dir=args.train_dataset
test_dir=args.trainval_dataset



N_CLASSES = 2
IMG_W = 208     # resize the image, if the input image is too large, training will be very slow.
IMG_H = 208
RATIO = 0.2     # take 20% of dataset as validation data
BATCH_SIZE = 64
CAPACITY = 2000
MAX_STEP = 300 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001


def training():

    train, train_label, val, val_label = input_data.get_files(train_dir, RATIO)
    train_batch, train_label_batch = input_data.get_batch(train,
                                                  train_label,
                                                  IMG_W,
                                                  IMG_H,
                                                  BATCH_SIZE,
                                                  CAPACITY)
    val_batch, val_label_batch = input_data.get_batch(val,
                                                  val_label,
                                                  IMG_W,
                                                  IMG_H,
                                                  BATCH_SIZE,
                                                  CAPACITY)

    logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    loss = model.losses(logits, train_label_batch)
    train_op = model.trainning(loss, learning_rate)
    acc = model.evaluation(logits, train_label_batch)

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE])
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(train_logs_dir, sess.graph)
        val_writer = tf.summary.FileWriter(val_logs_dir, sess.graph)

        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                        break
                tra_images,tra_labels = sess.run([train_batch, train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, acc],
                                                feed_dict={x:tra_images, y_:tra_labels})
                if step % 50 == 0:
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0), flush=True)
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)
                    
                if step % 200 == 0 or (step + 1) == MAX_STEP:
                    val_images, val_labels = sess.run([val_batch, val_label_batch])
                    val_loss, val_acc = sess.run([loss, acc],
                                                 feed_dict={x:val_images, y_:val_labels})
                    print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc*100.0), flush=True)
                    summary_str = sess.run(summary_op)
                    val_writer.add_summary(summary_str, step)

                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(train_logs_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
        
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached', flush=True)
        finally:
            coord.request_stop()
        coord.join(threads)


#Test one image

def get_one_image(file_dir):
    """
    Randomly pick one image from test data
    Return: ndarray
    """
    test =[]
    for file in os.listdir(file_dir):
        print(file_dir + file)
        test.append(file_dir + file)
    print('There are %d test pictures\n' %(len(test)),flush=True)

    n = len(test)
    ind = np.random.randint(0, n)
    print(ind)
    img_test = test[0]
    from PIL import Image
    image = Image.open(img_test)
    image = image.resize([208, 208])
    image = np.array(image)
    return image

def save_model():
    """
    Test one image with the saved models and parameters
    """

    test_image = get_one_image(test_dir)

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2

        image = tf.cast(test_image, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3],name='input')
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[208, 208, 3])

        saver = tf.train.Saver()
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_g)
            sess.run(init_l)
            builder = tf.saved_model.builder.SavedModelBuilder('/output/model')
            tensor_info_x = tf.saved_model.utils.build_tensor_info(image) # 输入
            tensor_info_y = tf.saved_model.utils.build_tensor_info(logit) # 输出

            prediction_signature = (
              tf.saved_model.signature_def_utils.build_signature_def(
                  inputs={'input_0': tensor_info_x},
                  outputs={'output_0': tensor_info_y},
                  method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
              sess, [tf.saved_model.tag_constants.SERVING],
              signature_def_map={
                  'predict_images':
                      prediction_signature,
              },
              legacy_init_op=legacy_init_op)

            builder.save() 


def test_one_image():
    """
    Test one image with the saved models and parameters
    """

    test_image = get_one_image(test_dir)

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2

        image = tf.cast(test_image, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[208, 208, 3])

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(train_logs_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: test_image})
            max_index = np.argmax(prediction)
            if max_index==0:
                print('This is a cat with possibility %.6f' %prediction[:, 0])
            else:
                print('This is a dog with possibility %.6f' %prediction[:, 1])


if __name__ == '__main__':
    training()
    save_model()

