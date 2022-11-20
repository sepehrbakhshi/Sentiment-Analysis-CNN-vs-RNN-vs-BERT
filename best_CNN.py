
import numpy as np
import logging
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split

def new_conv_layer(input, num_input_channels, filter_size, num_filters, name):
    with tf.variable_scope(name) as scope:
        shape1 = [filter_size, filter_size, num_input_channels, num_filters]
        weights = tf.Variable(tf.truncated_normal(shape1, stddev=0.05))
        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
        layer += biases
        return layer, weights

def new_batchnorm_layer(input, name):
    with tf.variable_scope(name) as scope:
        layer = tf.layers.batch_normalization(input)
        return layer

def new_dropout_layer(input, r, name):
    with tf.variable_scope(name) as scope:
        layer = tf.nn.dropout(input, rate=r)
        return layer

def new_pool_layer(input, name):
    with tf.variable_scope(name) as scope:
        layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return layer

def new_relu_layer(input, name):
    with tf.variable_scope(name) as scope:
        layer = tf.nn.relu(input)
        return layer

def new_fc_layer(input, num_inputs, num_outputs, name):
    with tf.variable_scope(name) as scope:
        weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
        biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
        layer = tf.matmul(input, weights) + biases
        return layer

dim = 1
MAX_TWEET_LENGTH = 25
EMBEDDING_SIZE = 25

if(__name__ == '__main__'):

    x_train = np.load('twitter_train.npy').reshape(-1,MAX_TWEET_LENGTH * EMBEDDING_SIZE,dim)
    y_train_ = np.load('train_labels_twitter.npy')
    x_valid = np.load('twitter_test.npy').reshape(-1,MAX_TWEET_LENGTH * EMBEDDING_SIZE,dim)
    y_valid_ = np.load('test_labels_twitter.npy')
    x_test = np.load('twitter_test.npy').reshape(-1,MAX_TWEET_LENGTH * EMBEDDING_SIZE,dim)
    y_test_ = np.load('test_labels_twitter.npy')

    y_true_ = tf.placeholder(tf.float32, shape=[None, 3], name='Label')
    y_true = tf.argmax(y_true_, dimension=1)
    x = tf.placeholder(tf.float32, shape=[None, MAX_TWEET_LENGTH*EMBEDDING_SIZE,1], name='Matching_Matrix')
    x_reshaped = tf.reshape(x, [-1, MAX_TWEET_LENGTH, EMBEDDING_SIZE, 1])
    dropout_rate = tf.placeholder_with_default(0.5, shape=())

    conv_1, weights_conv1 = new_conv_layer(input=x_reshaped, num_input_channels=1, filter_size=5, num_filters=256, name ="conv1")
    relu_1 = new_relu_layer(conv_1, name="relu1")
    #dropout_3 = new_dropout_layer(relu_1, r=dropout_rate, name="dropout3")
    maxpool_1 = new_pool_layer(relu_1, name="maxpool1")
    bn_1 = new_batchnorm_layer(maxpool_1, name="batchnorm_1")
    conv_2, weights_conv2 = new_conv_layer(input=bn_1, num_input_channels=256, filter_size=3, num_filters=128, name ="conv2")
    relu_2 = new_relu_layer(conv_2, name="relu2")
    #dropout_1 = new_dropout_layer(relu_2, r=dropout_rate, name="dropout1")

    maxpool_2 = new_pool_layer(relu_2, name="maxpool2")
    bn_2 = new_batchnorm_layer(maxpool_2, name="batchnorm_2")
    conv_3, weights_conv3 = new_conv_layer(input=bn_2, num_input_channels=128, filter_size=3, num_filters=64, name ="conv3")
    relu_3 = new_relu_layer(conv_3, name="relu3")
    #dropout_3 = new_dropout_layer(relu_3, r=dropout_rate, name="dropout3")
    maxpool_3 = new_pool_layer(relu_3, name="maxpool3")
    bn_3 = new_batchnorm_layer(maxpool_3, name="batchnorm_3")

    conv_4, weights_conv4 = new_conv_layer(input=bn_3, num_input_channels=64, filter_size=3, num_filters=32, name ="conv4")
    relu_4 = new_relu_layer(conv_4, name="relu4")
    #maxpool_3 = new_pool_layer(relu_3, name="maxpool3")

    num_features = relu_4.get_shape()[1:64].num_elements()
    flat = tf.reshape(relu_4, [-1, num_features])

    #dropout_2 = new_dropout_layer(flat, r=dropout_rate, name="dropout2")
    fc_0 = new_fc_layer(flat, num_inputs=num_features, num_outputs=128, name="fc0")
    relu_0 = new_relu_layer(fc_0, name="relu0")
    dropout_1 = new_dropout_layer(relu_0, r=dropout_rate, name="dropout1")
    fc_1 = new_fc_layer(dropout_1, num_inputs=128, num_outputs=64, name="fc1")
    relu_4 = new_relu_layer(fc_1, name="relu4")
    dropout_2 = new_dropout_layer(relu_4, r=dropout_rate, name="dropout2")
    fc_2 = new_fc_layer(dropout_2, num_inputs=64, num_outputs=3, name="fc1")
    """
    relu_5 = new_relu_layer(fc_2, name="relu4")
    dropout_3 = new_dropout_layer(relu_5, r=dropout_rate, name="dropout2")
    fc_3 = new_fc_layer(dropout_3, num_inputs=32, num_outputs=2, name="fc2")
    """
    with tf.variable_scope("Softmax"):
        y_pred = tf.nn.softmax(fc_2)
        y_pred_cls = tf.cast(tf.argmax(y_pred, dimension=1), tf.float32)

    with tf.name_scope("cost"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc_2, labels=y_true_)
        cost = tf.reduce_mean(cross_entropy)

    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
        #optimizer = tf.train.AdagradOptimizer(learning_rate=1e-2).minimize(cost)
    with tf.name_scope("Result"):
        Result = flat
    with tf.name_scope("accuracy"):
        y_pred_cls = tf.cast(tf.argmax(y_pred, dimension=1), tf.float32)
        y_true_cls = tf.cast(tf.argmax(y_true_, dimension=1), tf.float32)
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.name_scope("f1"):
        y_true_cls = tf.cast(tf.argmax(y_true_, dimension=1), tf.float32)
        y_pred_cls = tf.cast(tf.argmax(y_pred, dimension=1), tf.float32)
        TP = tf.count_nonzero(y_pred_cls * y_true_cls)
        TN = tf.count_nonzero((y_pred_cls-1) * (y_true_cls-1))
        FP = tf.count_nonzero((y_pred_cls) * (y_true_cls-1))
        FN = tf.count_nonzero((y_pred_cls-1) * (y_true_cls))
        precision = tf.divide (TP, (TP + FP))
        recall = tf.divide(TP, (TP + FN))
        f1 = (tf.divide((2 * precision * recall), (precision + recall)))

    with tf.name_scope("True_Positive"):
        y_pred_cls = tf.cast(tf.argmax(y_pred, dimension=1), tf.float32)
        y_true_cls = tf.cast(tf.argmax(y_true_, dimension=1), tf.float32)
        True_Positive = tf.count_nonzero(y_pred_cls * y_true_cls)

    with tf.name_scope("True_Negative"):
        y_pred_cls = tf.cast(tf.argmax(y_pred, dimension=1), tf.float32)
        y_true_cls = tf.cast(tf.argmax(y_true_, dimension=1), tf.float32)
        True_Negative = tf.count_nonzero((y_pred_cls-1) * (y_true_cls-1))

    with tf.name_scope("False_Positive"):
        y_pred_cls = tf.cast(tf.argmax(y_pred, dimension=1), tf.float32)
        y_true_cls = tf.cast(tf.argmax(y_true_, dimension=1), tf.float32)
        False_Positive = tf.count_nonzero((y_pred_cls) * (y_true_cls-1))

    with tf.name_scope("False_Negative"):
        y_pred_cls = tf.cast(tf.argmax(y_pred, dimension=1), tf.float32)
        y_true_cls = tf.cast(tf.argmax(y_true_, dimension=1), tf.float32)
        False_Negative = tf.count_nonzero((y_pred_cls-1) * (y_true_cls))

    writer = tf.summary.FileWriter("Training_FileWriter/")
    writer1 = tf.summary.FileWriter("Validation_FileWriter/")
    tf.summary.scalar('loss', cost)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary = tf.summary.merge_all()
    num_epochs = 30
    batch_size = 2
    train_acc_record=np.empty(num_epochs)
    valid_acc_record=np.empty(num_epochs)
    train_loss_record=np.empty(num_epochs)
    valid_loss_record=np.empty(num_epochs)

    with tf.Session() as sess:

        init_l = tf.local_variables_initializer()
        sess.run(init_l)
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        for epoch in range(num_epochs):

            start_time = time.time()
            train_accuracy = 0
            train_loss = 0
            valid_accuracy = 0
            valid_loss = 0
            train_f1 = 0
            valid_f1 = 0

            tp = 0.0
            tn = 0.0
            fp = 0.0
            fn = 0.0

            precision = 0.0
            recall = 0.0

            for batch in range(0, int(len(y_train_)/batch_size)):

                x_batch = x_train[batch*batch_size: (batch+1)*batch_size]
                y_true_batch = y_train_[batch*batch_size: (batch+1)*batch_size]
                feed_dict_train = {x: x_batch, y_true_: y_true_batch}
                sess.run(optimizer, feed_dict=feed_dict_train)
                train_accuracy += sess.run(accuracy, feed_dict=feed_dict_train)
                train_loss += sess.run(cost, feed_dict=feed_dict_train)
                tp += sess.run(True_Positive, feed_dict=feed_dict_train)
                tn += sess.run(True_Negative, feed_dict=feed_dict_train)
                fp += sess.run(False_Positive, feed_dict=feed_dict_train)
                fn += sess.run(False_Negative, feed_dict=feed_dict_train)
                summ = sess.run(merged_summary, feed_dict=feed_dict_train)
                result = sess.run(Result, feed_dict=feed_dict_train)
                writer.add_summary(summ, epoch*int(len(y_train_)/batch_size) + batch)

            train_accuracy /= int(len(y_train_)/batch_size)
            train_loss /= int(len(y_train_)/batch_size)
            precision = tf.divide (TP, (TP + FP))
            recall = tf.divide(TP, (TP + FN))
            train_f1 = ((2 * precision * recall) / (precision + recall)).eval(feed_dict=feed_dict_train)
            tp = 0.0
            tn = 0.0
            fp = 0.0
            fn = 0.0

            for valid_batch in range(0, int(len(y_valid_)/batch_size)):

                x_batch = x_valid[valid_batch*batch_size: (valid_batch+1)*batch_size]
                y_true_batch = y_valid_[valid_batch*batch_size: (valid_batch+1)*batch_size]
                feed_dict_valid = {x: x_batch, y_true_: y_true_batch, dropout_rate: 0.0}
                valid_accuracy += sess.run(accuracy, feed_dict=feed_dict_valid)
                valid_loss += sess.run(cost, feed_dict=feed_dict_valid)
                tp += sess.run(True_Positive, feed_dict=feed_dict_valid)
                tn += sess.run(True_Negative, feed_dict=feed_dict_valid)
                fp += sess.run(False_Positive, feed_dict=feed_dict_valid)
                fn += sess.run(False_Negative, feed_dict=feed_dict_valid)
                summ = sess.run(merged_summary, feed_dict=feed_dict_valid)
                writer.add_summary(summ, epoch*int(len(y_valid_)/batch_size) + batch)

            valid_accuracy /= int(len(y_valid_)/batch_size)
            valid_loss /= int(len(y_valid_)/batch_size)
            precision = tf.divide (TP, (TP + FP))
            recall = tf.divide(TP, (TP + FN))
            valid_f1 = ((2 * precision * recall) / (precision + recall)).eval(feed_dict=feed_dict_valid)
            tp = 0.0
            tn = 0.0
            fp = 0.0
            fn = 0.0
            #summ, valid_accuracy, valid_loss = sess.run([merged_summary, accuracy, cost], feed_dict={x: x_valid, y_true_: y_valid_})
            writer1.add_summary(summ, epoch)
            end_time = time.time()

            train_acc_record[epoch] = train_accuracy
            valid_acc_record[epoch] = valid_accuracy
            train_loss_record[epoch] = train_loss
            valid_loss_record[epoch] = valid_loss

            print("Epoch "+str(epoch+1)+" completed : Time usage "+str(int(end_time-start_time))+" seconds")
            print ("\t- Training Accuracy:\t{}".format(train_accuracy))
            print ("\t- Validation Accuracy:\t{}".format(valid_accuracy))
            print ("\t- Training Loss:\t{}".format(train_loss))
            print ("\t- Validation Loss:\t{}".format(valid_loss))
            print ("\t- Training F1-Score:\t{}".format(train_f1))
            print ("\t- Validation F1-Score:\t{}".format(valid_f1))
            #print("----------------")
            #print(result.shape)

            test_accuracy = 0.0
            test_loss = 0.0
            test_f1 = 0.0
            tp = 0.0
            tn = 0.0
            fp = 0.0
            fn = 0.0

            for test_batch in range(0, int(len(y_test_)/batch_size)):

                x_batch = x_test[test_batch*batch_size: (test_batch+1)*batch_size]
                y_true_batch = y_test_[test_batch*batch_size: (test_batch+1)*batch_size]
                feed_dict_test = {x: x_batch, y_true_: y_true_batch, dropout_rate: 0.0}
                test_accuracy += sess.run(accuracy, feed_dict=feed_dict_test)
                test_loss += sess.run(cost, feed_dict=feed_dict_test)
                tp += sess.run(True_Positive, feed_dict=feed_dict_test)
                tn += sess.run(True_Negative, feed_dict=feed_dict_test)
                fp += sess.run(False_Positive, feed_dict=feed_dict_test)
                fn += sess.run(False_Negative, feed_dict=feed_dict_test)
                summ = sess.run(merged_summary, feed_dict=feed_dict_test)
                writer.add_summary(summ, epoch*int(len(y_valid_)/batch_size) + batch)

            test_accuracy /= int(len(y_valid_)/batch_size)
            test_loss /= int(len(y_valid_)/batch_size)
            precision = tf.divide (TP, (TP + FP))
            recall = tf.divide(TP, (TP + FN))
            test_f1 = ((2 * precision * recall) / (precision + recall)).eval(feed_dict=feed_dict_test)

            #summ, test_accuracy, test_loss, test_f1 = sess.run([merged_summary, accuracy, cost, f1], feed_dict={x: x_test, y_true_: y_test_, dropout_rate: 0})
            print ("\t- Test Accuracy:\t{}".format(test_accuracy))
            print ("\t- Test Loss:\t{}".format(test_loss))
            print ("\t- Test F1-Score:\t{}".format(test_f1))

        test_accuracy = 0.0
        test_loss = 0.0
        test_f1 = 0.0
        tp = 0.0
        tn = 0.0
        fp = 0.0
        fn = 0.0

        for test_batch in range(0, int(len(y_test_)/batch_size)):

            x_batch = x_test[test_batch*batch_size: (test_batch+1)*batch_size]
            y_true_batch = y_test_[test_batch*batch_size: (test_batch+1)*batch_size]
            feed_dict_test = {x: x_batch, y_true_: y_true_batch, dropout_rate: 0.0}
            test_accuracy += sess.run(accuracy, feed_dict=feed_dict_test)
            test_loss += sess.run(cost, feed_dict=feed_dict_test)
            tp += sess.run(True_Positive, feed_dict=feed_dict_test)
            tn += sess.run(True_Negative, feed_dict=feed_dict_test)
            fp += sess.run(False_Positive, feed_dict=feed_dict_test)
            fn += sess.run(False_Negative, feed_dict=feed_dict_test)
            summ = sess.run(merged_summary, feed_dict=feed_dict_test)
            writer.add_summary(summ, epoch*int(len(y_valid_)/batch_size) + batch)

        test_accuracy /= int(len(y_valid_)/batch_size)
        test_loss /= int(len(y_valid_)/batch_size)
        precision = tf.divide (TP, (TP + FP))
        recall = tf.divide(TP, (TP + FN))
        test_f1 = ((2 * precision * recall) / (precision + recall)).eval(feed_dict=feed_dict_test)

        #summ, test_accuracy, test_loss, test_f1 = sess.run([merged_summary, accuracy, cost, f1], feed_dict={x: x_test, y_true_: y_test_, dropout_rate: 0})
        print("Training Finished.")
        print ("\t- Test Accuracy:\t{}".format(test_accuracy))
        print ("\t- Test Loss:\t{}".format(test_loss))
        print ("\t- Test F1-Score:\t{}".format(test_f1))
        np.savetxt('train_acc_tmp.txt', train_acc_record)
        np.savetxt('valid_acc_tmp.txt', valid_acc_record)
        np.savetxt('train_loss_tmp.txt', train_loss_record)
        np.savetxt('valid_loss_tmp.txt', valid_loss_record)
        #plot_acc(train_acc_record, valid_acc_record)
        #plot_error(train_loss_record, valid_loss_record)
