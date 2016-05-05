import tensorflow as tf

def main(_):

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"local": ["localhost:2222"]})
  
  # Create and start a server for the local task.
  server = tf.train.Server(cluster, job_name="local", task_index=0)

  # Build model...
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.nn.softmax(tf.matmul(x, W) + b)
  y_ = tf.placeholder(tf.float32, [None, 10])
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  init = tf.initialize_all_variables()
  sess = tf.Session(server.target)
  sess.run(init)
  for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == "__main__":
  tf.app.run()
