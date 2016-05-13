import tensorflow as tf
import numpy as np

gpu_options = tf.GPUOptions(
    per_process_gpu_memory_fraction=0.01,
    # allow_growth=True,
    # deferred_deletion_bytes=1,
)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Create a variable
w1 = tf.Variable(tf.random_normal([1000,1000], stddev=0.01, dtype=tf.float32), name="test_variable")

#initialize the variable
sess.run(tf.initialize_all_variables())

new_value_array = np.zeros((1000,1000), dtype=np.float32)

print new_value_array
for i in range(3000):
    print "Assigning i:{}".format(i)
    sess.run(w1.assign(new_value_array))
