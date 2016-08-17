import tensorflow as tf

sess = tf.Session()
v1 = tf.Variable(1)
v2 = tf.Variable(2)
v3 = tf.Variable(0)

updatev2 = v2.assign_add(v1)
with tf.control_dependencies([updatev2]):
    updatev3 = v3.assign(v2)

sess.run(tf.initialize_all_variables())

res = sess.run(updatev3)
assert 3 == res
print "\n\nlooks like it called updatev2 because of control dependencies! (3 == {})".format(res)

