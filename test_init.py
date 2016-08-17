import tensorflow as tf

v1 = tf.Variable([1])
v2 = tf.Variable([2])
v3 = tf.Variable([3])
v4 = tf.Variable([4])
v5 = tf.Variable([5])
v6 = tf.Variable([6])

sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(tf.initialize_variables([v1,v2,v3]))
sess2.run(tf.initialize_variables([v4,v5,v6]))

print v1.eval(sess1)
print v2.eval(sess1)
print v3.eval(sess1)
print v4.eval(sess2)
print v5.eval(sess2)
print v6.eval(sess2)


# print v6.eval(sess1) # Can't use with a session that hasn't initialized it.
sess1.run(tf.initialize_variables([v6]))
print v6.eval(sess1)
print v6.eval(sess2)

print ""

sess1.run([v6.assign_add([1])])
sess2.run([v6.assign_add([2])])

print v6.eval(sess1)
print v6.eval(sess2)

sess1.run([v6.assign_add(v6.eval(sess2))]) # meh, seems ok...

print v6.eval(sess1)
print v6.eval(sess2)

