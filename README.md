**Paper drafts can be found in the [wiki](http://pccgit.cs.byu.edu/tetchart/modularDNN_Practice/wikis/home).**

### Design chart can be seen here:
https://www.lucidchart.com/documents/view/d52e3617-3df2-46df-b3f1-ee42f73cf476

## Requires:
- Server Client messages are handled via `ZMQ`.
- Networks are built using `Keras`.
 - Keras uses `theano` or `tensorflow`
 - Obviously, Nvidia `CUDA` and `cuDNN` are also required

# Potential Errors
### If you are running out of GPU memory when running multiple simulators
Both Theano and Tensorflow default to claiming all of the GPU to utilize the memory more efficiently.
To get around this:
 - If using theano, make sure you set `cnmem` to `0` (false) with the code below for `~/.theanorc`
 - If using tensorflow, you need to somehow do this correction.
  - I'm not sure how to do this in keras, so for now just use theano instead of TF


##### ~/.theanorc
```
[global]
floatX = float32
device = gpu
[lib]
cnmem = 0
```

##### Tensorflow GPU limit:
```
# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
```