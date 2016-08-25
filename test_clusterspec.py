from __future__ import print_function
import os
import subprocess
import shlex
import slurm_tf_man as slm

orighostname = os.environ["HOSTNAME"]

print("original hostname: {}\n\n".format(orighostname))

os.environ['SLURM_NPROCS'] =  str(3)
os.environ['SLURM_NODELIST'] =  "m8g-3-[1-2,4]"

# TEST Comp1
os.environ["HOSTNAME"] = 'm8g-3-1'
os.environ['SLURM_PROCID'] =  str(0)
res1 = slm.build_cluster_spec()
assert res1[1] == 'ps'
assert res1[2] == 0

# test comp2
os.environ["HOSTNAME"] = 'm8g-3-2'
os.environ['SLURM_PROCID'] =  str(1)
res2 = slm.build_cluster_spec()
assert res2[1] == 'worker'
assert res2[2] == 0

# test comp3
os.environ["HOSTNAME"] = 'm8g-3-4'
os.environ['SLURM_PROCID'] =  str(2)
res3 = slm.build_cluster_spec()
assert res3[1] == 'worker'
assert res3[2] == 1

for k1, k2, k3 in zip(res1[0], res2[0], res3[0]):
    for v1, v2, v3 in zip(k1, k2, k3):
        assert v1 == v2
        assert v2 == v3
print (res1[0])

os.environ["HOSTNAME"] = orighostname