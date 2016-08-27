import slurm_cluster_manager as slm

dictionary, myjob, mytaskid = slm.SlurmClusterManager().build_cluster_spec()
print("++++++++++++++++")
print("CLUSERT SPEC: {}".format(dictionary))
print("\nMyRole: {} [{}]".format(myjob, mytaskid))
print("----------------\n\n")