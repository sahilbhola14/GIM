import numpy as np

original_data_spatial_res = 100
num_sub_sample_points = 10

sub_sample_points = np.arange(0, original_data_spatial_res)
np.random.shuffle(sub_sample_points)
sub_sample_points = sub_sample_points[:num_sub_sample_points]

np.save("sub_sample_idx_array_num_{}_spatial_res_{}.npy".format(num_sub_sample_points, original_data_spatial_res), sub_sample_points)
