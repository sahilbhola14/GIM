import numpy as np
import os
import shutil


class linear_gaussian:
    def __init__(self, true_theta=np.arange(1, 4), spatial_res=100):
        self.spatial_res = spatial_res
        self.xtrain = np.linspace(-1, 1, self.spatial_res)
        self.true_theta = true_theta
        self.num_parameters = self.true_theta.shape[0]
        self.vm = self.compute_vm()

    def compute_vm(self):
        vm = np.tile(self.xtrain[:, None], (1, self.num_parameters))
        vm = np.cumprod(vm, axis=1)
        return vm

    def compute_prediction(self, theta):
        return self.vm @ theta.reshape(-1, 1)


def main():
    # Begin User Input
    true_theta = np.array([1, 2, 3])
    spatial_res = 100
    # End User Input
    model = linear_gaussian(true_theta=true_theta, spatial_res=spatial_res)
    prediction = model.compute_prediction(true_theta)

    data = np.zeros((spatial_res, 2))
    data[:, 0] = model.xtrain
    data[:, 1] = prediction[:, 0]

    # Create the folders
    save_path = "./Output"

    if os.path.exists(save_path) is True:
        shutil.rmtree(save_path)

    os.mkdir(save_path)

    data_save_path = os.path.join(save_path, "true_output.npy")
    np.save(data_save_path, data)


if __name__ == "__main__":
    main()
