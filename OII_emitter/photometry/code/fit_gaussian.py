import numpy as np
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.visualization import ZScaleInterval, ImageNormalize
import matplotlib.pyplot as plt

class gaussian_2D:
    def __init__(self, data, x_init=(0, False), y_init=(0, False), x_stddev_init=(1, False), y_stddev_init=(1, False), amplitude_init=(1, False), theta_init=(0, True)):
        self.data = data

        self.init_model = models.Gaussian2D()
        self.init_model.x_mean, self.init_model.x_mean.fixed = x_init
        self.init_model.y_mean, self.init_model.y_mean.fixed = y_init
        self.init_model.x_stddev, self.init_model.x_stddev.fixed = x_stddev_init
        self.init_model.y_stddev, self.init_model.y_stddev.fixed = y_stddev_init
        self.init_model.amplitude, self.init_model.amplitude.fixed = amplitude_init
        self.init_model.theta, self.init_model.theta.fixed = theta_init

        self.fitting()

    def __profile_2D(self, data, mode='mean'):
        nx, ny = data.shape
        if mode == 'mean':
            x_marg = [np.mean(data[i, :]) for i in range(data.shape[0])]
            y_marg = [np.mean(data[:, j]) for j in range(data.shape[1])]
        elif mode == 'cut':
            if nx%2 == 0: x_marg = data[(nx//2-1):(nx//2+1)].mean(axis=0)
            else: x_marg = data[nx//2]
            if ny%2 == 0: y_marg = data[:, (ny//2-1):(ny//2+1)].mean(axis=1)
            else: y_marg = data[:, ny//2]
        return np.array(x_marg), np.array(y_marg)

    def fitting(self, ):
        fig_g = fitting.LevMarLSQFitter()
        self.y, self.x = np.mgrid[:self.data.shape[0], :self.data.shape[1]]
        self.result_model = fig_g(self.init_model, self.x, self.y, self.data)

    def plot_img_profile(self, name=None, extent=None, x_label=None, y_label=None, mode='mean'):
        img = self.data
        model = self.result_model(self.x, self.y)
        fig = plt.figure(figsize=(6, 6))
        gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.05)
        ax_img = fig.add_subplot(gs[1, 0])
        ax_x = fig.add_subplot(gs[0, 0])
        ax_y = fig.add_subplot(gs[1, 1])
        norm = ImageNormalize(img, interval=ZScaleInterval())
        if extent: ax_img.imshow(img, extent=extent, origin='lower')
        else: ax_img.imshow(img, origin='lower')
        xmin, xmax = ax_img.get_xlim()
        ymin, ymax = ax_img.get_ylim() 
        if x_label: ax_img.set_xlabel(x_label)
        if y_label: ax_img.set_ylabel(y_label)
        if name: ax_img.text(0.95*xmax, 0.95*ymin, name, color='white', ha='right', va='bottom', fontstyle='italic', fontweight='bold')
        array_x_data, array_y_data = self.__profile_2D(img, mode=mode)
        array_x_model, array_y_model = self.__profile_2D(model, mode=mode)
        array_x_res, array_y_res = self.__profile_2D(img-model, mode=mode)
        ax_x.plot(array_x_data, 'b-', label='data')
        ax_x.plot(array_x_model, 'r--', label='model')
        ax_x.plot(array_x_res, 'k-', label='residual')
        ax_x.set_xticks([])
        ax_x.legend()
        ax_y.plot(array_y_data, np.arange(len(array_y_data)), 'b-', label='data')
        ax_y.plot(array_y_model, np.arange(len(array_y_model)), 'r--', label='model')
        ax_y.plot(array_y_res, np.arange(len(array_y_res)), 'k-', label='residual')
        ax_y.set_yticks([])
        plt.show()

    def plot_fitting_result(self):
        plt.figure(figsize=(8, 2.5))
        plt.subplot(1, 3, 1)
        plt.imshow(self.data, origin='lower', interpolation='nearest',)
        plt.title("Data")
        plt.subplot(1, 3, 2)
        plt.imshow(self.result_model(self.x, self.y), origin='lower', interpolation='nearest',)
        plt.title("Model")
        plt.subplot(1, 3, 3)
        plt.imshow(self.data - self.result_model(self.x, self.y), origin='lower', interpolation='nearest',)
        plt.colorbar()
        plt.title("Residual")
        plt.show()

    def get_fwhm(self, stddev):
        return stddev * 2 * np.sqrt(2 * np.log(2))