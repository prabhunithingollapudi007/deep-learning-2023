import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        # avoid truncated tiles
        if (self.resolution) % (self.tile_size * 2) != 0:
            raise ValueError('resolution should be evenly dividable by 2· tile size')
        size = round(self.resolution / (self.tile_size * 2)) # number of tiles per unit black and white
        # create a black tile
        black_tile = np.zeros((self.tile_size, self.tile_size))
        # create a white tile
        white_tile = np.ones((self.tile_size, self.tile_size))
        # create a balck and white 2 x 2 tile
        tile = np.block([[black_tile, white_tile], [white_tile, black_tile]])
        # repeat the 2 x 2 tile to create a black and white checker
        self.output = np.tile(tile, (size, size))
        # return a copy of the output
        return np.copy(self.output)

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()
    

class Circle:

    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position

    def draw(self):
        # create a linear spacing of x and y axis with size as big as resolution
        x = np.linspace(0, self.resolution, self.resolution)
        y = np.linspace(0, self.resolution, self.resolution)
        x_centre, y_centre = self.position
        # get the coordinate axis from meshgrid
        xx, yy = np.meshgrid(x, y, sparse=True)
        # create circle equation
        circle_equation = np.sqrt(((xx - x_centre) ** 2) + ((yy - y_centre) ** 2))
        # make 1 or 0 outside of our radius
        self.output = np.where(circle_equation <= self.radius, 1, 0)
        # return a copy of output
        return np.copy(self.output)

    
    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Spectrum:

    def __init__(self, resolution):
        self.resolution = resolution
    
    def draw(self):
        # top left - blue - rgb - (0, 0, 255)    ---->   top right - red - rgb - (255, 0, 0)
        # bottom - cyan - left rgb - (0, 255, 255) ---->   bottom right - yellow - rgb - (255, 255, 0)
        
        # create dummy image of zeros ( pure black image ), fill the inputs
        image = np.zeros((self.resolution, self.resolution, 3))

        
        increasing_axis = np.linspace(0, 1, self.resolution)
        decreasing_axis = 1 - np.linspace(0, 1, self.resolution)

        # observe that R is increasing and B is decreasing along horizontal direction
        image[:, :, 0] += increasing_axis
        image[:, :, 2] += decreasing_axis

        # observe that G is increaseing along vertical axis 
        increasing_axis_vertical = np.expand_dims(increasing_axis, axis=1)
        image[:, :, 1] += increasing_axis_vertical

        self.output = image
        # return a copy of output
        return np.copy(self.output)

    def show(self):
        plt.imshow(self.output)
        plt.show()
