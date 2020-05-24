import numpy as np
import cv2
import os
import sys

class IMAGE(object):

    def __init__(self, path_to_image, name_of_image, gray=False):
        self.valid_image = self.check_valid_extension(name_of_image)
        if self.valid_image != True:
            print("'{}' is not an image or does not have valid image type".format(name_of_image))
            return None
        self.gray = gray
        self.read_path = str(path_to_image)
        self.name = str(name_of_image)
        if gray == True:
            self.image = cv2.imread(str(path_to_image) + '/' + str(name_of_image))
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.n_H, self.n_W = self.image.shape
        else:
            self.image = cv2.imread(str(path_to_image) + str(name_of_image))
            self.n_H, self.n_W, self.n_C = self.image.shape

    def check_valid_extension(self, name_of_image):
        self.name = str(name_of_image[:-4])
        self.extension = str(name_of_image[-4:])
        extension_types_list = self.define_extension_types()
        if self.extension in extension_types_list:
            return True
        else:
            return False

    def define_extension_types(self):
        extension_types = ['.png', '.jpg']
        return extension_types

    '''
    Generic image processing functions performed 'in-place'
    '''

    def add_gaussian_noise(self, mean=0, std=1, scale=100):
        if (self.gray == True):
            noise = scale * np.random.rand(self.n_H, self.n_W)
            img_with_noise = self.image + noise
            self.image = img_with_noise
            self.name = self.name + '_noised'
        else:
            noise = scale * np.random.rand(self.n_H, self.n_W, self.n_C)
            img_with_noise = self.image + noise
            self.image = img_with_noise
            self.name = self.name + '_noised'

    def add_gaussian_blur(self, kernel_size=(5, 5)):
        image_with_blur = cv2.GaussianBlur(self.image, kernel_size, cv2.BORDER_DEFAULT)
        self.image = image_with_blur
        self.name = self.name + '_blurred'

    def write_to_file(self, path_to_write):
        cv2.imwrite(path_to_write, self.image)

    def shape(self):
        return (self.image.shape)

    def label(self, name):
        labels = {"vinci": 64, "vanson": 128}
        if name in labels.keys():
            self.image[:8, :8] = labels[name]
        else:
            print("No such label, choose from: {}".format(labels.keys()))

    '''
    Other Augmentation techniques
    '''

    def fliplr(self):
        flipped_image = np.fliplr(self.image)
        self.image = flipped_image
        self.name = self.name + '_flippedlr'

    def flipud(self):
        flipped_image = np.flipud(self.image)
        self.image = flipped_image
        self.name = self.name + '_flippedud'

    '''
    To ensure images are of expected size and extract subsections of desired sizes
    '''

    def check_ratios_to_expected(self, n_H_expected, n_W_expected):
        r_H = self.n_H / (1.0 * n_H_expected)
        r_W = self.n_W / (1.0 * n_W_expected)
        return r_H, r_W

    def assert_sufficiently_sized(self, n_H_expected, n_W_expected):
        r_H, r_W = self.check_ratios_to_expected(n_H_expected=n_H_expected, n_W_expected=n_W_expected)
        if (r_H != 1 or r_W != 1):
            if r_H < r_W:
                self.re_size(n_H_new=n_H_expected, n_W_new=int(self.n_W / (1.0 * r_H)))
            elif r_W < r_H:
                self.re_size(n_H_new=int(self.n_H / (1.0 * r_W)), n_W_new=n_W_expected)
            else:
                self.re_size(n_H_new=n_H_expected, n_W_new=n_W_expected)
        assert (self.n_H >= n_H_expected and self.n_W >= n_W_expected)

    def save_left_centre_right(self, folder_to_save, n_H_expected, n_W_expected):
        self.assert_sufficiently_sized(n_H_expected=n_H_expected, n_W_expected=n_W_expected)
        r_H, r_W = self.check_ratios_to_expected(n_H_expected=n_H_expected, n_W_expected=n_W_expected)
        if r_H > r_W:
            t_i = 0
            c_i = self.n_H // 2 - n_H_expected // 2
            b_i = self.n_H - n_H_expected
            self.save_section(t_i, 0, folder_to_save, n_H=n_H_expected, n_W=n_W_expected, save_name=self.name + '_t')
            self.save_section(c_i, 0, folder_to_save, n_H=n_H_expected, n_W=n_W_expected, save_name=self.name + '_c')
            self.save_section(b_i, 0, folder_to_save, n_H=n_H_expected, n_W=n_W_expected, save_name=self.name + '_b')
        else:
            l_i = 0
            c_i = self.n_W // 2 - n_W_expected // 2
            r_i = self.n_W - n_W_expected
            self.save_section(0, l_i, folder_to_save, n_H=n_H_expected, n_W=n_W_expected, save_name=self.name + '_l')
            self.save_section(0, c_i, folder_to_save, n_H=n_H_expected, n_W=n_W_expected, save_name=self.name + '_c')
            self.save_section(0, r_i, folder_to_save, n_H=n_H_expected, n_W=n_W_expected, save_name=self.name + '_r')

    '''
    Methods to resize and save sub-sections; singly or strided
    '''

    def re_size(self, n_H_new=8192, n_W_new=8192):
        # Note that resizing uses the opposite dimensionality convensions for some silly reason
        self.image = cv2.resize(self.image, (n_W_new, n_H_new))
        self.n_H, self.n_W = n_H_new, n_W_new

    def save_section(self, i_H, i_W, folder_to_save, n_H=1024, n_W=1024, save_name=None):
        sub_image = self.image[i_H: i_H + n_H, i_W: i_W + n_W]
        if save_name == None:
            cv2.imwrite(str(folder_to_save) + '/' + '_' + str(i_H) + '_' + str(i_W) + '_' + str(n_H) + '_' + str(
                n_W) + '_' + '.png', sub_image)
        else:
            cv2.imwrite(str(folder_to_save) + '/' + str(save_name) + '.png', sub_image)

    def decompose_image(self, n_H_sub=1024, n_W_sub=1024, stride=64):
        num_subsections_processed = 0
        num_subsections = (((self.n_H - n_H_sub) / stride) + 1) * (((self.n_W - n_W_sub) / stride) + 1)
        for i_H in range(0, self.n_H - n_H_sub + 1, stride):
            for i_W in range(0, self.n_W - n_W_sub + 1, stride):
                self.save_section(i_H, i_W, 'sub_sections', n_H_sub, n_W_sub)
                num_subsections_processed += 1
                print("Number subsections= " + str(num_subsections_processed) + "/" + str(int(num_subsections)))
        print("Number of subsections: ", num_subsections_processed)