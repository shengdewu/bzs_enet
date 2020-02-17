from scipy.misc import imread

class weight(object):

    def median_frequency_balancing(self, image_files, num_class=12):
        '''
        note: we weight each pixel by αc = median freq/freq(c)
              where freq(c) is (the number of pixels of class c) divided by (the total number of pixels in images where c is present),
              and (median freq is the median of these frequencies)

            "the number of pixels of class c": Represents the total number of pixels of class c across all images of the dataset.
            "The total number of pixels in images where c is present": Represents the total number of pixels across all images (where there is at least one pixel of class c) of the dataset.
            "median frequency is the median of these frequencies": Sort the frequencies calculated above and pick the median.

        :param image_files:
        :param num_class:
        :return:
        '''


        for image_file in image_files:
            image = imread(image_file)

        return