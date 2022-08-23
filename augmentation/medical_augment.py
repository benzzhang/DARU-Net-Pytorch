'''
Author: Pengbo
LastEditTime: 2022-05-21 18:59:00
Description: augmentation for landmark detection of medical image

'''
import numpy as np
from skimage import transform as sktrans


def rotate(angle):
    '''
        angle: °
    '''
    def func(img):
        ''' img: ndarray, channel x imgsize
        '''
        ret = []
        for i in range(img.shape[0]):
            ret.append(sktrans.rotate(img[i], angle))
        return np.array(ret)
    return func


def translate(offsets):
    ''' translation
        offsets: n-item list-like, for each dim
    '''
    offsets = tuple(offsets)
    new_sls = tuple(slice(i, None) for i in offsets)

    def func(img):
        ''' img: ndarray, channel x imgsize
        '''
        ret = []
        size = img.shape[1:]
        old_sls = tuple(slice(0, j-i) for i, j in zip(offsets, size))

        for old in img:
            new = np.zeros(size)
            new[new_sls] = old[old_sls]
            ret.append(new)
        return np.array(ret)
    return func


def flip(axis=1):
    '''
    axis=0: flip all
       else flip axis
    '''
    f_sls = slice(None, None, -1)
    sls = slice(None, None)

    def func(img):
        dim = img.ndim
        cur_axis = axis % dim
        if cur_axis == 0:
            all_sls = tuple([f_sls])*dim
        else:
            all_sls = tuple(
                            f_sls if i == cur_axis else sls for i in range(dim))
            return img[all_sls]
    return func


def LmsDetectTrainTransform(rotate_angle=15, offset=[15, 15]):
    transform_list = []
    #if np.random.rand() < 0.5:
    #    transform_list.append(flip(1))
    if np.random.rand() < 0.5:
        transform_list.append(rotate((np.random.rand()-0.5)*rotate_angle))
    if np.random.rand() < 0.5:
        rorate_x = int((np.random.rand()-0.5)*offset[0])
        rorate_y = int((np.random.rand()-0.5)*offset[1])
        transform_list.append(translate([rorate_x, rorate_y]))

    def trans(*imgs):
        ''' img: chanel x imageshape
        '''
        ret = []
        for img in imgs:
            # copy is necessary, to avoid modifying origin data
            cur_img = img.copy()
            for f in transform_list:
                cur_img = f(cur_img)
            # copy is necessary, torch needs ascontiguousarray
            ret.append(cur_img.copy())
        return tuple(ret)
    return trans


def LmsDetectTestTransform():
    def trans(*imgs):
        ''' img: chanel x imageshape
        '''
        ret = []
        for img in imgs:
            # copy is necessary, to avoid modifying origin data
            cur_img = img.copy()
            ret.append(cur_img)
        return tuple(ret)
    return trans
