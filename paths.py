
import os, datetime, pytz

import names
import scipy.ndimage as ndi
import numpy as np

from constants import *

def create_names_list():
    set1 = {
        fname[:-4]
        for fname in  os.listdir(os.path.join(prefix, 'img'))
        if fname.endswith('.png')
    }
    set2 = {
        fname[:-4]
        for fname in  os.listdir(os.path.join(prefix, 'mask'))
        if fname.endswith('.png')
    }

    return list(set1 & set2)

def mask_path_of_name(name):
    return os.path.join(prefix, 'mask', name + '.png')

def img_path_of_name(name):
    return os.path.join(prefix, 'img', name  + '.png')

def create_name(tags=()):
    t = pytz.utc.localize(datetime.datetime.now()).astimezone(pytz.timezone('Europe/Paris'))
    t = t.strftime(time_format)
    name = names.get_first_name().lower()
    return '{}_{}_{}'.format(t, '-'.join(tags), name)

def get_model_name_list():
    l = [fname[:-5]
         for fname in os.listdir(os.path.join(prefix, 'models'))
         if fname.endswith('.hdf5')
    ]
    return l

def create_model_name(epoch, loss, acc):
    t = pytz.utc.localize(datetime.datetime.now()).astimezone(pytz.timezone('Europe/Paris'))
    t = t.strftime(time_format)
    name = names.get_last_name().lower()
    return '{}_{:04d}_{:08.6f}_{:08.6f}_{}'.format(t, epoch, loss, acc, name)

def create_model_path(epoch, loss, acc):
    return os.path.join(prefix, 'models', create_model_name(epoch, loss, acc) + '.hdf5')

def get_latest_model_path_opt():
    def _epoch_of_model_name(name):
        t = name.split('_')[0]
        t = datetime.datetime.strptime(t, time_format)
        t = (t - datetime.datetime.utcfromtimestamp(0)).total_seconds()
        return t

    l = get_model_name_list()
    if len(l) == 0:
        return None
    name = max(l, key=_epoch_of_model_name)
    return os.path.join(prefix, 'models', name + '.hdf5')

def img_of_name(name):
    shapes = []

    path = os.path.join(prefix, 'img', name + '.png')
    arr = ndi.imread(path).astype('uint8')
    shapes.append(str(arr.shape))

    arr = ndi.zoom(arr, (img_h / 1080, img_w / 1920, 1))
    shapes.append(str(arr.shape))

    # arr = np.moveaxis(arr, 2, 0)
    # shapes.append(str(arr.shape))

    print("Read img  {}, shapes:{}".format(
        name, ' -> '.join(shapes)
    ))
    return arr

def mask_of_name(name):
    shapes = []

    path = os.path.join(prefix, 'mask', name + '.png')
    arr = ndi.imread(path)
    shapes.append(str(arr.shape))

    arr = ndi.zoom(arr, (img_h / 1080, img_w / 1920), order=1)
    shapes.append(str(arr.shape))

    arr = arr.astype('bool')
    # arr = arr[..., np.newaxis]
    # shapes.append(str(arr.shape))

    print("Read mask {}, shapes:{}".format(
        name, ' -> '.join(shapes)
    ))
    return arr


# Build test names ********************************************************** **
def group_tags_of_names(names):
    return {
        tuple(name.split('_')[1].split('-'))
        for name in names
    }

def random_sample_group_tag(group_tags, taken_tags):
    allowed_group_tags = [
        group_tag
        for group_tag in group_tags
        if not any(taken_tag in group_tag for taken_tag in taken_tags)
    ]
    print("sampling a group tag... from {} to {} group tags".format(
        len(group_tags), len(allowed_group_tags)
    ))
    assert len(allowed_group_tags) > 0
    np.random.shuffle(allowed_group_tags)
    return allowed_group_tags[0]

def random_sample_group_tags(group_tags, count):
    taken_tags = set()
    sampled_group_tags = []

    for i in range(count):
        sampled_group_tag = random_sample_group_tag(group_tags, taken_tags)
        # print("took {}".format(sampled_group_tag))
        taken_tags |= set(sampled_group_tag)
        sampled_group_tags.append(sampled_group_tag)
    return sampled_group_tags

def create_test_names():
    names = create_names_list()
    group_tags = list(group_tags_of_names(names))
    test_group_tags = random_sample_group_tags(group_tags, 3)
    return [
        name
        for name in names
        if any(
            all(tag in name for tag in group_tag)
            for group_tag in test_group_tags
        )
    ]
