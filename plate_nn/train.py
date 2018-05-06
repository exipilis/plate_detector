import argparse
import random
from multiprocessing import cpu_count

import numpy as np
from imageio import imread, imsave
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence
from scipy.misc import imresize, imrotate

from model import is_plate_model


class PSequence(Sequence):
    def __init__(self,
                 image_shape: tuple,
                 image_filenames: list,
                 file_templates: list,
                 batch_size: int = 32,
                 shuffle: bool = True):

        self.batch_size = batch_size
        self.image_filenames = image_filenames
        self.file_templates = file_templates
        self.classes_count = len(file_templates)

        self.image_shape = image_shape

        self.shuffle = shuffle
        self.batches_fetched = 0
        self.debug = False

        self.len_images = len(self.image_filenames)

    def __len__(self):
        return np.math.ceil(self.len_images / self.batch_size)

    def __getitem__(self, index):
        s = index * self.batch_size
        e = s + self.batch_size
        self.batches_fetched += 1

        if not (self.batches_fetched % len(self)) and self.shuffle:
            random.shuffle(self.image_filenames)

        filenames_batch = [self.image_filenames[i % self.len_images] for i in range(s, e)]

        x_batch = np.zeros((self.batch_size,) + self.image_shape)
        y_batch = np.zeros((self.batch_size, self.classes_count if self.classes_count > 2 else 1))

        for i, image_fn in enumerate(filenames_batch):
            img = imread(image_fn, pilmode='RGB')
            # horizontal flip
            if random.uniform(0, 1) > 0.5:
                np.flip(img, 1)

            # random crop
            h, w = img.shape[0:2]
            x1 = random.randint(0, w // 10)
            x2 = w - random.randint(0, w // 10)
            y1 = random.randint(0, h // 10)
            y2 = h - random.randint(0, h // 10)
            img = img[y1:y2, x1:x2]

            # rotate
            angle = random.uniform(-10, 10)
            img = imrotate(img, angle)

            # final resize
            img = imresize(img, self.image_shape)
            if self.debug:
                imsave('/tmp/%s.jpg' % i, img)

            x_batch[i] = img

            class_id = -1
            for j, s in enumerate(self.file_templates):
                if s in image_fn:
                    class_id = j

            if class_id < 0:
                print('error: ' + image_fn + ' pattern does not match any class template')

            if self.classes_count == 2:  # one class - 0/1 encoding
                y_batch[i] = class_id
            else:  # more than one class - 1-hot encoding
                y_batch[i, class_id] = 1

        x_batch = x_batch / 127.5 - 1

        return x_batch, y_batch


def main(train_file, val_file):
    file_templates = ['/negative/', '/positive/']

    weights_filename = 'checkpoints/plate.h5'
    model = is_plate_model()

    model.summary()

    try:
        model.load_weights(weights_filename)
        print('weights loaded')
    except:
        print('weights random')

    train_dataset = [s.strip() for s in open(train_file)]
    val_dataset = [s.strip() for s in open(val_file)]

    print('Train examples %s, validation examples %s' % (len(train_dataset), len(val_dataset)))

    model_shape = model.input_shape[1:]

    train_generator = PSequence(model_shape, train_dataset, file_templates)
    val_generator = PSequence(model_shape, val_dataset, file_templates)

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=1000,
        epochs=1000,
        callbacks=[ModelCheckpoint(weights_filename, save_best_only=True)],
        validation_data=val_generator,
        validation_steps=100,
        max_queue_size=train_generator.batch_size * 2,
        workers=cpu_count(),
        use_multiprocessing=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('-train_dataset', default='dataset/train.txt', help='Train dataset list  directory')
    parser.add_argument('-val_dataset', default='dataset/val.txt', help='Validation percentage')
    args = parser.parse_args()

    main(args.train_dataset, args.val_dataset)
