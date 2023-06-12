"""
 *  @file  face_detect.py
 *  @brief Parses already saved videos and detect faces. Delete them after parsing is done.
 *
 *  @author Kalp Garg.
"""
import argparse
import json
import os
import shutil
import random
import time
from datetime import datetime, timezone
import pytz
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model

import numpy as np
from matplotlib import pyplot as plt

from py_logging import get_logger
import albumentations as alb

global logger


def return_datetime(mode=1):
    date_format = '%Y-%m-%d__%H_%M_%S'
    dnt_utc = datetime.now(tz=pytz.utc)
    dnt_pdt = dnt_utc.astimezone()
    if mode == 0:
        return dnt_pdt
    elif mode == 1:
        return dnt_pdt.strftime(date_format)


class FaceTracker(Model):
    def __init__(self, eyetracker, **kwargs):
        super().__init__(**kwargs)
        self.model = eyetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt

    def train_step(self, batch, **kwargs):
        X, y = batch

        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)

            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)

            total_loss = batch_localizationloss + 0.5 * batch_classloss

            grad = tape.gradient(total_loss, self.model.trainable_variables)

        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))

        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def test_step(self, batch, **kwargs):
        X, y = batch

        classes, coords = self.model(X, training=False)

        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss + 0.5 * batch_classloss

        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def call(self, X, **kwargs):
        return self.model(X, **kwargs)


class FaceTrain(object):
    def __init__(self, data_folder, data_splitting_needed=False, albumentation_needed=False) -> None:
        if not os.path.exists(data_folder):
            logger.error("Given path {} doesn't exist. Please check.. ".format(self.in_dir))
            return 0
        self.input_db_path = data_folder
        self.raw_untouched_data_dir = self.input_db_path + "raw_untouched_data"

        if data_splitting_needed:
            self.create_train_test_val_split()
        if albumentation_needed:
            self.do_albumentation()

        self.merge_images_and_labels()
        self.train_model()


    def create_train_test_val_split(self):
        total_files = len(os.listdir(self.raw_untouched_data_dir))
        # print(os.listdir(data_dir_path), total_files)
        total_training_sample = int(0.7 * total_files)
        total_test_sample = int(0.16 * total_files)
        total_val_sample = int(0.14 * total_files)

        logger.info(
            "Total \n training sample: {} \n test_sample: {} \n validation_sample: {}".format(total_training_sample,
                                                                                              total_test_sample,
                                                                                              total_val_sample))
        self.segregrate_data(data_type="train", total_data_sample=total_training_sample)
        self.segregrate_data(data_type="test", total_data_sample=total_test_sample)
        self.segregrate_data(data_type="val", total_data_sample=total_val_sample)

    def segregrate_data(self, data_type=None, total_data_sample=0):
        if (data_type is None) | (total_data_sample == 0):
            logger.error(
                "Total data sample can't be zero. data type is required. Can be either 'test', 'train' or 'validate'.")
            return 0

        # create dir if not exists
        data_type = str(data_type)
        if not os.path.exists(os.path.join(self.input_db_path, data_type, "data")):
            os.makedirs(os.path.join(self.input_db_path, data_type, "data"))
        if not os.path.exists(os.path.join(self.input_db_path, data_type, "labels")):
            os.makedirs(os.path.join(self.input_db_path, data_type, "labels"))
        for i in range(total_data_sample):
            data_files = os.listdir(self.raw_untouched_data_dir)
            total_files = len(data_files)
            rand_int = random.randint(0, total_files - 1)
            logger.info("Total data files: {}. random file: {}".format(total_files, rand_int))

            existing_fpath = os.path.join(self.raw_untouched_data_dir, data_files[rand_int])
            new_fpath = os.path.join(self.input_db_path, data_type, "data", data_files[rand_int])
            os.replace(existing_fpath, new_fpath)

            # put correspondning label into label dir
            f_name = data_files[rand_int].split('.')[0] + '.json'
            existing_fpath = os.path.join(self.input_db_path, "raw_untouched_data_labels", f_name)
            if os.path.exists(existing_fpath):
                new_fpath = os.path.join(self.input_db_path, data_type, "labels", f_name)
                os.replace(existing_fpath, new_fpath)
                # shutil.copyfile(existing_fpath, new_fpath)

    def do_albumentation(self):
        augmentor = alb.Compose([alb.RandomCrop(width=900, height=900),
                                 alb.HorizontalFlip(p=0.5),
                                 alb.RandomBrightnessContrast(p=0.2),
                                 alb.RandomGamma(p=0.2),
                                 alb.RGBShift(p=0.2),
                                 alb.VerticalFlip(p=0.5)],
                                bbox_params=alb.BboxParams(format='albumentations',
                                                           label_fields=['class_labels']))

        for partition in ['train', 'test', 'val']:
            # create aug_data dir if not exists
            if not os.path.exists(os.path.join(self.input_db_path, 'aug_data', partition, 'images')):
                os.makedirs(os.path.join(self.input_db_path, 'aug_data', partition, 'images'))

            if not os.path.exists(os.path.join(self.input_db_path, 'aug_data', partition, 'labels')):
                os.makedirs(os.path.join(self.input_db_path, 'aug_data', partition, 'labels'))

            for image in os.listdir(os.path.join(self.input_db_path, partition, 'data')):
                img = cv2.imread(os.path.join(self.input_db_path, partition, 'data', image))

                coords = [0, 0, 0.00001, 0.00001]
                label_path = os.path.join(self.input_db_path, partition, 'labels', f'{image.split(".")[0]}.json')
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        label = json.load(f)

                    coords[0] = label['shapes'][0]['points'][0][0]
                    coords[1] = label['shapes'][0]['points'][0][1]
                    coords[2] = label['shapes'][0]['points'][1][0]
                    coords[3] = label['shapes'][0]['points'][1][1]
                    coords = list(np.divide(coords, [960, 1080, 960, 1080]))

                try:
                    for x in range(60):
                        augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                        cv2.imwrite(os.path.join(self.input_db_path, 'aug_data', partition, 'images',
                                                 f'{image.split(".")[0]}.{x}.jpg'),
                                    augmented['image'])

                        annotation = {}
                        annotation['image'] = image

                        if os.path.exists(label_path):
                            if len(augmented['bboxes']) == 0:
                                annotation['bbox'] = [0, 0, 0, 0]
                                annotation['class'] = 0
                            else:
                                annotation['bbox'] = augmented['bboxes'][0]
                                annotation['class'] = 1
                        else:
                            annotation['bbox'] = [0, 0, 0, 0]
                            annotation['class'] = 0

                        with open(os.path.join(self.input_db_path, 'aug_data', partition, 'labels',
                                               f'{image.split(".")[0]}.{x}.json'),
                                  'w') as f:
                            json.dump(annotation, f)

                except Exception as e:
                    print(e)

    def merge_images_and_labels(self):  
        train_images = tf.data.Dataset.list_files(
            os.path.join(self.input_db_path, 'aug_data', 'train', 'images', "*.jpg"), shuffle=False)
        train_images = train_images.map(self.load_image)
        train_images = train_images.map(lambda x: tf.image.resize(x, (500, 500)))
        train_images = train_images.map(lambda x: x / 255)

        test_images = tf.data.Dataset.list_files(
            os.path.join(self.input_db_path, 'aug_data', 'test', 'images', '*.jpg'), shuffle=False)
        test_images = test_images.map(self.load_image)
        test_images = test_images.map(lambda x: tf.image.resize(x, (500, 500)))
        test_images = test_images.map(lambda x: x / 255)

        val_images = tf.data.Dataset.list_files(os.path.join(self.input_db_path, 'aug_data', 'val', 'images', '*.jpg'),
                                                shuffle=False)
        val_images = val_images.map(self.load_image)
        val_images = val_images.map(lambda x: tf.image.resize(x, (500, 500)))
        val_images = val_images.map(lambda x: x / 255)

        train_labels = tf.data.Dataset.list_files(
            os.path.join(self.input_db_path, 'aug_data', 'train', 'labels', '*.json'), shuffle=False)
        train_labels = train_labels.map(lambda x: tf.py_function(self.load_labels, [x], [tf.uint8, tf.float16]))

        test_labels = tf.data.Dataset.list_files(
            os.path.join(self.input_db_path, 'aug_data', 'test', 'labels', '*.json'), shuffle=False)
        test_labels = test_labels.map(lambda x: tf.py_function(self.load_labels, [x], [tf.uint8, tf.float16]))

        val_labels = tf.data.Dataset.list_files(os.path.join(self.input_db_path, 'aug_data', 'val', 'labels', '*.json'),
                                                shuffle=False)
        val_labels = val_labels.map(lambda x: tf.py_function(self.load_labels, [x], [tf.uint8, tf.float16]))

        logger.info(
            "len(train_images): {}, len(train_labels): {}, len(test_images): {}, len(test_labels): {}, len(val_images): {}, len(val_labels): {}".format(
                len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images),
                len(val_labels)
            ))
        
        train = tf.data.Dataset.zip((train_images, train_labels))
        train = train.shuffle(5000)
        train = train.batch(8)
        train = train.prefetch(4)

        self.test = tf.data.Dataset.zip((test_images, test_labels))
        self.test = self.test.shuffle(1300)
        self.test = self.test.batch(8)
        self.test = self.test.prefetch(4)

        val = tf.data.Dataset.zip((val_images, val_labels))
        val = val.shuffle(1000)
        val = val.batch(8)
        val = val.prefetch(4)

                # for i in range(10):
        #     res = train.as_numpy_iterator().next()
        #     # res = data_samples.next()
            
        #     fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
        #     for idx in range(4):
        #         sample_image = res[0][idx]
        #         sample_coords = res[1][1][idx]
            
        #         cv2.rectangle(sample_image,
        #                     tuple((sample_coords[:2]).astype(int)),
        #                     tuple((sample_coords[2:]).astype(int)),
        #                     (255, 0, 0), 2)
            
        #         ax[idx].imshow(sample_image)
        #     fig.savefig("{}iiiiiii.jpg".format(i))
            # plt.show()

    def train_model(self):
        self.limit_gpu_growth()

        facetracker = self.build_model()

        # facetracker.summary()
        X, y = train.as_numpy_iterator().next()
        print(X.shape)
        classes, coords = facetracker.predict(X)
        print(classes, coords)

        batches_per_epoch = len(train)
        lr_decay = (1. / 0.75 - 1) / batches_per_epoch
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, decay=lr_decay)

        classloss = tf.keras.losses.BinaryCrossentropy()
        regressloss = self.localization_loss

        print(self.localization_loss(y[1], coords))
        print(classloss(y[0], classes))
        print(regressloss(y[1], coords))

        model = FaceTracker(facetracker)
        model.compile(opt, classloss, regressloss)

        logdir = 'logs'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])

        print(hist.history)
        facetracker.save('facetracker_new.h5')

        fig, ax = plt.subplots(ncols=3, figsize=(20, 5))

        ax[0].plot(hist.history['total_loss'], color='teal', label='loss')
        ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
        ax[0].title.set_text('Loss')
        ax[0].legend()

        ax[1].plot(hist.history['class_loss'], color='teal', label='class loss')
        ax[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')
        ax[1].title.set_text('Classification Loss')
        ax[1].legend()

        ax[2].plot(hist.history['regress_loss'], color='teal', label='regress loss')
        ax[2].plot(hist.history['val_regress_loss'], color='orange', label='val regress loss')
        ax[2].title.set_text('Regression Loss')
        ax[2].legend()

        plt.show()

    def limit_gpu_growth(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("GPU: {}".format(tf.config.list_physical_devices('GPU')))

    def test_model(self):
        facetracker = load_model('facetracker_new.h5')

        for i in range(20):
            test_data = self.test.as_numpy_iterator()
            test_sample = test_data.next()
            # print(test_sample)
            yhat = facetracker.predict(test_sample[0])

            fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
            for idx in range(4):
                sample_image = test_sample[0][idx]
                sample_coords = yhat[1][idx]

                if yhat[0][idx] > 0.9:
                    cv2.rectangle(sample_image,
                                  tuple(np.multiply(sample_coords[:2], [500, 500]).astype(int)),
                                  tuple(np.multiply(sample_coords[2:], [500, 500]).astype(int)),
                                  (255, 0, 0), 2)

                ax[idx].imshow(sample_image)
            fig.savefig("{}_{}.jpg".format(i, idx))

    def localization_loss(self, y_true, yhat):
        delta_coord = tf.reduce_sum(tf.square(y_true[:, :2] - yhat[:, :2]))

        h_true = y_true[:, 3] - y_true[:, 1]
        w_true = y_true[:, 2] - y_true[:, 0]

        h_pred = yhat[:, 3] - yhat[:, 1]
        w_pred = yhat[:, 2] - yhat[:, 0]

        delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))

        return delta_coord + delta_size

    def build_model(self):
        input_layer = Input(shape=(500, 500, 3))

        vgg = VGG16(include_top=False)(input_layer)

        # Classification Model
        f1 = GlobalMaxPooling2D()(vgg)
        class1 = Dense(2048, activation='relu')(f1)
        class2 = Dense(1, activation='sigmoid')(class1)

        # Bounding box model
        f2 = GlobalMaxPooling2D()(vgg)
        regress1 = Dense(2048, activation='relu')(f2)
        regress2 = Dense(4, activation='sigmoid')(regress1)

        facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
        return facetracker

    def load_labels(self, label_path):
        with open(label_path.numpy(), 'r', encoding="utf-8") as f:
            label = json.load(f)
            # label['bbox'] = np.multiply(label['bbox'], [500, 500, 500, 500])
        
        return [label['class'], label['bbox']]

    def load_image(self, x):
        byte_img = tf.io.read_file(x)
        img = tf.io.decode_jpeg(byte_img)
        return img


class FaceDetect(object):
    def __init__(self, in_folder):
        self.in_dir = in_folder
        # self.input_db_path = "/Users/kgarg/Documents/extras/home_cam_security/input/input_db/"
        self.input_db_path = in_folder
        self.parse_in_dir()

    def parse_in_dir(self):
        if not os.path.exists(self.in_dir):
            logger.error("Given path {} doesn't exist. Please check.. ".format(self.in_dir))
            return 0
        total_files = len(os.listdir(self.in_dir))
        return os.listdir(self.in_dir)

    def video_processing_pipeline(self):
        current_dir = self.parse_in_dir()
        i_time = time.time()
        for id, video in enumerate(current_dir):
            video_path = os.path.join(self.in_dir, video)
            logger.info("Processing video : {}".format(video))
            i_time = time.time()
            self.face_detect_and_identify(video_path)
            logger.info("Time to process : {}".format(time.time() - i_time))

    def face_detect_and_identify(self, v_path):
        cap = cv2.VideoCapture(v_path)
        pTime = 0
        count = 1
        while cap.isOpened():
            success, img = cap.read()
            if success:
                new_size = (int(960 / 2), int(1080 / 2))
                img = cv2.resize(img, new_size)
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                count = count + 1
                cv2.imshow("CP_PLUS", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    func_list = ["train_model", "test_model"]
    face_detect_args = argparse.ArgumentParser(description="Parse videos and detect face")
    face_detect_args.version = "23.03.01"  # yy.mm.vv
    face_detect_args.add_argument('-v', '--version', action='version', help="displays the version. Format = yy.mm.v")
    face_detect_args.add_argument('-l', '--log_folder', type=str, metavar='face_detect_log',
                                  default="face_detect_log",
                                  help="Location of the log folder")

    subparser = face_detect_args.add_subparsers(
        help="Function to choose from. Either provide train model or test model",
        dest='subparser_name')
    
    parser_train_model = subparser.add_parser(func_list[0], help="Trains the model. Requires folder location of saved captures of face.")
    parser_train_model.add_argument('-id', '--input_data_folder', type=str, metavar='input_data_folder', required=True,
                                  help="Location of the input folder which has saved face captures")

    parser_test_model = subparser.add_parser(func_list[1], help="Test the model. Requires folder location which contains streams.")
    parser_train_model.add_argument('-il', '--input_log_folder', type=str, metavar='input_log_folder', required=True,
                                  help="Location of the input folder which has saved streams")

    args = face_detect_args.parse_args()

    addl_file_loc = os.path.join("face_detect", args.log_folder,
                                 "{}_{}.txt".format("face_detect_", return_datetime(mode=1)))
    logger = get_logger(__name__, addl_file_loc, save_to_file=True)
    logger.info("Script version is: {}".format(face_detect_args.version))

    if args.subparser_name == func_list[0]:
        #train model
        pass
    elif args.subparser_name == func_list[1]:
        # test model
        pass

    face_detect = FaceDetect(in_folder=args.input_log_folder)
    # face_detect.video_processing_pipeline()
    face_detect.load_images()
    # face_detect.create_train_test_val_split()
    face_detect.test_model()
