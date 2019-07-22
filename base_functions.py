import os
import numpy as np
import matplotlib.image as mpimg


def get_train_data(train_images_dir, train_labels_dir, img_h, img_w, N_channels=1, C=2, gt_list = None):
    print('-'*30)
    print('Loading train images...')
    print('-'*30)
    assert(C == 2 or C > 2)
    files = os.listdir(train_images_dir)  # get file names
    total_images = np.zeros([len(files), img_h, img_w, N_channels])  # for storing training imgs
    for idx in range(len(files)):  #
        img = mpimg.imread(os.path.join(train_images_dir, files[idx]))
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        total_images[idx, :, :, :img.shape[-1]] = img
    total_images = total_images/np.max(total_images)
    mean = np.mean(total_images, axis=0)
    np.save('./mean_img.npy', mean)
    total_images -= mean
    total_images = np.transpose(total_images, [0, 3, 1, 2])

    print('-'*30)
    print('Loading train labels...')
    print('-'*30)
    files2 = os.listdir(train_labels_dir)
    total_labels = np.zeros([len(files), img_h, img_w, C])
    if C == 2:
        for idx in range(len(files)):
            ground_truth = mpimg.imread(os.path.join(train_labels_dir, files2[idx]))
            total_labels[idx, :, :, 0] = ((ground_truth == 0)*1)
            total_labels[idx, :, :, 1] = ((ground_truth != 0)*1)
    else:
        if gt_list is None:
            print('-' * 30 + '\n' + 'There is a lack of a list of GT values!')
            raise Exception
        else:
            for idx in range(len(files)):
                ground_truth = mpimg.imread(os.path.join(train_labels_dir, files2[idx]))
                for ch in range(C):
                    total_labels[idx, :, :, ch] = ((ground_truth == gt_list[ch]) * 1)

    return total_images, total_labels


def get_test_data(test_images_dir, test_labels_dir, img_h, img_w, N_channels=1, C=2):
    print('-'*30)
    print('Loading test images...')

    files = os.listdir(test_images_dir)  # get file names
    total_images = np.zeros([len(files), img_h, img_w, N_channels])
    for idx in range(len(files)):
        img = mpimg.imread(os.path.join(test_images_dir,files[idx]))
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        total_images[idx, :, :, :img.shape[-1]] = img
    total_images = total_images/np.max(total_images)
    mean = np.load('./mean_img.npy')
    total_images -= mean
    total_images = np.transpose(total_images, [0, 3, 1, 2])  # transpose to shape[N, channels, h, w]

    print('-'*30)
    print('Loading test labels...')
    print('-'*30)
    files2 = os.listdir(test_labels_dir)
    total_labels = np.zeros([len(files2), img_h, img_w])
    for idx in range(len(files2)):
        total_labels[idx] = mpimg.imread(os.path.join(test_labels_dir, files2[idx]))

    return total_images, total_labels


def pred_to_imgs(predictions, img_h, img_w, C=2):
    assert(len(predictions.shape) == 3)
    assert(predictions.shape[1] == img_h*img_w)
    N_images = predictions.shape[0]
    predictions = np.reshape(predictions, [N_images, img_h, img_w, C])
    pred_images = np.argmax(predictions, axis=3)

    return pred_images


def pred_to_imgs_bak(predictions, img_h, img_w, C=2):
    assert(len(predictions.shape) == 3)
    assert(predictions.shape[1] == img_h*img_w)
    N_images = predictions.shape[0]
    predictions = np.reshape(predictions,[N_images, img_h, img_w, C])
    pred_images = np.zeros([N_images, img_h, img_w])
    for img in range(N_images):
        for h in range(img_h):
            for w in range(img_w):
                l = list(predictions[img, h, w, :])
                pred_images[img, h, w] = l.index(max(l))
    pred_images /= np.max(pred_images)
    assert(np.min(pred_images) == 0 and np.max(pred_images) == 1)

    return pred_images











