# MIT License
#
# Copyright (c) 2017 Maarten Bloemen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import os
import cv2
import align.detect_face
import facenet
import numpy as np
import tensorflow as tf
from scipy import misc
from sklearn.cluster import DBSCAN
from functions import GoogleFunctions, ImdbFunctions


def margin_crop(det, margin, img_size):
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0] - margin / 2, 0)
    bb[1] = np.maximum(det[1] - margin / 2, 0)
    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])

    return bb


# This function aligns the data and extracts all faces from the picture before feeding to the network
# It returns a list for the model to use and a list with the user specified size and margin
def align_data(image_list, image_size, margin, pnet, rnet, onet):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img_list = []
    safe_list = []

    for x in xrange(len(image_list)):
        img_size = np.asarray(image_list[x].shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(image_list[x], minsize, pnet, rnet, onet, threshold, factor)
        nrof_samples = len(bounding_boxes)
        if nrof_samples > 0:
            for i in xrange(nrof_samples):
                if bounding_boxes[i][4] > 0.95:
                    det = np.squeeze(bounding_boxes[i, 0:4])
                    bb_model = margin_crop(det, 44, img_size)
                    bb_safe = margin_crop(det, margin, img_size)
                    # Crop the face out of the image
                    cropped_model = image_list[x][bb_model[1]:bb_model[3], bb_model[0]:bb_model[2], :]
                    cropped_safe = image_list[x][bb_safe[1]:bb_safe[3], bb_safe[0]:bb_safe[2], :]
                    # Check if the face isn't to blurry
                    if cv2.Laplacian(cropped_model, cv2.CV_64F).var() > 100:
                        aligned_model = misc.imresize(cropped_model, (160, 160), interp='bilinear')
                        aligned_safe = misc.imresize(cropped_safe, (image_size, image_size), interp='bilinear')
                        prewhitened_model = facenet.prewhiten(aligned_model)
                        prewhitened_safe = facenet.prewhiten(aligned_safe)
                        # Add the aligned face to the list to return
                        img_list.append(prewhitened_model)
                        safe_list.append(prewhitened_safe)

    # Check if the list is not empty
    if len(img_list) > 0:
        images_model = np.stack(img_list)
        images_safe = np.stack(safe_list)
        return images_model, images_safe
    else:
        return None, None


def create_network_face_detection(gpu_memory_fraction):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=(gpu_memory_fraction*0.3))
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet


def main(args, pnet, rnet, onet):
    _pnet = pnet
    _rnet = rnet
    _onet = onet

    imdb_name_scraper = ImdbFunctions()
    google_scraper = GoogleFunctions()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=(args.gpu_memory_fraction*0.7))

    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                              log_device_placement=False)) as sess:
            # Open the facenet model
            facenet.load_model(args.model)
            images_file_path = os.path.join(args.out_dir, 'sources.txt')

            # Create a new file containing the sources of the images
            with open(images_file_path, 'w+') as source_file:
                source_file.write('Url\tDate\n')

            # Read names from txt file else scrape IMDB for names
            if '.txt' in args.name_source:
                names = []
                with open(args.name_source) as file:
                    for name in file.readlines():
                        names.append(name.strip())
            else:
                names = imdb_name_scraper.get_celebrity_names(args.limit)

            for name in names:
                images = google_scraper.get_images(images_file_path, name, args.safe_mode)

                # Check if the image list is not empty
                if images is not None:
                    # Align the image data
                    images_aligned_model, images_aligned_safe = \
                        align_data(images, args.image_size, args.margin, _pnet, _rnet, _onet)

                    if images_aligned_model is not None:
                        # Get the required input and output tensors
                        images_placeholder = sess.graph.get_tensor_by_name("input:0")
                        embeddings = sess.graph.get_tensor_by_name("embeddings:0")
                        phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
                        feed_dict = {images_placeholder: images_aligned_model, phase_train_placeholder: False}
                        emb = sess.run(embeddings, feed_dict=feed_dict)

                        # Get number of faces in the list after alignment
                        nrof_images = len(images_aligned_model)

                        print(nrof_images)

                        # Create empty distance matrix
                        matrix = np.zeros((nrof_images, nrof_images))
                        for i in range(nrof_images):
                            for j in range(nrof_images):
                                # Calc distance and fill the matrix
                                dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
                                matrix[i][j] = dist

                        # Instantiate the cluster algorithm, eps = the min distance to cluster
                        db = DBSCAN(eps=args.cluster_threshold, min_samples=args.min_cluster_size, metric='precomputed')
                        # Fit the distance matrix to the algorithm
                        db.fit(matrix)
                        labels = db.labels_

                        # Find how many clusters there are
                        no_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                        # Check if there is more than 1 cluster
                        if no_clusters > 0:
                            print('No of clusters:', no_clusters)
                            biggest_cluster = 0
                            len_biggest_cluster = 0
                            for i in range(no_clusters):
                                print('Cluster ' + str(i) + ' : ', np.nonzero(labels == i)[0])
                                # Find the biggest cluster
                                if len(np.nonzero(labels == i)[0]) > len_biggest_cluster:
                                    biggest_cluster = i
                                    len_biggest_cluster = len(np.nonzero(labels == i)[0])

                            print('Biggest cluster: ' + str(biggest_cluster))
                            cnt = 1
                            # Putting the full path in a variable to make it easy
                            path = os.path.join(args.out_dir, str(name.strip()))
                            if not os.path.exists(path):
                                # Create a dir in the chosen output location with the name of the persons sub Reddit if it doesn't exist
                                os.makedirs(path)
                                # Loop over the images array positions in the largest dir
                                for j in np.nonzero(labels == biggest_cluster)[0]:
                                    # Save the image to the output dir
                                    misc.imsave(os.path.join(path, name.strip() + '_' + str('%0*d' % (4, cnt)) + '.png'),
                                                images_aligned_safe[j])
                                    cnt += 1
                            else:
                                for j in np.nonzero(labels == biggest_cluster)[0]:
                                    misc.imsave(os.path.join(path, name.strip() + '_ ' + str('%0*d' % (4, cnt)) + '.png'),
                                                images_aligned_safe[j])
                                    cnt += 1


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='Path to the model a protobuf (.pb) file.')
    parser.add_argument('out_dir', type=str,
                        help='The output directory where the image clusters will be saved.')
    parser.add_argument('--name_source', type=str,
                        help='Uses IMDB to find name, you can also give a path to your own txt file containing names.',
                        default='imdb')
    parser.add_argument('--limit', type=int,
                        help='Maximum number of IMDB pages to get names from '
                             '(number of people being scraped is limit * 50)', default=100)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--min_cluster_size', type=int,
                        help='The minimum amount of pictures required for a cluster.', default=10)
    parser.add_argument('--cluster_threshold', type=float,
                        help='The minimum distance for faces to be in the same cluster', default=1.0)
    parser.add_argument('--safe_mode', type=str,
                        help='When this is set off, explicit images will be included.',
                        default='on', choices=['on', 'off'])
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)


    args = parser.parse_args()

    # Creating the 3 layers for 3 step face detection network
    # pnet = proposal network
    # rnet = refinement network
    # onet = output network
    pnet, rnet, onet = create_network_face_detection(args.gpu_memory_fraction)
    # Main entry point into the application, passing the args and the layers for the face detection
    main(args, pnet, rnet, onet)
