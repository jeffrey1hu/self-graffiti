
import hickle
import tensorflow as tf
from config import *
from core.vggnet import Vgg19, load_and_resize_image
from core.utils import *

test_image_dir = TEST_DATA_PATH + '/caption_test1_images_20170923/'
batch_size = 100


def main():
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)
    tf.reset_default_graph()
    vggnet = Vgg19(VGG_MODEL_PATH)
    vggnet.build()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        save_path = './data/%s/%s.features.hkl' % ('test', 'test')
        image_path = map(lambda _path: test_image_dir + _path, os.listdir(test_image_dir))
        n_examples = len(image_path)

        all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)

        for start, end in zip(range(0, n_examples, batch_size),
                              range(batch_size, n_examples + batch_size, batch_size)):
            image_batch_file = image_path[start:end]
            image_batch = np.array(map(lambda x: load_and_resize_image(x), image_batch_file)).astype(
                np.float32)
            feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
            all_feats[start:end, :] = feats
            print ("Processed %d %s features.." % (end, 'test'))

        # use hickle to save huge feature vectors
        hickle.dump(all_feats, save_path)
        print ("Saved %s.." % (save_path))


if __name__ == '__main__':
    main()