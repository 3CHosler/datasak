#Brian Hosler
#January 2022


from .. import util

#TODO: intelligent dims unpacking
#TODO: variable dtype images
#TODO: variable sized labels


def pack(xmple):
    datum = {
            'label': util.int64_feature(xmple[1]),
            'patch': util.bytes_feature(xmple[0].tostring())
            }
    return tf.train.Example(features=tf.train.Features(feature=datum))


def unpack(xmple, dims=128):
    dims = (128,128,3)
    datum = tf.io.parse_single_example(xmple, features={
        'label':tf.io.FixedLenFeature([], tf.int64),
        'patch':tf.io.FixelLenFeature([], tf.string)
        })
    image = tf.decode_raw(datum['patch'], tf.uint8)
    image = tf.reshape(image, dims)
    label = tf.cast(datum['label'], tf.int32)
    return image, label
