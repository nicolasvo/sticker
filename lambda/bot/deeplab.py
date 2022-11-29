from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras.models import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import Add
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import DepthwiseConv2D
from tensorflow.python.keras.layers import ZeroPadding2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.utils.layer_utils import get_source_inputs
from tensorflow.keras import backend as K

import numpy as np
from PIL import Image
import cv2
import time
from datetime import datetime
import imantics
from imantics import Polygons, Mask


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


def SepConv_BN(
    x,
    filters,
    prefix,
    stride=1,
    kernel_size=3,
    rate=1,
    depth_activation=False,
    epsilon=1e-3,
):
    if stride == 1:
        depth_padding = "same"
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = "valid"

    if not depth_activation:
        x = Activation(tf.nn.relu)(x)
    x = DepthwiseConv2D(
        (kernel_size, kernel_size),
        strides=(stride, stride),
        dilation_rate=(rate, rate),
        padding=depth_padding,
        use_bias=False,
        name=prefix + "_depthwise",
    )(x)
    x = BatchNormalization(name=prefix + "_depthwise_BN", epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(tf.nn.relu)(x)
    x = Conv2D(
        filters, (1, 1), padding="same", use_bias=False, name=prefix + "_pointwise"
    )(x)
    x = BatchNormalization(name=prefix + "_pointwise_BN", epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(tf.nn.relu)(x)

    return x


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):

    if stride == 1:
        return Conv2D(
            filters,
            (kernel_size, kernel_size),
            strides=(stride, stride),
            padding="same",
            use_bias=False,
            dilation_rate=(rate, rate),
            name=prefix,
        )(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(
            filters,
            (kernel_size, kernel_size),
            strides=(stride, stride),
            padding="valid",
            use_bias=False,
            dilation_rate=(rate, rate),
            name=prefix,
        )(x)


def _xception_block(
    inputs,
    depth_list,
    prefix,
    skip_connection_type,
    stride,
    rate=1,
    depth_activation=False,
    return_skip=False,
):
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(
            residual,
            depth_list[i],
            prefix + "_separable_conv{}".format(i + 1),
            stride=stride if i == 2 else 1,
            rate=rate,
            depth_activation=depth_activation,
        )
        if i == 1:
            skip = residual
    if skip_connection_type == "conv":
        shortcut = _conv2d_same(
            inputs, depth_list[-1], prefix + "_shortcut", kernel_size=1, stride=stride
        )
        shortcut = BatchNormalization(name=prefix + "_shortcut_BN")(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == "sum":
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == "none":
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def Deeplab_xcep_pascal(
    weights=None,
    input_tensor=None,
    input_shape=(512, 512, 3),
    num_classes=21,
    backbone="xception",
    OS=8,
    model_path="",
    activation=None,
):

    if not (weights in {"pascalvoc", None}):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `pascalvoc` "
            "(pre-trained on PascalVoc)"
        )

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    if OS == 8:

        entry_block3_stride = 1
        middle_block_rate = 2  # ! Not mentioned in paper, but required
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36)

        x = Conv2D(
            32,
            (3, 3),
            strides=(2, 2),
            name="entry_flow_conv1_1",
            use_bias=False,
            padding="same",
        )(img_input)
        x = BatchNormalization(name="entry_flow_conv1_1_BN")(x)
        x = Activation(tf.nn.relu)(x)

        x = _conv2d_same(x, 64, "entry_flow_conv1_2", kernel_size=3, stride=1)
        x = BatchNormalization(name="entry_flow_conv1_2_BN")(x)
        x = Activation(tf.nn.relu)(x)

        x = _xception_block(
            x,
            [128, 128, 128],
            "entry_flow_block1",
            skip_connection_type="conv",
            stride=2,
            depth_activation=False,
        )
        x, skip1 = _xception_block(
            x,
            [256, 256, 256],
            "entry_flow_block2",
            skip_connection_type="conv",
            stride=2,
            depth_activation=False,
            return_skip=True,
        )

        x = _xception_block(
            x,
            [728, 728, 728],
            "entry_flow_block3",
            skip_connection_type="conv",
            stride=entry_block3_stride,
            depth_activation=False,
        )
        for i in range(16):
            x = _xception_block(
                x,
                [728, 728, 728],
                "middle_flow_unit_{}".format(i + 1),
                skip_connection_type="sum",
                stride=1,
                rate=middle_block_rate,
                depth_activation=False,
            )

        x = _xception_block(
            x,
            [728, 1024, 1024],
            "exit_flow_block1",
            skip_connection_type="conv",
            stride=1,
            rate=exit_block_rates[0],
            depth_activation=False,
        )
        x = _xception_block(
            x,
            [1536, 1536, 2048],
            "exit_flow_block2",
            skip_connection_type="none",
            stride=1,
            rate=exit_block_rates[1],
            depth_activation=True,
        )

    else:

        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        atrous_rates = (6, 12, 18)

        x = Conv2D(
            32,
            (3, 3),
            strides=(2, 2),
            name="entry_flow_conv1_1",
            use_bias=False,
            padding="same",
        )(img_input)
        x = BatchNormalization(name="entry_flow_conv1_1_BN")(x)
        x = Activation(tf.nn.relu)(x)

        x = _conv2d_same(x, 64, "entry_flow_conv1_2", kernel_size=3, stride=1)
        x = BatchNormalization(name="entry_flow_conv1_2_BN")(x)
        x = Activation(tf.nn.relu)(x)

        x = _xception_block(
            x,
            [128, 128, 128],
            "entry_flow_block1",
            skip_connection_type="conv",
            stride=2,
            depth_activation=False,
        )
        x, skip1 = _xception_block(
            x,
            [256, 256, 256],
            "entry_flow_block2",
            skip_connection_type="conv",
            stride=2,
            depth_activation=False,
            return_skip=True,
        )

        x = _xception_block(
            x,
            [728, 728, 728],
            "entry_flow_block3",
            skip_connection_type="conv",
            stride=entry_block3_stride,
            depth_activation=False,
        )
        for i in range(16):
            x = _xception_block(
                x,
                [728, 728, 728],
                "middle_flow_unit_{}".format(i + 1),
                skip_connection_type="sum",
                stride=1,
                rate=middle_block_rate,
                depth_activation=False,
            )

        x = _xception_block(
            x,
            [728, 1024, 1024],
            "exit_flow_block1",
            skip_connection_type="conv",
            stride=1,
            rate=exit_block_rates[0],
            depth_activation=False,
        )
        x = _xception_block(
            x,
            [1536, 1536, 2048],
            "exit_flow_block2",
            skip_connection_type="none",
            stride=1,
            rate=exit_block_rates[1],
            depth_activation=True,
        )

    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding="same", use_bias=False, name="image_pooling")(b4)
    b4 = BatchNormalization(name="image_pooling_BN", epsilon=1e-5)(b4)
    b4 = Activation(tf.nn.relu)(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = Lambda(
        lambda x: tf.compat.v1.image.resize(
            x, size_before[1:3], method="bilinear", align_corners=True
        )
    )(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding="same", use_bias=False, name="aspp0")(x)
    b0 = BatchNormalization(name="aspp0_BN", epsilon=1e-5)(b0)
    b0 = Activation(tf.nn.relu, name="aspp0_activation")(b0)

    if backbone == "xception":
        # rate = 6 (12)
        b1 = SepConv_BN(
            x, 256, "aspp1", rate=atrous_rates[0], depth_activation=True, epsilon=1e-5
        )
        # rate = 12 (24)
        b2 = SepConv_BN(
            x, 256, "aspp2", rate=atrous_rates[1], depth_activation=True, epsilon=1e-5
        )
        # rate = 18 (36)
        b3 = SepConv_BN(
            x, 256, "aspp3", rate=atrous_rates[2], depth_activation=True, epsilon=1e-5
        )

        # concatenate ASPP branches & project
        x = Concatenate()([b4, b0, b1, b2, b3])

    x = Conv2D(256, (1, 1), padding="same", use_bias=False, name="concat_projection")(x)
    x = BatchNormalization(name="concat_projection_BN", epsilon=1e-5)(x)
    x = Activation(tf.nn.relu)(x)
    x = Dropout(0.1)(x)
    # DeepLab v.3+ decoder

    if backbone == "xception":
        # Feature projection
        # x4 (x2) block
        size_before2 = tf.keras.backend.int_shape(x)
        x = Lambda(
            lambda xx: tf.compat.v1.image.resize(
                xx, skip1.shape[1:3], method="bilinear", align_corners=True
            )
        )(x)

        dec_skip1 = Conv2D(
            48, (1, 1), padding="same", use_bias=False, name="feature_projection0"
        )(skip1)
        dec_skip1 = BatchNormalization(name="feature_projection0_BN", epsilon=1e-5)(
            dec_skip1
        )
        dec_skip1 = Activation(tf.nn.relu)(dec_skip1)
        x = Concatenate()([x, dec_skip1])
        x = SepConv_BN(x, 256, "decoder_conv0", depth_activation=True, epsilon=1e-5)
        x = SepConv_BN(x, 256, "decoder_conv1", depth_activation=True, epsilon=1e-5)

    if weights == "pascalvoc" and num_classes == 21:
        last_layer_name = "logits_semantic"
    else:
        last_layer_name = "custom_logits_semantic"

    x = Conv2D(num_classes, (1, 1), padding="same", name=last_layer_name)(x)
    size_before3 = tf.keras.backend.int_shape(img_input)
    x = Lambda(
        lambda xx: tf.compat.v1.image.resize(
            xx, size_before3[1:3], method="bilinear", align_corners=True
        )
    )(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    if activation in {"softmax", "sigmoid"}:
        x = tf.keras.layers.Activation(activation)(x)

    model = Model(inputs, x, name="deeplabv3plus")

    if weights == "pascalvoc":
        weights_path = model_path
        model.load_weights(weights_path, by_name=True)
    return model


def Deeplab_xcep_ade20k(
    weights=None,
    input_tensor=None,
    input_shape=(512, 512, 3),
    num_classes=151,
    backbone="xception",
    OS=8,
    model_path="",
    activation=None,
):

    if not (weights in {"ade20k", None}):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `ade20k` "
            "(pre-trained on ADE20K)"
        )

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    if OS == 8:

        entry_block3_stride = 1
        middle_block_rate = 2  # ! Not mentioned in paper, but required
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36)

        x = Conv2D(
            32,
            (3, 3),
            strides=(2, 2),
            name="entry_flow_conv1_1",
            use_bias=False,
            padding="same",
        )(img_input)
        x = BatchNormalization(name="entry_flow_conv1_1_BN")(x)
        x = Activation(tf.nn.relu)(x)

        x = _conv2d_same(x, 64, "entry_flow_conv1_2", kernel_size=3, stride=1)
        x = BatchNormalization(name="entry_flow_conv1_2_BN")(x)
        x = Activation(tf.nn.relu)(x)

        x = _xception_block(
            x,
            [128, 128, 128],
            "entry_flow_block1",
            skip_connection_type="conv",
            stride=2,
            depth_activation=False,
        )
        x, skip1 = _xception_block(
            x,
            [256, 256, 256],
            "entry_flow_block2",
            skip_connection_type="conv",
            stride=2,
            depth_activation=False,
            return_skip=True,
        )

        x = _xception_block(
            x,
            [728, 728, 728],
            "entry_flow_block3",
            skip_connection_type="conv",
            stride=entry_block3_stride,
            depth_activation=False,
        )
        for i in range(16):
            x = _xception_block(
                x,
                [728, 728, 728],
                "middle_flow_unit_{}".format(i + 1),
                skip_connection_type="sum",
                stride=1,
                rate=middle_block_rate,
                depth_activation=False,
            )

        x = _xception_block(
            x,
            [728, 1024, 1024],
            "exit_flow_block1",
            skip_connection_type="conv",
            stride=1,
            rate=exit_block_rates[0],
            depth_activation=False,
        )
        x = _xception_block(
            x,
            [1536, 1536, 2048],
            "exit_flow_block2",
            skip_connection_type="none",
            stride=1,
            rate=exit_block_rates[1],
            depth_activation=True,
        )

    else:

        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        atrous_rates = (6, 12, 18)

        x = Conv2D(
            32,
            (3, 3),
            strides=(2, 2),
            name="entry_flow_conv1_1",
            use_bias=False,
            padding="same",
        )(img_input)
        x = BatchNormalization(name="entry_flow_conv1_1_BN")(x)
        x = Activation(tf.nn.relu)(x)

        x = _conv2d_same(x, 64, "entry_flow_conv1_2", kernel_size=3, stride=1)
        x = BatchNormalization(name="entry_flow_conv1_2_BN")(x)
        x = Activation(tf.nn.relu)(x)

        x = _xception_block(
            x,
            [128, 128, 128],
            "entry_flow_block1",
            skip_connection_type="conv",
            stride=2,
            depth_activation=False,
        )
        x, skip1 = _xception_block(
            x,
            [256, 256, 256],
            "entry_flow_block2",
            skip_connection_type="conv",
            stride=2,
            depth_activation=False,
            return_skip=True,
        )

        x = _xception_block(
            x,
            [728, 728, 728],
            "entry_flow_block3",
            skip_connection_type="conv",
            stride=entry_block3_stride,
            depth_activation=False,
        )
        for i in range(16):
            x = _xception_block(
                x,
                [728, 728, 728],
                "middle_flow_unit_{}".format(i + 1),
                skip_connection_type="sum",
                stride=1,
                rate=middle_block_rate,
                depth_activation=False,
            )

        x = _xception_block(
            x,
            [728, 1024, 1024],
            "exit_flow_block1",
            skip_connection_type="conv",
            stride=1,
            rate=exit_block_rates[0],
            depth_activation=False,
        )
        x = _xception_block(
            x,
            [1536, 1536, 2048],
            "exit_flow_block2",
            skip_connection_type="none",
            stride=1,
            rate=exit_block_rates[1],
            depth_activation=True,
        )

    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding="same", use_bias=False, name="image_pooling")(b4)
    b4 = BatchNormalization(name="image_pooling_BN", epsilon=1e-5)(b4)
    b4 = Activation(tf.nn.relu)(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = Lambda(
        lambda x: tf.compat.v1.image.resize(
            x, size_before[1:3], method="bilinear", align_corners=True
        )
    )(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding="same", use_bias=False, name="aspp0")(x)
    b0 = BatchNormalization(name="aspp0_BN", epsilon=1e-5)(b0)
    b0 = Activation(tf.nn.relu, name="aspp0_activation")(b0)

    if backbone == "xception":
        # rate = 6 (12)
        b1 = SepConv_BN(
            x, 256, "aspp1", rate=atrous_rates[0], depth_activation=True, epsilon=1e-5
        )
        # rate = 12 (24)
        b2 = SepConv_BN(
            x, 256, "aspp2", rate=atrous_rates[1], depth_activation=True, epsilon=1e-5
        )
        # rate = 18 (36)
        b3 = SepConv_BN(
            x, 256, "aspp3", rate=atrous_rates[2], depth_activation=True, epsilon=1e-5
        )

        # concatenate ASPP branches & project
        x = Concatenate()([b4, b0, b1, b2, b3])

    x = Conv2D(256, (1, 1), padding="same", use_bias=False, name="concat_projection")(x)
    x = BatchNormalization(name="concat_projection_BN", epsilon=1e-5)(x)
    x = Activation(tf.nn.relu)(x)
    x = Dropout(0.1)(x)
    # DeepLab v.3+ decoder

    if backbone == "xception":
        # Feature projection
        # x4 (x2) block
        size_before2 = tf.keras.backend.int_shape(x)
        x = Lambda(
            lambda xx: tf.compat.v1.image.resize(
                xx, skip1.shape[1:3], method="bilinear", align_corners=True
            )
        )(x)

        dec_skip1 = Conv2D(
            48, (1, 1), padding="same", use_bias=False, name="feature_projection0"
        )(skip1)
        dec_skip1 = BatchNormalization(name="feature_projection0_BN", epsilon=1e-5)(
            dec_skip1
        )
        dec_skip1 = Activation(tf.nn.relu)(dec_skip1)
        x = Concatenate()([x, dec_skip1])
        x = SepConv_BN(x, 256, "decoder_conv0", depth_activation=True, epsilon=1e-5)
        x = SepConv_BN(x, 256, "decoder_conv1", depth_activation=True, epsilon=1e-5)

    # if (weights == 'ade20k' and num_classes == 150):
    last_layer_name = "logits_semantic"
    # else:
    # last_layer_name = 'custom_logits_semantic'

    x = Conv2D(num_classes, kernel_size=(1, 1), padding="same", name=last_layer_name)(x)
    size_before3 = tf.keras.backend.int_shape(img_input)
    x = Lambda(
        lambda xx: tf.compat.v1.image.resize(
            xx, size_before3[1:3], method="bilinear", align_corners=True
        )
    )(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    if activation in {"softmax", "sigmoid"}:
        x = tf.keras.layers.Activation(activation)(x)

    model = Model(inputs, x, name="deeplabv3plus")

    if weights == "ade20k":
        weights_path = model_path
        model.load_weights(weights_path, by_name=True)
    return model


class semantic_segmentation:
    INPUT_TENSOR_NAME = "ImageTensor:0"
    OUTPUT_TENSOR_NAME = "SemanticPredictions:0"
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = "frozen_inference_graph"

    def __init__(self, model_type="h5"):
        global model_file
        self.model_type = model_type
        model_file = model_type

        self.model = Deeplab_xcep_pascal()

        self.model2 = Deeplab_xcep_ade20k()

    def load_pascalvoc_model(self, model_path):
        if model_file == "pb":
            self.graph = tf.Graph()

            graph_def = None

            with tf.compat.v1.gfile.GFile(model_path, "rb") as file_handle:
                graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())

            if graph_def is None:
                raise RuntimeError("Cannot find inference graph")

            with self.graph.as_default():
                tf.graph_util.import_graph_def(graph_def, name="")

            self.sess = tf.compat.v1.Session(graph=self.graph)

        else:
            self.model.load_weights(model_path)

    def load_ade20k_model(self, model_path):
        self.model2.load_weights(model_path)

    ####### SEMANTIC SEGMENTATION WITH PASCALVOC MODEL ######

    def segmentAsPascalvoc(
        self,
        image_path,
        process_frame=False,
        output_image_name=None,
        overlay=False,
        verbose=None,
    ):

        if model_file == "pb":

            if process_frame == True:
                image = image_path

            else:
                image = cv2.imread(image_path)

            w, h, n = image.shape

            image_overlay = image.copy()

            if n > 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            resize_ratio = 1.0 * self.INPUT_SIZE / max(w, h)
            target_size = (int(resize_ratio * w), int(resize_ratio * h))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            batch_seg_map = self.sess.run(
                self.OUTPUT_TENSOR_NAME,
                feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]},
            )

            seg_image = batch_seg_map[0]
            raw_labels = seg_image

            """Access  the unique class ids of the masks """
            unique_labels = np.unique(raw_labels)

            raw_labels = np.array(
                Image.fromarray(raw_labels.astype("uint8")).resize((h, w))
            )

            """ Convert the indexed masks to boolean masks """
            raw_labels = np.ma.make_mask(raw_labels)
            segvalues = {"class_ids": unique_labels, "masks": raw_labels}

            # Apply segmentation color map
            labels = labelP_to_color_image(seg_image)
            labels = np.array(Image.fromarray(labels.astype("uint8")).resize((h, w)))
            new_img = cv2.cvtColor(labels, cv2.COLOR_RGB2BGR)

            if overlay == True:
                alpha = 0.7
                cv2.addWeighted(
                    new_img, alpha, image_overlay, 1 - alpha, 0, image_overlay
                )

                if output_image_name is not None:
                    cv2.imwrite(output_image_name, image_overlay)
                    print(
                        "Processed Image saved successfully in your current working directory."
                    )

                return segvalues, image_overlay

            else:
                if output_image_name is not None:

                    cv2.imwrite(output_image_name, new_img)

                    print(
                        "Processed Image saved successfuly in your current working directory."
                    )

                return segvalues, new_img

        else:

            trained_image_width = 512
            mean_subtraction_value = 127.5

            if process_frame == True:
                image = image_path
            else:
                image = np.array(Image.open(image_path))

            # resize to max dimension of images from training dataset
            w, h, n = image.shape

            if n > 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

            image_overlay = image.copy()

            ratio = float(trained_image_width) / np.max([w, h])
            resized_image = np.array(
                Image.fromarray(image.astype("uint8")).resize(
                    (int(ratio * h), int(ratio * w))
                )
            )
            resized_image = (resized_image / mean_subtraction_value) - 1

            # pad array to square image to match training images
            pad_x = int(trained_image_width - resized_image.shape[0])
            pad_y = int(trained_image_width - resized_image.shape[1])
            resized_image = np.pad(
                resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode="constant"
            )

            if verbose is not None:
                print("Processing image....")

            # run prediction
            res = self.model.predict(np.expand_dims(resized_image, 0))

            labels = np.argmax(res.squeeze(), -1)
            # remove padding and resize back to original image
            if pad_x > 0:
                labels = labels[:-pad_x]
            if pad_y > 0:
                labels = labels[:, :-pad_y]

            raw_labels = labels

            """ Access the unique class ids of the masks"""
            unique_labels = np.unique(raw_labels)

            raw_labels = np.array(
                Image.fromarray(raw_labels.astype("uint8")).resize((h, w))
            )

            """ Convert the indexed masks to boolean  """
            raw_labels = np.ma.make_mask(raw_labels)

            segvalues = {"class_ids": unique_labels, "masks": raw_labels}

            # Apply segmentation color map
            labels = labelP_to_color_image(labels)
            labels = np.array(Image.fromarray(labels.astype("uint8")).resize((h, w)))

            new_img = cv2.cvtColor(labels, cv2.COLOR_RGB2BGR)

            if overlay == True:
                alpha = 0.7
                cv2.addWeighted(
                    new_img, alpha, image_overlay, 1 - alpha, 0, image_overlay
                )

                if output_image_name is not None:
                    cv2.imwrite(output_image_name, image_overlay)
                    print(
                        "Processed Image saved successfully in your current working directory."
                    )

                return segvalues, image_overlay

            else:
                if output_image_name is not None:

                    cv2.imwrite(output_image_name, new_img)

                    print(
                        "Processed Image saved successfuly in your current working directory."
                    )

                return segvalues, new_img

    def segmentFrameAsPascalvoc(
        self, frame, output_frame_name=None, overlay=False, verbose=None
    ):
        if overlay == True:
            raw_labels, frame_overlay = self.segmentAsPascalvoc(
                frame, overlay=True, process_frame=True
            )

            if output_frame_name is not None:
                cv2.imwrite(output_frame_name, frame_overlay)

            return raw_labels, frame_overlay

        else:
            raw_labels, new_frame = self.segmentAsPascalvoc(frame, process_frame=True)

            if output_frame_name is not None:
                cv2.imwrite(output_frame_name, new_frame)

            return raw_labels, new_frame

    def process_video_pascalvoc(
        self, video_path, overlay=False, frames_per_second=None, output_video_name=None
    ):
        capture = cv2.VideoCapture(video_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if output_video_name is not None:
            save_video = cv2.VideoWriter(
                output_video_name,
                cv2.VideoWriter_fourcc(*"DIVX"),
                frames_per_second,
                (width, height),
            )

        counter = 0
        start = time.time()

        if overlay == True:
            while True:
                counter += 1

                ret, frame = capture.read()

                if ret:
                    raw_labels, frame_overlay = self.segmentAsPascalvoc(
                        frame, overlay=True, process_frame=True
                    )
                    print("No. of frames:", counter)
                    output = cv2.resize(
                        frame_overlay, (width, height), interpolation=cv2.INTER_AREA
                    )
                    if output_video_name is not None:
                        save_video.write(output)
                else:
                    break

            end = time.time()
            print(f"Processed {counter} frames in {end-start:.1f} seconds")
            capture.release()
            if output_video_name is not None:
                save_video.release()
            return raw_labels, output

        else:
            while True:

                counter += 1
                ret, frame = capture.read()

                if ret:
                    raw_labels, new_frame = self.segmentAsPascalvoc(
                        frame, process_frame=True
                    )
                    print("No. of frames:", counter)
                    output = cv2.resize(
                        new_frame, (width, height), interpolation=cv2.INTER_AREA
                    )
                    if output_video_name is not None:
                        save_video.write(output)

                else:
                    break

            capture.release()

            end = time.time()
            print(f"Processed {counter} frames in {end-start:.1f} seconds")

            if frames_per_second is not None:
                save_video.release()

            return raw_labels, output

    def process_camera_pascalvoc(
        self,
        cam,
        overlay=False,
        check_fps=False,
        frames_per_second=None,
        output_video_name=None,
        show_frames=False,
        frame_name=None,
        verbose=None,
    ):
        capture = cam
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_video_name is not None:
            save_video = cv2.VideoWriter(
                output_video_name,
                cv2.VideoWriter_fourcc(*"DIVX"),
                frames_per_second,
                (width, height),
            )

        counter = 0
        start = datetime.now()

        if overlay == True:
            while True:

                ret, frame = capture.read()
                if ret:
                    raw_labels, frame_overlay = self.segmentAsPascalvoc(
                        frame, overlay=True, process_frame=True
                    )
                    counter += 1
                    if show_frames == True:
                        if frame_name is not None:
                            cv2.imshow(frame_name, frame_overlay)
                            if cv2.waitKey(25) & 0xFF == ord("q"):
                                break

                    if output_video_name is not None:
                        output = cv2.resize(
                            frame_overlay, (width, height), interpolation=cv2.INTER_AREA
                        )
                        save_video.write(output)

                elif counter == 30:
                    break

            end = datetime.now()

            if check_fps == True:
                timetaken = (end - start).total_seconds()
                fps = counter / timetaken
                print(f"{fps} frames per seconds")

            capture.release()

            if verbose is not None:
                print(f"Processed {counter} frames in {timetaken:.1f} seconds")

            if output_video_name is not None:
                save_video.release()

            return raw_labels, frame_overlay

        else:
            while True:
                ret, frame = capture.read()
                if ret:
                    raw_labels, new_frame = self.segmentAsPascalvoc(
                        frame, process_frame=True
                    )
                    counter += 1

                    if show_frames == True:
                        if frame_name is not None:
                            cv2.imshow(frame_name, new_frame)
                            if cv2.waitKey(25) & 0xFF == ord("q"):
                                break

                    if output_video_name is not None:
                        output = cv2.resize(
                            new_frame, (width, height), interpolation=cv2.INTER_AREA
                        )
                        save_video.write(output)

                elif counter == 30:
                    break
            end = datetime.now()
            if check_fps == True:
                timetaken = (end - start).total_seconds()
                fps = counter / timetaken
                print(f"{fps} frames per seconds")

            capture.release()

            if verbose is not None:
                print(f"Processed {counter} frames in {timetaken:.1f} seconds")

            if output_video_name is not None:
                save_video.release()

            return raw_labels, new_frame

    ################# SEMANTIC SEGMENTATION WITH ADE20K MODEL ##########################

    def segmentAsAde20k(
        self,
        image_path,
        output_image_name=None,
        overlay=False,
        extract_segmented_objects=False,
        process_frame=False,
        verbose=None,
    ):
        trained_image_width = 512
        mean_subtraction_value = 127.5
        if process_frame == True:
            image = image_path
        else:
            image = np.array(Image.open(image_path))

        # resize to max dimension of images from training dataset
        w, h, n = image.shape

        if n > 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        image_overlay = image.copy()

        ratio = float(trained_image_width) / np.max([w, h])
        resized_image = np.array(
            Image.fromarray(image.astype("uint8")).resize(
                (int(ratio * h), int(ratio * w))
            )
        )
        resized_image = (resized_image / mean_subtraction_value) - 1

        # pad array to square image to match training images
        pad_x = int(trained_image_width - resized_image.shape[0])
        pad_y = int(trained_image_width - resized_image.shape[1])
        resized_image = np.pad(
            resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode="constant"
        )

        if verbose is not None:
            print("Processing image....")
        # run prediction
        res = self.model2.predict(np.expand_dims(resized_image, 0))

        labels = np.argmax(res.squeeze(), -1)

        # remove padding and resize back to original image
        if pad_x > 0:
            labels = labels[:-pad_x]
        if pad_y > 0:
            labels = labels[:, :-pad_y]

        raw_labels = labels

        """ Convert indexed masks to boolean """
        # raw_labels = np.ma.make_mask(raw_labels)
        # segvalues = {"class_ids":unique_labels,  "masks":raw_labels}

        # Apply segmentation color map
        labels = labelAde20k_to_color_image(labels)
        labels = np.array(Image.fromarray(labels.astype("uint8")).resize((h, w)))
        new_img = cv2.cvtColor(labels, cv2.COLOR_RGB2BGR)

        if extract_segmented_objects == True:
            segvalues, objects_masks = ade20k_map_color_mask(
                raw_labels, extract_segmented_objects=extract_segmented_objects
            )

        else:
            segvalues = ade20k_map_color_mask(
                raw_labels, extract_segmented_objects=extract_segmented_objects
            )

        if overlay == True:
            alpha = 0.7
            cv2.addWeighted(new_img, alpha, image_overlay, 1 - alpha, 0, image_overlay)

            if output_image_name is not None:
                cv2.imwrite(output_image_name, image_overlay)
                print(
                    "Processed Image saved successfully in your current working directory."
                )

            """ Get list of each segmeneted object with their %ratio, masks, class name, class index, and color"""
            if extract_segmented_objects == True:
                resize_masks = np.array(
                    Image.fromarray(segvalues["masks"].astype("uint8")).resize((h, w))
                )
                segvalues["masks"] = resize_masks
                segvalues["masks"] = np.ma.make_mask(segvalues["masks"])

                return segvalues, objects_masks, image_overlay

            else:
                resize_masks = np.array(
                    Image.fromarray(segvalues["masks"].astype("uint8")).resize((h, w))
                )
                segvalues["masks"] = resize_masks
                segvalues["masks"] = np.ma.make_mask(segvalues["masks"])

                return segvalues, image_overlay

        else:
            if output_image_name is not None:

                cv2.imwrite(output_image_name, new_img)

                print(
                    "Processed Image saved successfuly in your current working directory."
                )

            if extract_segmented_objects == True:
                resize_masks = np.array(
                    Image.fromarray(segvalues["masks"].astype("uint8")).resize((h, w))
                )
                segvalues["masks"] = resize_masks
                segvalues["masks"] = np.ma.make_mask(segvalues["masks"])

                return segvalues, objects_masks, new_img

            else:
                resize_masks = np.array(
                    Image.fromarray(segvalues["masks"].astype("uint8")).resize((h, w))
                )
                segvalues["masks"] = resize_masks
                segvalues["masks"] = np.ma.make_mask(segvalues["masks"])

                return segvalues, new_img

    def segmentFrameAsAde20k(
        self,
        frame,
        output_frame_name=None,
        overlay=False,
        verbose=None,
        extract_segmented_objects=False,
    ):
        if overlay == True:
            if extract_segmented_objects == False:
                segvalues, frame_overlay = self.segmentAsAde20k(
                    frame,
                    overlay=True,
                    process_frame=True,
                    extract_segmented_objects=False,
                )
                return segvalues, frame_overlay

            else:
                segvalues, extracted_objects, frame_overlay = self.segmentAsAde20k(
                    frame,
                    overlay=True,
                    process_frame=True,
                    extract_segmented_objects=True,
                )

                return segvalues, extracted_objects, frame_overlay

            if output_frame_name is not None:
                cv2.imwrite(output_frame_name, frame_overlay)

        else:
            if extract_segmented_objects == False:
                segvalues, new_frame = self.segmentAsAde20k(
                    frame, process_frame=True, extract_segmented_objects=False
                )

                return segvalues, new_frame

            else:
                segvalues, extracted_objects, new_frame = self.segmentAsAde20k(
                    frame, process_frame=True, extract_segmented_objects=True
                )

                return segvalues, extracted_objects, new_frame

            if output_frame_name is not None:
                cv2.imwrite(output_frame_name, new_frame)

    def process_video_ade20k(
        self,
        video_path,
        overlay=False,
        frames_per_second=None,
        output_video_name=None,
        extract_segmented_objects=False,
    ):
        capture = cv2.VideoCapture(video_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if output_video_name is not None:
            save_video = cv2.VideoWriter(
                output_video_name,
                cv2.VideoWriter_fourcc(*"DIVX"),
                frames_per_second,
                (width, height),
            )

        counter = 0
        start = time.time()

        if overlay == True:
            while True:
                counter += 1

                ret, frame = capture.read()

                if ret:
                    if extract_segmented_objects == False:
                        segvalues, frame_overlay = self.segmentAsAde20k(
                            frame,
                            overlay=True,
                            process_frame=True,
                            extract_segmented_objects=False,
                        )

                    else:
                        (
                            segvalues,
                            extracted_objects,
                            frame_overlay,
                        ) = self.segmentAsAde20k(
                            frame,
                            overlay=True,
                            process_frame=True,
                            extract_segmented_objects=True,
                        )

                    print("No. of frames:", counter)
                    output = cv2.resize(
                        frame_overlay, (width, height), interpolation=cv2.INTER_AREA
                    )
                    if output_video_name is not None:
                        save_video.write(output)

                else:
                    break

            end = time.time()
            print(f"Processed {counter} frames in {end-start:.1f} seconds")
            capture.release()
            if output_video_name is not None:
                save_video.release()

            if extract_segmented_objects == False:
                return segvalues, output

            else:
                return segvalues, extracted_objects, output

        else:
            while True:

                counter += 1
                ret, frame = capture.read()

                if ret:
                    if extract_segmented_objects == False:
                        segvalues, new_frame = self.segmentAsAde20k(
                            frame, process_frame=True, extract_segmented_objects=False
                        )

                    else:
                        segvalues, extracted_objects, new_frame = self.segmentAsAde20k(
                            frame, process_frame=True, extract_segmented_objects=True
                        )

                    print("No. of frames:", counter)
                    output = cv2.resize(
                        new_frame, (width, height), interpolation=cv2.INTER_AREA
                    )
                    if output_video_name is not None:
                        save_video.write(output)

                else:
                    break

            capture.release()

            end = time.time()
            print(f"Processed {counter} frames in {end-start:.1f} seconds")

            if frames_per_second is not None:
                save_video.release()
            if extract_segmented_objects == False:
                return segvalues, output

            else:
                return segvalues, extracted_objects, new_frame

    def process_camera_ade20k(
        self,
        cam,
        overlay=False,
        check_fps=False,
        frames_per_second=None,
        output_video_name=None,
        show_frames=False,
        frame_name=None,
        verbose=None,
        extract_segmented_objects=False,
    ):
        capture = cam
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_video_name is not None:
            save_video = cv2.VideoWriter(
                output_video_name,
                cv2.VideoWriter_fourcc(*"DIVX"),
                frames_per_second,
                (width, height),
            )

        counter = 0
        start = datetime.now()

        if overlay == True:
            while True:

                ret, frame = capture.read()
                if ret:
                    if extract_segmented_objects == False:
                        segvalues, frame_overlay = self.segmentAsAde20k(
                            frame,
                            overlay=True,
                            process_frame=True,
                            extract_segmented_objects=False,
                        )

                    else:
                        (
                            segvalues,
                            extracted_objects,
                            frame_overlay,
                        ) = self.segmentAsAde20k(
                            frame,
                            overlay=True,
                            process_frame=True,
                            extract_segmented_objects=True,
                        )

                    counter += 1
                    if show_frames == True:
                        if frame_name is not None:
                            cv2.imshow(frame_name, frame_overlay)
                            if cv2.waitKey(25) & 0xFF == ord("q"):
                                break

                    if output_video_name is not None:
                        output = cv2.resize(
                            frame_overlay, (width, height), interpolation=cv2.INTER_AREA
                        )
                        save_video.write(output)

                elif counter == 30:
                    break

            end = datetime.now()

            if check_fps == True:
                timetaken = (end - start).total_seconds()
                fps = counter / timetaken
                print(f"{fps} frames per seconds")

            capture.release()

            if verbose is not None:
                print(f"Processed {counter} frames in {timetaken:.1f} seconds")

            if output_video_name is not None:
                save_video.release()

            if extract_segmented_objects == False:
                return segvalues, frame_overlay

            else:
                return segvalues, extracted_objects, frame_overlay

        else:
            while True:
                ret, frame = capture.read()
                if ret:
                    if extract_segmented_objects == False:
                        segvalues, new_frame = self.segmentAsAde20k(
                            frame, process_frame=True, extract_segmented_objects=False
                        )

                    else:
                        segvalues, extracted_objects, new_frame = self.segmentAsAde20k(
                            frame, process_frame=True, extract_segmented_objects=True
                        )

                    counter += 1

                    if show_frames == True:
                        if frame_name is not None:
                            cv2.imshow(frame_name, new_frame)
                            if cv2.waitKey(25) & 0xFF == ord("q"):
                                break

                    if output_video_name is not None:
                        output = cv2.resize(
                            new_frame, (width, height), interpolation=cv2.INTER_AREA
                        )
                        save_video.write(output)

                elif counter == 30:
                    break
            end = datetime.now()
            if check_fps == True:
                timetaken = (end - start).total_seconds()
                fps = counter / timetaken
                print(f"{fps} frames per seconds")

            capture.release()

            if verbose is not None:
                print(f"Processed {counter} frames in {timetaken:.1f} seconds")

            if output_video_name is not None:
                save_video.release()
            if extract_segmented_objects == False:
                return segvalues, new_frame

            else:
                return segvalues, extracted_objects, new_frame


##Create Pascalvoc colormap format ##


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
      A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


# Obtain the results of segmentation#
def label_pascal():
    return np.asarray(
        [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [255, 255, 255],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [255, 255, 255],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ]
    )


def obtain_segmentation(image, nc=21):
    colors = label_pascal()
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for a in range(0, nc):
        index = image == a
        r[index] = colors[a, 0]
        g[index] = colors[a, 1]
        b[index] = colors[a, 2]
        rgb = np.stack([r, g, b], axis=2)

    return rgb


# Assign colors to objects #


def labelP_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")

    return colormap[label]


## Extracting class index, masks, names, % ratios, and classes
def ade20k_map_color_mask(raw_mask, extract_segmented_objects):
    names = create_ade20k_label_namemap()
    colors = create_ade20k_label_colormap()

    uniques, counts = np.unique(raw_mask, return_counts=True)

    class_index = []
    masks = []
    ratios = []
    class_name = []
    class_color = []

    d_dict = []

    for idx in np.argsort(counts)[::-1]:
        index_label = uniques[idx]
        label_mask = raw_mask == index_label

        class_index.append(index_label)
        masks.append(label_mask)
        ratios.append(counts[idx] / raw_mask.size * 100)
        class_name.append(names[index_label])
        class_color.append(colors[index_label])

        if extract_segmented_objects == True:

            d_dict.append(
                {
                    "class_id": index_label,
                    "class_name": names[index_label],
                    "class_color": colors[index_label],
                    "masks": label_mask,
                    "ratio": counts[idx] / raw_mask.size * 100,
                }
            )

    d_segment = {
        "class_ids": class_index,
        "class_names": class_name,
        "class_colors": class_color,
        "masks": label_mask,
        "ratios": ratios,
    }

    if extract_segmented_objects == False:
        return d_segment

    else:
        return d_segment, d_dict


##Create Ade20k namemap format ##
def create_ade20k_label_namemap():
    """Creates a label namemap used in ADE20K segmentation benchmark.
    Returns:
      A dict of classes names.
    """
    return {
        0: "no class",
        1: "wall",
        2: "building",
        3: "sky",
        4: "floor",
        5: "tree",
        6: "ceiling",
        7: "road",
        8: "bed",
        9: "windowpane",
        10: "grass",
        11: "cabinet",
        12: "sidewalk",
        13: "person",
        14: "earth",
        15: "door",
        16: "table",
        17: "mountain",
        18: "plant",
        19: "curtain",
        20: "chair",
        21: "car",
        22: "water",
        23: "painting",
        24: "sofa",
        25: "shelf",
        26: "house",
        27: "sea",
        28: "mirror",
        29: "rug",
        30: "field",
        31: "armchair",
        32: "seat",
        33: "fence",
        34: "desk",
        35: "rock",
        36: "wardrobe",
        37: "lamp",
        38: "bathtub",
        39: "railing",
        40: "cushion",
        41: "base",
        42: "box",
        43: "column",
        44: "signboard",
        45: "chest",
        46: "counter",
        47: "sand",
        48: "sink",
        49: "skyscraper",
        50: "fireplace",
        51: "refrigerator",
        52: "grandstand",
        53: "path",
        54: "stairs",
        55: "runway",
        56: "case",
        57: "pool",
        58: "pillow",
        59: "screen_door",
        60: "stairway",
        61: "river",
        62: "bridge",
        63: "bookcase",
        64: "blind",
        65: "coffee",
        66: "toilet",
        67: "flower",
        68: "book",
        69: "hill",
        70: "bench",
        71: "countertop",
        72: "stove",
        73: "palm",
        74: "kitchen",
        75: "computer",
        76: "swivel",
        77: "boat",
        78: "bar",
        79: "arcade",
        80: "hovel",
        81: "bus",
        82: "towel",
        83: "light",
        84: "truck",
        85: "tower",
        86: "chandelier",
        87: "awning",
        88: "streetlight",
        89: "booth",
        90: "television",
        91: "airplane",
        92: "dirt",
        93: "apparel",
        94: "pole",
        95: "land",
        96: "bannister",
        97: "escalator",
        98: "ottoman",
        99: "bottle",
        100: "buffet",
        101: "poster",
        102: "stage",
        103: "van",
        104: "ship",
        105: "fountain",
        106: "conveyer",
        107: "canopy",
        108: "washer",
        109: "plaything",
        110: "swimming",
        111: "stool",
        112: "barrel",
        113: "basket",
        114: "waterfall",
        115: "tent",
        116: "bag",
        117: "minibike",
        118: "cradle",
        119: "oven",
        120: "ball",
        121: "food",
        122: "step",
        123: "tank",
        124: "trade",
        125: "microwave",
        126: "pot",
        127: "animal",
        128: "bicycle",
        129: "lake",
        130: "dishwasher",
        131: "screen_projection",
        132: "blanket",
        133: "sculpture",
        134: "hood",
        135: "sconce",
        136: "vase",
        137: "traffic",
        138: "tray",
        139: "ashcan",
        140: "fan",
        141: "pier",
        142: "crt",
        143: "plate",
        144: "monitor",
        145: "bulletin",
        146: "shower",
        147: "radiator",
        148: "glass",
        149: "clock",
        150: "flag",
    }


##Create Ade20k colormap format ##
def create_ade20k_label_colormap():
    """Creates a label colormap used in ADE20K segmentation benchmark.
    Returns:
      A colormap for visualizing segmentation results.
    """
    return np.asarray(
        [
            [0, 0, 0],
            [120, 120, 120],
            [180, 120, 120],
            [6, 230, 230],
            [80, 50, 50],
            [4, 200, 3],
            [120, 120, 80],
            [140, 140, 140],
            [204, 5, 255],
            [230, 230, 230],
            [4, 250, 7],
            [224, 5, 255],
            [235, 255, 7],
            [150, 5, 61],
            [120, 120, 70],
            [8, 255, 51],
            [255, 6, 82],
            [143, 255, 140],
            [204, 255, 4],
            [255, 51, 7],
            [204, 70, 3],
            [0, 102, 200],
            [61, 230, 250],
            [255, 6, 51],
            [11, 102, 255],
            [255, 7, 71],
            [255, 9, 224],
            [9, 7, 230],
            [220, 220, 220],
            [255, 9, 92],
            [112, 9, 255],
            [8, 255, 214],
            [7, 255, 224],
            [255, 184, 6],
            [10, 255, 71],
            [255, 41, 10],
            [7, 255, 255],
            [224, 255, 8],
            [102, 8, 255],
            [255, 61, 6],
            [255, 194, 7],
            [255, 122, 8],
            [0, 255, 20],
            [255, 8, 41],
            [255, 5, 153],
            [6, 51, 255],
            [235, 12, 255],
            [160, 150, 20],
            [0, 163, 255],
            [140, 140, 140],
            [250, 10, 15],
            [20, 255, 0],
            [31, 255, 0],
            [255, 31, 0],
            [255, 224, 0],
            [153, 255, 0],
            [0, 0, 255],
            [255, 71, 0],
            [0, 235, 255],
            [0, 173, 255],
            [31, 0, 255],
            [11, 200, 200],
            [255, 82, 0],
            [0, 255, 245],
            [0, 61, 255],
            [0, 255, 112],
            [0, 255, 133],
            [255, 0, 0],
            [255, 163, 0],
            [255, 102, 0],
            [194, 255, 0],
            [0, 143, 255],
            [51, 255, 0],
            [0, 82, 255],
            [0, 255, 41],
            [0, 255, 173],
            [10, 0, 255],
            [173, 255, 0],
            [0, 255, 153],
            [255, 92, 0],
            [255, 0, 255],
            [255, 0, 245],
            [255, 0, 102],
            [255, 173, 0],
            [255, 0, 20],
            [255, 184, 184],
            [0, 31, 255],
            [0, 255, 61],
            [0, 71, 255],
            [255, 0, 204],
            [0, 255, 194],
            [0, 255, 82],
            [0, 10, 255],
            [0, 112, 255],
            [51, 0, 255],
            [0, 194, 255],
            [0, 122, 255],
            [0, 255, 163],
            [255, 153, 0],
            [0, 255, 10],
            [255, 112, 0],
            [143, 255, 0],
            [82, 0, 255],
            [163, 255, 0],
            [255, 235, 0],
            [8, 184, 170],
            [133, 0, 255],
            [0, 255, 92],
            [184, 0, 255],
            [255, 0, 31],
            [0, 184, 255],
            [0, 214, 255],
            [255, 0, 112],
            [92, 255, 0],
            [0, 224, 255],
            [112, 224, 255],
            [70, 184, 160],
            [163, 0, 255],
            [153, 0, 255],
            [71, 255, 0],
            [255, 0, 163],
            [255, 204, 0],
            [255, 0, 143],
            [0, 255, 235],
            [133, 255, 0],
            [255, 0, 235],
            [245, 0, 255],
            [255, 0, 122],
            [255, 245, 0],
            [10, 190, 212],
            [214, 255, 0],
            [0, 204, 255],
            [20, 0, 255],
            [255, 255, 0],
            [0, 153, 255],
            [0, 41, 255],
            [0, 255, 204],
            [41, 0, 255],
            [41, 255, 0],
            [173, 0, 255],
            [0, 245, 255],
            [71, 0, 255],
            [122, 0, 255],
            [0, 255, 184],
            [0, 92, 255],
            [184, 255, 0],
            [0, 133, 255],
            [255, 214, 0],
            [25, 194, 194],
            [102, 255, 0],
            [92, 0, 255],
        ]
    )


# Assign colors to objects#
def labelAde20k_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")

    colormap = create_ade20k_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")

    return colormap[label]


def labelAde20k_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")

    colormap = create_ade20k_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")

    return colormap[label]


def labelAde20k_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")

    colormap = create_ade20k_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")

    return colormap[label]


def labelAde20k_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")

    colormap = create_ade20k_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")

    return colormap[label]


class alter_bg:
    INPUT_TENSOR_NAME = "ImageTensor:0"
    OUTPUT_TENSOR_NAME = "SemanticPredictions:0"
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = "frozen_inference_graph"

    def __init__(self, model_type="h5"):
        global model_file
        self.model_type = model_type
        model_file = model_type

        self.model = Deeplab_xcep_pascal()

    def load_pascalvoc_model(self, model_path):
        if model_file == "pb":
            self.graph = tf.Graph()

            graph_def = None

            with tf.compat.v1.gfile.GFile(model_path, "rb") as file_handle:
                graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())

            if graph_def is None:
                raise RuntimeError("Cannot find inference graph")

            with self.graph.as_default():
                tf.graph_util.import_graph_def(graph_def, name="")

            self.sess = tf.compat.v1.Session(graph=self.graph)

        else:
            self.model.load_weights(model_path)

    def segmentAsPascalvoc(self, image_path, process_frame=False):
        if model_file == "pb":

            if process_frame == True:
                image = image_path
            else:
                image = cv2.imread(image_path)

            h, w, n = image.shape

            if n > 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            resize_ratio = 1.0 * self.INPUT_SIZE / max(w, h)
            target_size = (int(resize_ratio * w), int(resize_ratio * h))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            batch_seg_map = self.sess.run(
                self.OUTPUT_TENSOR_NAME,
                feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]},
            )

            seg_image = batch_seg_map[0]
            raw_labels = seg_image
            labels = obtain_segmentation(seg_image)
            labels = np.array(Image.fromarray(labels.astype("uint8")).resize((w, h)))
            labels = cv2.cvtColor(labels, cv2.COLOR_RGB2BGR)

            return raw_labels, labels

        else:
            trained_image_width = 512
            mean_subtraction_value = 127.5

            if process_frame == True:
                image = image_path

            else:
                image = np.array(Image.open(image_path))

            # resize to max dimension of images from training dataset
            w, h, n = image.shape

            if n > 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

            ratio = float(trained_image_width) / np.max([w, h])
            resized_image = np.array(
                Image.fromarray(image.astype("uint8")).resize(
                    (int(ratio * h), int(ratio * w))
                )
            )
            resized_image = (resized_image / mean_subtraction_value) - 1

            # pad array to square image to match training images
            pad_x = int(trained_image_width - resized_image.shape[0])
            pad_y = int(trained_image_width - resized_image.shape[1])
            resized_image = np.pad(
                resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode="constant"
            )

            # run prediction
            res = self.model.predict(np.expand_dims(resized_image, 0))

            labels = np.argmax(res.squeeze(), -1)
            # remove padding and resize back to original image
            if pad_x > 0:
                labels = labels[:-pad_x]
            if pad_y > 0:
                labels = labels[:, :-pad_y]

            raw_labels = labels

            # Apply segmentation color map
            labels = obtain_segmentation(labels)
            labels = np.array(Image.fromarray(labels.astype("uint8")).resize((h, w)))

            new_img = cv2.cvtColor(labels, cv2.COLOR_RGB2BGR)

            return raw_labels, new_img

    ####FUNCTION TO FILTER OUT OTHER DETECTIONS AND ENSURE THE DETECTION OF A DISTINCT OBJECT####

    def target_obj(self, segment):

        self.segment = segment
        if segment == "person":
            segment = (255, 255, 255)
            return segment

        if segment == "car":
            segment = (128, 128, 128)
            return segment

        if segment == "aeroplane":
            segment = (128, 0, 0)
            return segment

        if segment == "bicycle":
            segment = (0, 128, 0)
            return segment

        if segment == "bird":
            segment = (255, 255, 255)
            return segment

        if segment == "boat":
            segment = (0, 0, 128)
            return segment

        if segment == "bottle":
            segment = (128, 0, 128)
            return segment

        if segment == "bus":
            segment = (0, 128, 128)
            return segment

        if segment == "cat":
            segment = (64, 0, 0)
            return segment

        if segment == "chair":
            segment = (192, 0, 0)
            return segment

        if segment == "cow":
            segment = (64, 128, 0)
            return segment

        if segment == "diningtable":
            segment = (192, 128, 0)
            return segment

        if segment == "dog":
            segment = (64, 0, 128)
            return segment

        if segment == "horse":
            segment = (192, 0, 128)
            return segment

        if segment == "motorbike":
            segment = (64, 128, 128)
            return segment

        if segment == "pottedplant":
            segment = (0, 64, 0)
            return segment

        if segment == "sheep":
            segment = (128, 64, 0)
            return segment

        if segment == "sofa":
            segment = (0, 192, 0)
            return segment

        if segment == "train":
            segment = (128, 192, 0)
            return segment

        if segment == "monitor":
            segment = (0, 64, 128)
            return segment

    #### ALTER IMAGE BACKGROUND WITH A NEW PICTURE ###

    def change_bg_img(
        self,
        f_image_path,
        b_image_path,
        output_image_name=None,
        verbose=None,
        detect=None,
    ):
        if verbose is not None:
            print("processing image......")

        seg_image = self.segmentAsPascalvoc(f_image_path)

        if detect is not None:
            target_class = self.target_obj(detect)
            seg_image[1][seg_image[1] != target_class] = 0

        ori_img = cv2.imread(f_image_path)

        bg_img = cv2.imread(b_image_path)
        w, h, _ = ori_img.shape
        bg_img = cv2.resize(bg_img, (h, w))

        result = np.where(seg_image[1], ori_img, bg_img)
        if output_image_name is not None:
            cv2.imwrite(output_image_name, result)

        return result

    def change_frame_bg(self, frame, b_image_path, verbose=None, detect=None):
        if verbose is not None:
            print("processing frame......")

        seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)

        if detect is not None:
            target_class = self.target_obj(detect)
            seg_frame[1][seg_frame[1] != target_class] = 0

        bg_img = cv2.imread(b_image_path)
        w, h, _ = frame.shape
        bg_img = cv2.resize(bg_img, (h, w))

        result = np.where(seg_frame[1], frame, bg_img)

        return result

    ## CREATE A VIRTUAL BACKGROUND FOR A VIDEO USING AN IMAGE ##

    def change_video_bg(
        self,
        video_path,
        b_image_path,
        frames_per_second=None,
        output_video_name=None,
        detect=None,
    ):
        capture = cv2.VideoCapture(video_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frames_per_second is not None:
            save_video = cv2.VideoWriter(
                output_video_name,
                cv2.VideoWriter_fourcc(*"DIVX"),
                frames_per_second,
                (width, height),
            )

        counter = 0
        start = time.time()

        while True:
            counter += 1
            ret, frame = capture.read()
            if ret:

                seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)
                print("No. of frames:", counter)
                if detect is not None:
                    target_class = self.target_obj(detect)
                    seg_frame[1][seg_frame[1] != target_class] = 0
                w, h, _ = seg_frame[1].shape
                img = cv2.imread(b_image_path)
                img = cv2.resize(img, (h, w))
                out = np.where(seg_frame[1], frame, img)

                output = cv2.resize(out, (width, height), interpolation=cv2.INTER_AREA)
                if output_video_name is not None:
                    save_video.write(output)

            else:
                break

        capture.release()

        end = time.time()
        print(f"Processed {counter} frames in {end-start:.1f} seconds")

        if frames_per_second is not None:
            save_video.release()

        return output

    ## CREATE A VIRTUAL BACKGROUND FOR A CAMERA FEED USING AN IMAGE ##

    def change_camera_bg(
        self,
        cam,
        b_image_path,
        frames_per_second=None,
        check_fps=False,
        show_frames=False,
        frame_name=None,
        verbose=None,
        output_video_name=None,
        detect=None,
    ):
        capture = cam
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_video_name is not None:
            save_video = cv2.VideoWriter(
                output_video_name,
                cv2.VideoWriter_fourcc(*"DIVX"),
                frames_per_second,
                (width, height),
            )

        counter = 0
        start = datetime.now()

        while True:
            ret, frame = capture.read()
            if ret:
                seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)

                if detect is not None:
                    target_class = self.target_obj(detect)
                    seg_frame[1][seg_frame[1] != target_class] = 0

                w, h, _ = seg_frame[1].shape
                img = cv2.imread(b_image_path)
                img = cv2.resize(img, (h, w))
                output = np.where(seg_frame[1], frame, img)
                counter += 1

                if show_frames == True:
                    if frame_name is not None:
                        cv2.imshow(frame_name, output)
                        if cv2.waitKey(25) & 0xFF == ord("q"):
                            break

                if output_video_name is not None:
                    output = cv2.resize(
                        output, (width, height), interpolation=cv2.INTER_AREA
                    )
                    save_video.write(output)

            elif counter == 30:

                break

        end = datetime.now()

        if check_fps == True:
            timetaken = (end - start).total_seconds()
            fps = counter / timetaken
            print(f"{fps} frames per seconds")

        capture.release()

        if verbose is not None:
            print(f"Processed {counter} frames in {timetaken:.1f} seconds")

        if output_video_name is not None:
            save_video.release()

        return output

    ##### GIVE THE BACKGROUND OF AN IMAGE A DISTINCT COLOR ######

    def color_bg(
        self, image_path, colors, output_image_name=None, verbose=None, detect=None
    ):
        if verbose is not None:
            print("processing image......")

        seg_image = self.segmentAsPascalvoc(image_path)
        if detect is not None:
            target_class = self.target_obj(detect)
            seg_image[1][seg_image[1] != target_class] = 0

        ori_img = cv2.imread(image_path)

        obtain_img = cv2.subtract(seg_image[1], ori_img)
        out = cv2.subtract(seg_image[1], obtain_img)

        out[np.where((out == [0, 0, 0]).all(axis=2))] = [colors]
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        result = np.where(seg_image[1], ori_img, out)

        if output_image_name is not None:
            cv2.imwrite(output_image_name, result)

        return result

    ##### GIVE THE BACKGROUND OF A FRAME A DISTINCT COLOR ######

    def color_frame(self, frame, colors, verbose=None, detect=None):
        if verbose is not None:
            print("processing frame....")

        seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)

        if detect is not None:
            target_class = self.target_obj(detect)
            seg_frame[1][seg_frame[1] != target_class] = 0

        obtain_frame = cv2.subtract(seg_frame[1], frame)
        out = cv2.subtract(seg_frame[1], obtain_frame)
        out[np.where((out == [0, 0, 0]).all(axis=2))] = [colors]
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        result = np.where(seg_frame[1], frame, out)

        return result

    ##### GIVE THE BACKGROUND OF A VIDEO A DISTINCT COLOR ######

    def color_video(
        self,
        video_path,
        colors,
        frames_per_second=None,
        output_video_name=None,
        detect=None,
    ):
        capture = cv2.VideoCapture(video_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frames_per_second is not None:
            save_video = cv2.VideoWriter(
                output_video_name,
                cv2.VideoWriter_fourcc(*"DIVX"),
                frames_per_second,
                (width, height),
            )
        counter = 0
        start = time.time()

        while True:
            counter += 1
            ret, frame = capture.read()
            if ret:
                seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)

                if detect is not None:
                    target_class = self.target_obj(detect)
                    seg_frame[1][seg_frame[1] != target_class] = 0

                print("No. of frames:", counter)
                obtain_frame = cv2.subtract(seg_frame[1], frame)
                out = cv2.subtract(seg_frame[1], obtain_frame)

                out[np.where((out == [0, 0, 0]).all(axis=2))] = [colors]
                out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                out = np.where(seg_frame[1], frame, out)

                output = cv2.resize(out, (width, height), interpolation=cv2.INTER_AREA)
                if output_video_name is not None:
                    save_video.write(output)

            else:

                break

        capture.release()

        end = time.time()
        print(f"Processed {counter} frames in {end-start:.1f} seconds")

        if frames_per_second is not None:
            save_video.release()

        return output

    ##### GIVE THE BACKGROUND OF A CAMERA FRAME A DISTINCT COLOR ######

    def color_camera(
        self,
        cam,
        colors,
        frames_per_second=None,
        check_fps=False,
        show_frames=False,
        frame_name=None,
        verbose=None,
        output_video_name=None,
        detect=None,
    ):
        capture = cam
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_video_name is not None:
            save_video = cv2.VideoWriter(
                output_video_name,
                cv2.VideoWriter_fourcc(*"DIVX"),
                frames_per_second,
                (width, height),
            )
        counter = 0
        start = datetime.now()

        while True:

            ret, frame = capture.read()
            if ret:
                seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)

                if detect is not None:
                    target_class = self.target_obj(detect)
                    seg_frame[1][seg_frame[1] != target_class] = 0

                obtain_frame = cv2.subtract(seg_frame[1], frame)
                out = cv2.subtract(seg_frame[1], obtain_frame)

                out[np.where((out == [0, 0, 0]).all(axis=2))] = [colors]
                out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                output = np.where(seg_frame[1], frame, out)
                counter += 1

                if show_frames == True:
                    if frame_name is not None:
                        cv2.imshow(frame_name, output)
                        if cv2.waitKey(25) & 0xFF == ord("q"):
                            break

                if output_video_name is not None:
                    output = cv2.resize(
                        output, (width, height), interpolation=cv2.INTER_AREA
                    )
                    save_video.write(output)

            elif counter == 30:

                break
        end = datetime.now()
        if check_fps == True:
            timetaken = (end - start).total_seconds()
            fps = counter / timetaken
            print(f"{fps} frames per seconds")

        capture.release()

        if verbose is not None:
            print(f"Processed {counter} frames in {timetaken:.1f} seconds")

        if output_video_name is not None:
            save_video.release()

        return output

    ##### BLUR THE BACKGROUND OF AN IMAGE #####

    def blur_bg(
        self,
        image_path,
        low=False,
        moderate=False,
        extreme=False,
        output_image_name=None,
        verbose=None,
        detect=None,
    ):
        if verbose is not None:
            print("processing image......")

        seg_image = self.segmentAsPascalvoc(image_path)

        if detect is not None:
            target_class = self.target_obj(detect)
            seg_image[1][seg_image[1] != target_class] = 0

        ori_img = cv2.imread(image_path)

        if low == True:
            blur_img = cv2.blur(ori_img, (21, 21), 0)

        if moderate == True:
            blur_img = cv2.blur(ori_img, (39, 39), 0)

        if extreme == True:
            blur_img = cv2.blur(ori_img, (81, 81), 0)

        out = np.where(seg_image[1], ori_img, blur_img)

        if output_image_name is not None:
            cv2.imwrite(output_image_name, out)

        return out

    ##### BLUR THE BACKGROUND OF A FRAME #####

    def blur_frame(
        self, frame, low=False, moderate=False, extreme=False, verbose=None, detect=None
    ):
        if verbose is not None:
            print("processing frame......")

        seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)
        if detect is not None:
            target_class = self.target_obj(detect)
            seg_frame[1][seg_frame[1] != target_class] = 0

        if low == True:
            blur_frame = cv2.blur(frame, (21, 21), 0)

        if moderate == True:
            blur_frame = cv2.blur(frame, (39, 39), 0)

        if extreme == True:
            blur_frame = cv2.blur(frame, (81, 81), 0)

        result = np.where(seg_frame[1], frame, blur_frame)

        return result

    ####  BLUR THE BACKGROUND OF A VIDEO #####

    def blur_video(
        self,
        video_path,
        low=False,
        moderate=False,
        extreme=False,
        frames_per_second=None,
        output_video_name=None,
        detect=None,
    ):
        capture = cv2.VideoCapture(video_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frames_per_second is not None:
            save_video = cv2.VideoWriter(
                output_video_name,
                cv2.VideoWriter_fourcc(*"DIVX"),
                frames_per_second,
                (width, height),
            )

        counter = 0
        start = time.time()

        while True:
            counter += 1
            ret, frame = capture.read()
            if ret:

                seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)
                print("No. of frames:", counter)
                if detect is not None:
                    target_class = self.target_obj(detect)
                    seg_frame[1][seg_frame[1] != target_class] = 0

                if low == True:
                    blur_frame = cv2.blur(frame, (21, 21), 0)

                if moderate == True:
                    blur_frame = cv2.blur(frame, (39, 39), 0)

                if extreme == True:
                    blur_frame = cv2.blur(frame, (81, 81), 0)

                out = np.where(seg_frame[1], frame, blur_frame)

                output = cv2.resize(out, (width, height), interpolation=cv2.INTER_AREA)
                if output_video_name is not None:
                    save_video.write(output)

            else:
                break

        capture.release()

        end = time.time()
        print(f"Processed {counter} frames in {end-start:.1f} seconds")

        if frames_per_second is not None:
            save_video.release()

        return output

    ##### BLUR THE BACKGROUND OF A CAMERA FRAME ######

    def blur_camera(
        self,
        cam,
        low=False,
        moderate=False,
        extreme=False,
        frames_per_second=None,
        check_fps=False,
        show_frames=False,
        frame_name=None,
        verbose=None,
        output_video_name=None,
        detect=None,
    ):
        capture = cam
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_video_name is not None:
            save_video = cv2.VideoWriter(
                output_video_name,
                cv2.VideoWriter_fourcc(*"DIVX"),
                frames_per_second,
                (width, height),
            )

        counter = 0
        start = datetime.now()

        while True:

            ret, frame = capture.read()
            if ret:

                seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)
                if detect is not None:
                    target_class = self.target_obj(detect)
                    seg_frame[1][seg_frame[1] != target_class] = 0

                if low == True:
                    blur_frame = cv2.blur(frame, (21, 21), 0)

                if moderate == True:
                    blur_frame = cv2.blur(frame, (39, 39), 0)

                if extreme == True:
                    blur_frame = cv2.blur(frame, (81, 81), 0)

                output = np.where(seg_frame[1], frame, blur_frame)
                counter += 1

                if show_frames == True:
                    if frame_name is not None:
                        cv2.imshow(frame_name, output)
                        if cv2.waitKey(25) & 0xFF == ord("q"):
                            break

                if output_video_name is not None:
                    output = cv2.resize(
                        output, (width, height), interpolation=cv2.INTER_AREA
                    )
                    save_video.write(output)

            elif couter == 30:
                break

        end = datetime.now()
        if check_fps == True:
            timetaken = (end - start).total_seconds()
            fps = counter / timetaken
            print(f"{fps} frames per seconds")

        capture.release()

        if verbose is not None:
            print(f"Processed {counter} frames in {timetaken:.1f} seconds")

        if output_video_name is not None:
            save_video.release()

        return output

    ### GRAYSCALE THE BACKGROUND OF AN IMAGE ###

    def gray_bg(self, image_path, output_image_name=None, verbose=None, detect=None):
        if verbose is not None:
            print("processing image......")

        seg_image = self.segmentAsPascalvoc(image_path)

        if detect is not None:
            target_class = self.target_obj(detect)
            seg_image[1][seg_image[1] != target_class] = 0

        ori_img = cv2.imread(image_path)

        gray_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        result = np.where(seg_image[1], ori_img, gray_img)

        if output_image_name is not None:
            cv2.imwrite(output_image_name, result)

        return result

    def gray_frame(self, frame, verbose=None, detect=None):
        if verbose is not None:
            print("processing frame......")

        seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)
        if detect is not None:
            target_class = self.target_obj(detect)
            seg_frame[1][seg_frame[1] != target_class] = 0

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        result = np.where(seg_frame[1], frame, gray_frame)

        return result

    ### GRAYSCALE THE BACKGROUND OF A VIDEO ###

    def gray_video(
        self, video_path, frames_per_second=None, output_video_name=None, detect=None
    ):
        capture = cv2.VideoCapture(video_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frames_per_second is not None:
            save_video = cv2.VideoWriter(
                output_video_name,
                cv2.VideoWriter_fourcc(*"DIVX"),
                frames_per_second,
                (width, height),
            )

        counter = 0
        start = time.time()

        while True:
            counter += 1
            ret, frame = capture.read()
            if ret:
                seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)
                if detect is not None:
                    target_class = self.target_obj(detect)
                    seg_frame[1][seg_frame[1] != target_class] = 0

                print("No. of frames:", counter)

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
                out = np.where(seg_frame[1], frame, gray_frame)

                output = cv2.resize(out, (width, height), interpolation=cv2.INTER_AREA)
                if output_video_name is not None:
                    save_video.write(output)

            else:
                break

        capture.release()

        end = time.time()
        print(f"Processed {counter} frames in {end-start:.1f} seconds")

        if frames_per_second is not None:
            save_video.release()

        return output

    ### GRAYSCALE THE BACKGROUND OF A CAMERA FEED ###

    def gray_camera(
        self,
        cam,
        frames_per_second=None,
        check_fps=False,
        show_frames=False,
        frame_name=None,
        verbose=None,
        output_video_name=None,
        detect=None,
    ):
        capture = cam
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_video_name is not None:
            save_video = cv2.VideoWriter(
                output_video_name,
                cv2.VideoWriter_fourcc(*"DIVX"),
                frames_per_second,
                (width, height),
            )
        counter = 0
        start = datetime.now()

        while True:

            ret, frame = capture.read()
            if ret:
                seg_frame = self.segmentAsPascalvoc(frame, process_frame=True)
                if detect is not None:
                    target_class = self.target_obj(detect)
                    seg_frame[1][seg_frame[1] != target_class] = 0

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
                output = np.where(seg_frame[1], frame, gray_frame)
                counter += 1

                if show_frames == True:
                    if frame_name is not None:
                        cv2.imshow(frame_name, output)
                        if cv2.waitKey(25) & 0xFF == ord("q"):
                            break

                if output_video_name is not None:
                    output = cv2.resize(
                        output, (width, height), interpolation=cv2.INTER_AREA
                    )
                    save_video.write(output)

            elif counter == 30:

                break

        end = datetime.now()
        if check_fps == True:
            timetaken = (end - start).total_seconds()
            fps = counter / timetaken
            print(f"{fps} frames per seconds")

        capture.release()

        if verbose is not None:
            print(f"Processed {counter} frames in {timetaken:.1f} seconds")

        if output_video_name is not None:
            save_video.release()

        return output
