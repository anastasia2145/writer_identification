from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import math_ops

def contrastive_loss_(labels, embeddings_anchor, embeddings_positive, margin=1.0):

    distances = math_ops.sqrt(
        math_ops.reduce_sum(math_ops.square(embeddings_anchor - embeddings_positive), 1))

    return math_ops.reduce_mean(
        math_ops.to_float(labels) * math_ops.square(distances) +
        (1. - math_ops.to_float(labels)) *
        math_ops.square(math_ops.maximum(margin - distances, 0.)),
        name='contrastive_loss')
            