# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Multi Similarity Loss"""

import tensorflow as tf
from typing import Any, Callable, Union

from tensorflow_similarity.distances import Distance, distance_canonicalizer
from tensorflow_similarity.algebra import build_masks, masked_max, masked_min
from tensorflow_similarity.types import FloatTensor, IntTensor
from .metric_loss import MetricLoss
from .utils import logsumexp


@tf.keras.utils.register_keras_serializable(package="Similarity")
@tf.function
def mueller_smola_loss(labels: IntTensor,
                       embeddings: FloatTensor,
                       distance: Callable) -> Any:
    """Mueller-Smola loss computations

    Args:
        labels: labels associated with the embed.

        embeddings: Embedded examples.

        distance: Which distance function to use to compute the pairwise.

    Returns:
        Loss: The loss value for the current batch.
    """
    # [Label]
    # ! Weirdness to be investigated
    # do not remove this code. It is actually needed for specific situation
    # Reshape label tensor to [batch_size, 1] if not already in that format.
    # labels = tf.reshape(labels, (labels.shape[0], 1))
    batch_size = tf.size(labels)

    # [distances]
    pairwise_distances = distance(embeddings)
    match_probabilities = tf.math.exp(-pairwise_distances)

    # [masks]
    positive_mask, negative_mask = build_masks(labels, batch_size)
    positive_mask = tf.cast(positive_mask, dtype='float32')
    negative_mask = tf.cast(negative_mask, dtype='float32')

    # [compute loss]

    # positive examples
    p_loss = pairwise_distances * positive_mask

    # negative examples
    one_minus_match_probabilities = tf.maximum(1.0 - match_probabilities, 1.0e-18)
    n_loss = -tf.math.log(one_minus_match_probabilities) * negative_mask

    # reduce and scale loss so it isn't a function of the batch size.
    mueller_smola_loss = tf.math.reduce_mean(p_loss + n_loss)

    return mueller_smola_loss


@tf.keras.utils.register_keras_serializable(package="Similarity")
class MuellerSmolaLoss(MetricLoss):
    """Computes the multi similarity loss in an online fashion.


    `y_true` must be  a 1-D integer `Tensor` of shape (batch_size,).
    It's values represent the classes associated with the examples as
    **integer  values**.

    `y_pred` must be 2-D float `Tensor`  of L2 normalized embedding vectors.
    you can use the layer `tensorflow_similarity.layers.L2Embedding()` as the
    last layer of your model to ensure your model output is properly
    normalized.
    """

    def __init__(self,
                 distance: Union[Distance, str] = 'cosine',
                 name: str = 'MuellerSmolaLoss'):
        """Initializes the Multi Similarity Loss

        Args:
            distance: Which distance function to use to compute the pairwise
            distances between embeddings. Defaults to 'cosine'.

            alpha: The exponential weight for the positive pairs. Increasing
            alpha makes the logsumexp softmax closer to the max positive pair
            distance, while decreasing it makes it closer to
            max(P) + log(batch_size).

            beta: The exponential weight for the negative pairs. Increasing
            beta makes the logsumexp softmax closer to the max negative pair
            distance, while decreasing it makes the softmax closer to
            max(N) + log(batch_size).

            epsilon: Used to remove easy positive and negative pairs. We only
            keep positives that we greater than the (smallest negative pair -
            epsilon) and we only keep negatives that are less than the
            (largest positive pair + epsilon).

            lmda: Used to weight the distance. Below this distance, negatives
            are up weighted and positives are down weighted. Similarly, above
            this distance negatives are down weighted and positive are up
            weighted.

            name: Loss name. Defaults to MultiSimilarityLoss.
        """

        # distance canonicalization
        distance = distance_canonicalizer(distance)
        self.distance = distance

        super().__init__(mueller_smola_loss,
                         name=name,
                         reduction=tf.keras.losses.Reduction.NONE,
                         distance=distance)
