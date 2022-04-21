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
"""Circle Loss
    Circle Loss: A Unified Perspective of Pair Similarity Optimization
    https://arxiv.org/abs/2002.10857
"""

import tensorflow as tf
from typing import Any, Callable, Union

from tensorflow_similarity.distances import Distance, distance_canonicalizer
from tensorflow_similarity.types import FloatTensor, IntTensor
from .circle_loss import circle_loss
from .metric_loss import MetricLoss
from .utils import global_orthogonal_regularization


@tf.keras.utils.register_keras_serializable(package="Similarity")
@tf.function
def regularized_circle_loss(labels: IntTensor,
                embeddings: FloatTensor,
                distance: Callable,
                gamma: float = 80,
                margin: float = 0.4,
                alpha: float = 1.0) -> Any:
    c_loss = circle_loss(labels, embeddings, distance, gamma, margin)
    r_loss = global_orthogonal_regularization(labels, embeddings)
    r_loss = tf.math.multiply(r_loss, alpha)
    return tf.math.add(c_loss, r_loss)


@tf.keras.utils.register_keras_serializable(package="Similarity")
class RegularizedCircleLoss(MetricLoss):
    """Computes the RegularizedCircleLoss

    Circle Loss: A Unified Perspective of Pair Similarity Optimization
    https://arxiv.org/abs/2002.10857

    The original paper used cosine similarity while this loss has been
    modified to work with cosine distance.

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
                 gamma: float = 80.0,
                 margin: float = 0.40,
                 alpha: float = 1.0,
                 name: str = 'RegularizedCircleLoss',
                 **kwargs):
        """Initializes a CircleLoss

        Args:
            distance: Which distance function to use to compute the pairwise
            distances between embeddings. The distance is expected to be
            between [0, 2]. Defaults to 'cosine'.

            gamma: Scaling term. Defaults to 80. Note: Large values cause the
            LogSumExp to return the Max pair and reduces the weighted mixing
            of all pairs. Should be hypertuned.

            margin: Used to weight the distance. Below this distance, negatives
            are up weighted and positives are down weighted. Similarly, above
            this distance negatives are down weighted and positive are up
            weighted. Defaults to 0.4.

            alpha: weight parametrizing the size of the contribution from the regularization loss terms.

            name: Loss name. Defaults to RegularizedCircleLoss.
        """

        # distance canonicalization
        distance = distance_canonicalizer(distance)
        self.distance = distance

        super().__init__(regularized_circle_loss,
                         name=name,
                         distance=distance,
                         gamma=gamma,
                         margin=margin,
                         alpha=alpha,
                         **kwargs)
