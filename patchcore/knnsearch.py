"""
This module provides a k-NN searcher class using faiss backend,
and sampling algorithms which returns a set of points that minimizes
the maximum distance of any point to a center.
"""

# Import third-party packages.
import faiss
import numpy as np
import rich
import sklearn.metrics
import sklearn.random_projection


class KCenterGreedy:
    """
    Python implementation of the k-Center-Greedy method in [1].

    Distance metric defaults to L2 distance. Features used to calculate distance
    are either raw features or if a model has transform method then uses the output
    of model.transform(X).

    This algorithm can be extended to a robust k centers algorithm that ignores
    a certain number of outlier datapoints. Resulting centers are solution to
    multiple integer programing problem.

    Reference:
        [1] O. Sener and S. Savarese, "A Geometric Approach to Active Learning for
            Convolutional Neural Networks", arXiv, 2017.
            <https://arxiv.org/abs/1708.00489>

    Notes:
        This code originally comesfrom the following code written by Google
        which is released under the Apache License 2.0 (as of Jan 25, 2022):
        
        <https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py>
        <https://github.com/google/active-learning/blob/master/sampling_methods/sampling_def.py>
        
        The following is the description of the license applied to these code.
        ---
        Copyright 2017 Google Inc.
        
        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at
        
             http://www.apache.org/licenses/LICENSE-2.0
        
        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.
    """
    def __init__(self, X, y, seed, metric="euclidean"):
        self.X = X
        self.y = y
        self.name = "kcenter"
        self.metric = metric
        self.min_distances = None
        self.n_obs = self.X.shape[0]
        self.already_selected = []

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """
        Update min distances given cluster centers.

        Args:
          cluster_centers (list): indices of cluster centers
          only_new (bool): only calculate distance for newly selected points
                           and update min_distances.
          rest_dist (bool): whether to reset min_distances variable.
        """
        if reset_dist:
            self.min_distances = None

        if only_new:
            cluster_centers = [d for d in cluster_centers if d not in self.already_selected]

        if cluster_centers:

            # Update min_distances for all examples given new cluster center.
            x = self.features[cluster_centers]
            dist = sklearn.metrics.pairwise_distances(self.features, x, metric=self.metric)

            if self.min_distances is None:
              self.min_distances = np.min(dist, axis=1).reshape(-1,1)
            else:
              self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch(self, model, already_selected, N, **kwargs):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.

        Args:
          model: model with scikit-like API with decision_function implemented
          already_selected: index of datapoints already selected
          N: batch size

        Returns:
          indices of points selected to minimize distance to cluster centers
        """
        # Assumes that the transform function takes in original data and not flattened data.
        if model is not None: self.features = model.transform(self.X)
        else                : self.features = self.X.reshape((self.X.shape[0], -1))

        # Compute distances.
        self.update_distances(already_selected, only_new=False, reset_dist=True)

        # Initialize sampling results.
        new_batch = []

        for _ in rich.progress.track(range(N), description="Sampling..."):

            # Initialize centers with a randomly selected datapoint
            if self.already_selected is None:
                ind = np.random.choice(np.arange(self.n_obs))

            # Otherwise, use the index of minimum distance.
            else:
                ind = np.argmax(self.min_distances)

            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in already_selected

            self.update_distances([ind], only_new=True, reset_dist=False)

            new_batch.append(ind)

        # Memorize the already selected indices.
        self.already_selected = already_selected

        # Print summaries.
        rich.print("Maximum distance from cluster centers: [magenta]%0.2f[/magenta]" % max(self.min_distances))
        rich.print("Initial number of features: [magenta]%d[/magenta]" % self.X.shape[0])
        rich.print("Sampled number of features: [magenta]%d[/magenta]" % len(new_batch))

        return new_batch


class KNNSearcher:
    """
    A class for k-NN search with dimention resuction (random projection)
    and subsampling (k-center greedy method) features.
    """
    def __init__(self, projection=True, subsampling=True, sampling_ratio=0.01):
        """
        Constructor of the KNNSearcher class.

        Args:
            projection     (bool) : Enable random projection if true.
            subsampling    (bool) : Enable subsampling if true.
            sampling_ratio (float): Ratio of subsampling.
        """
        self.projection     = projection
        self.subsampling    = subsampling
        self.sampling_ratio = sampling_ratio

    def fit(self, x):
        """
        Train k-NN search model.

        Args:
            x (np.ndarray): Training data of shape (n_samples, n_features).
        """
        # Apply random projection if specified. Random projection is used for reducing
        # dimention while keeping topology. It makes the k-center greedy algorithm faster.
        if self.projection:

            rich.print("Sparse random projection")

            # If number of features is much smaller than the number of samples, random
            # projection will fail due to the lack of number of features. In that case,
            # please increase the parameter `eps`, or just skip the random projection.
            projector = sklearn.random_projection.SparseRandomProjection(n_components="auto", eps=0.90)
            projector.fit(x)

            # Print the shape of random matrix: (n_features_after, n_features_before).
            shape = projector.components_.shape
            rich.print("  - [green]random matrix shape[/green]: [cyan]%s[/cyan]" % str(shape))

        # Set None if random projection is no specified.
        else: projector = None

        # Execute coreset subsampling.
        if self.subsampling:
            rich.print("Coreset subsampling")
            n_select = int(x.shape[0] * self.sampling_ratio)
            selector = KCenterGreedy(x, 0, 0)
            indices  = selector.select_batch(projector, [], n_select)
            x = x[indices, :]

        # Setup nearest neighbour finder using Faiss library.
        self.index = faiss.IndexFlatL2(x.shape[1])
        self.index.add(x) 

    def predict(self, x, k=3):
        """
        Run k-NN search prediction.

        Args:
            x (np.ndarray): Query data of shape (n_samples, n_features)
            k (int)       : Number of neighbors to be searched.

        Returns:
            dists   (np.ndarray): Distance between the query and searched data,
                                  where the shape is (n_samples, n_neighbors).
            indices (np.ndarray): List of indices to be searched of shape
                                  (n_samples, n_neighbors).
        """
        # Faiss searcher requires C-contiguous array as input,
        # therefore forcibly convert the input data to contiguous array.
        x = np.ascontiguousarray(x)

        # Run k-NN search.
        dists, indices = self.index.search(x , k=k)

        return (dists, indices)

    def load(self, filepath):
        self.index = faiss.read_index(filepath)
        # if torch.cuda.is_available():
        #     res = faiss.StandardGpuResources()
        #     self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def save(self, filepath):
        if hasattr(self, "index"): faiss.write_index(self.index, filepath)
        else                     : raise RuntimeError("this model not traint yet")
