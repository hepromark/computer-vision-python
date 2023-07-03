import sklearn
import numpy as np
import cluster_estimation_output

class Cluster_estimation:

    def __init__(self, co_type="spherical", components=10, rand_state=0, weight_conc_prior=100, mean_precision_prior=1E-6,
                 min_points=100, new_points_per_run=10):
        # Initalizes VMGMM model 
        self.__vgmm_model = sklearn.mixture.BayesianGaussianMixture(covariance_type=co_type, n_components=components, 
                                                                    random_state=rand_state, weight_concentration_prior=weight_conc_prior, 
                                                                    init_params='k-means++', mean_precision_prior=mean_precision_prior)
        # Points storage 
        self.__all_points = []
        self.__min_points = min_points
        self.__points_per_run = new_points_per_run
        self.__no_points_count = 0
        self.__current_bucket = []
    

    def run(self, points: "list[tuple[float, float]]", run_override: bool = False):
        """
        Run cluster estimation model.

        PARAMETERS
        ----------
        points: list[tuple[float, float]], list of points 
        run_override: bool, whether to run model regardless of number of points available 

        RETURNS
        -------
        bool, if model was ran and had output
        estimation: list[Cluster_estimation_output], formatted cluster positions and covariances. 
        """

        # Not enough points to run 
        if not self.decide_to_run(points, run_override):
            return False, None

        # Get locations and covariances
        good_points = self.cleanup_the_points(self.__all_points)
        num_clusters, locations, covariances = self.vgmm(good_points)

        # Nothing found
        if num_clusters == 0:
            return False, None

        # Format output 
        estimation = []
        for i in range(num_clusters):
            estimation.append(cluster_estimation_output.Cluster_estimation_output(locations[i,0].items(),
                                                                                  locations[i,1].items(),
                                                                                  covariances))

        return True, estimation

    def decide_to_run(self, points, run_override: bool) -> bool:
        """
        Decide to run model depending on minimum total points and new points requirements 
        """
        if run_override:
            return True
        
        if len(points) == 0:
            self.__no_points_count += 1

        # Update current bucket with points not ran by model
        self.__current_bucket += points
        # Don't run if total points < min amount for model or not enough new points
        if len(self.__current_bucket) < self.__points_per_run or len(self.__all_points) < self.__min_points:
            return False

        # Requirements met, empty bucket and run
        self.__all_points += self.__current_bucket
        self.__current_bucket = []

        return True

    @staticmethod
    def cleanup_the_points(points):
        # Future: Impliment input normalization if needed
        good_points = np.array(points)
        return good_points

    def vgmm(self, points: np.array):
        """
        Returns most probable clusters and their covariances from points 
        
        PARAMETERS
        points: np.array [number_of_points, number_of_points], the model input

        RETURNS
        num_viable_clusters: int, the number of clusters detected by model 
        clusters: np.array [number_of_clusters, 2], the cluster coordinate position 
        covariances: np.array [number_of_clusters,], the cluster covriance
        """
        self.__vgmm_model = self.__vgmm_model.fit(points)  # TODO: check continually update current model vs. make new one 
        clusters = self.__vgmm_model.means_
        covariances = self.__vgmm_model.covariances_
        weights = self.__vgmm_model.means_
        
        # Loop through each cluster
        # clusters is a list of centers ordered by weights
        # most likely cluster listed first in descending weights order
        total_clusters = clusters.shape[0]
        num_viable_clusters = 1

        # Drop all clusters after a 50% drop in weight occurs
        while num_viable_clusters < total_clusters:
            if (weights[num_viable_clusters] / weights[num_viable_clusters-1]) > 2:
                break
            num_viable_clusters += 1
        
        # TODO: Decide when to not return any cluster centers at all 

        # return viable clusters 
        return num_viable_clusters, clusters[:num_viable_clusters], covariances[:num_viable_clusters]