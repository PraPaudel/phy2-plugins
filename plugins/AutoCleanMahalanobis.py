"""Mahalanobis Distance Splitter Plugin for Phy."""

from phy import IPlugin, connect
import numpy as np
import logging

logger = logging.getLogger('phy')


class AutoCleanMahalanobis(IPlugin):
    def attach_to_controller(self, controller):
        """Attach the Mahalanobis splitter action to the controller."""
        @connect
        def on_gui_ready(sender, gui):
            @controller.supervisor.actions.add(shortcut='shift+x', prompt=True, prompt_default=lambda: "9")
            def Auto_Clean_Mahalanobis(threshold_str):
                """Split clusters based on Mahalanobis distance outliers.
                
                Provide threshold value (default: 9)
                Only processes clusters with firing rate > 2Hz.
                """
                logger.info("Starting Mahalanobis distance-based splitting...")
                
                # Parse threshold from input
                try:
                    mahalanobis_threshold = float(threshold_str)
                    logger.info(f"Using threshold: {mahalanobis_threshold}")
                except ValueError:
                    mahalanobis_threshold = 9.0
                    logger.info(f"Invalid input. Using default threshold: {mahalanobis_threshold}")
                
                # Fixed firing rate threshold
                firing_rate_thresh = 2.0
                
                try:
                    # Get all cluster IDs and filter by firing rate
                    all_cluster_ids = controller.supervisor.clustering.cluster_ids
                    high_fr_clusters = []
                    spike_times = controller.model.spike_times
                    
                    # Find high firing rate clusters
                    for cid in all_cluster_ids:
                        spike_ids = controller.supervisor.clustering.spikes_in_clusters([cid])
                        if len(spike_ids) > 1:
                            cluster_times = spike_times[spike_ids]
                            duration = cluster_times[-1] - cluster_times[0]
                            if duration > 0:
                                firing_rate = len(cluster_times) / duration
                                if firing_rate > firing_rate_thresh:
                                    high_fr_clusters.append(cid)
                    
                    # Process only high firing rate clusters
                    if not high_fr_clusters:
                        logger.warn(f"No clusters found with firing rate > {firing_rate_thresh} Hz")
                        return
                        
                    logger.info(f"Processing {len(high_fr_clusters)} clusters with firing rate > {firing_rate_thresh} Hz")
                    
                    # Track results
                    total_outliers_found = 0
                    clusters_processed = 0
                    
                    # Process each high firing rate cluster
                    for cid in high_fr_clusters:
                        # Get spikes for this cluster
                        spike_ids = controller.supervisor.clustering.spikes_in_clusters([cid])
                        
                        if len(spike_ids) < 10:
                            continue
                            
                        # Load features
                        data = controller.model._load_features().data[spike_ids]
                        reshaped_data = np.reshape(data, (data.shape[0], -1))
                        
                        if reshaped_data.shape[0] <= reshaped_data.shape[1]:
                            continue

                        # Calculate Mahalanobis distance (using original method)
                        def mahalanobis_dist_calc(X):
                            """Calculate Mahalanobis distance for each sample in X."""
                            mean_vec = np.mean(X, axis=0)
                            try:
                                cov_matrix = np.cov(X, rowvar=False)
                                inv_cov_matrix = np.linalg.inv(cov_matrix)
                            except np.linalg.LinAlgError:
                                logger.error(f"Singular covariance matrix for cluster {cid}. Skipping Mahalanobis calculation.")
                                return np.zeros(X.shape[0])
                            diff = X - mean_vec
                            md = np.sqrt(np.sum(diff @ inv_cov_matrix * diff, axis=1))
                            return md

                        MD = mahalanobis_dist_calc(reshaped_data)
                        outlier_indices = np.where(MD > mahalanobis_threshold)[0]
                        
                        # Split if outliers found
                        if len(outlier_indices) > 0:
                            labels = np.ones(len(spike_ids), dtype=np.int64)
                            labels[outlier_indices] = 2
                            controller.supervisor.actions.split(spike_ids, labels)
                            logger.info(f"Split {len(outlier_indices)} outliers in cluster {cid}")
                            total_outliers_found += len(outlier_indices)
                            clusters_processed += 1

                    logger.info(f"Completed: {total_outliers_found} outliers in {clusters_processed} high firing rate clusters")

                except Exception as e:
                    logger.error(f"Error: {str(e)}")