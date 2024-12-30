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
            @controller.supervisor.actions.add(shortcut='shift+x')
            def Auto_Clean_Mahalanobis():
                """Split clusters based on Mahalanobis distance outliers."""
                logger.info("Starting Mahalanobis distance-based splitting...")

                try:
                    threshold = 10  # Adjust threshold as needed
                    for cid in controller.supervisor.clustering.cluster_ids:
                        spike_ids = controller.supervisor.clustering.spikes_per_cluster[cid]
                        if len(spike_ids) < 2:
                            logger.warning(f"Cluster {cid} has too few spikes for Mahalanobis distance calculation.")
                            continue
                        spike_ids = np.array(spike_ids)
                        data = controller.model._load_features().data[spike_ids]
                        reshaped_data = np.reshape(data, (data.shape[0], -1))
                        if reshaped_data.shape[0] < reshaped_data.shape[1]:
                            logger.warning(f"Cluster {cid} has too few spikes for Mahalanobis distance calculation.")
                            continue

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
                        outlier_indices = np.where(MD > threshold)[0]  # Apply threshold
                        if len(outlier_indices) > 0:
                            labels = np.ones(len(spike_ids), dtype=np.int64)
                            labels[outlier_indices] = 2
                            controller.supervisor.actions.split(spike_ids.tolist(), labels.tolist())
                            logger.info(f"Split {len(outlier_indices)} Mahalanobis outliers in cluster {cid}.")

                    logger.info("Mahalanobis distance-based splitting completed.")

                except Exception as e:
                    logger.error(f"Error during Mahalanobis distance-based splitting: {e}")
