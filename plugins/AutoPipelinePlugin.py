from phy import IPlugin, connect
import numpy as np
import logging

logger = logging.getLogger("phy")


class AutoPipelinePlugin(IPlugin):
    def attach_to_controller(self, controller):
        """Attach actions to the GUI controller."""
        @connect
        def on_gui_ready(sender, gui):
            @controller.supervisor.actions.add(shortcut="alt+a")
            def run_auto_pipeline():
                """Run the auto pipeline for splitting clusters."""
                logger.info("Starting automated pipeline...")

                try:
                    # Step 1: Splitting short-ISI spikes
                    logger.info("Step 1: Splitting short-ISI spikes.")
                    short_isi_sec = 0.002
                    for cid in controller.supervisor.clustering.cluster_ids:
                        spike_ids = controller.supervisor.clustering.spikes_per_cluster[cid]
                        if len(spike_ids) < 2:
                            continue
                        spike_ids = np.array(spike_ids)  # Ensure this is a NumPy array
                        spike_times = controller.model.spike_times[spike_ids]
                        order = np.argsort(spike_times)
                        spike_times = spike_times[order]
                        spike_ids = spike_ids[order]
                        isi = np.diff(spike_times)
                        short_idx = np.where(isi < short_isi_sec)[0] + 1
                        if len(short_idx) > 0:
                            labels = np.ones(len(spike_ids), dtype=np.int64)
                            labels[short_idx] = 2
                            controller.supervisor.actions.split(spike_ids.tolist(), labels.tolist())
                            logger.info(f"Split {len(short_idx)} short-ISI spikes in cluster {cid}.")

                    # Step 2: Mahalanobis Distance Splitting
                    logger.info("Step 2: Splitting clusters using Mahalanobis distance.")
                    threshold = 10  # Adjust threshold as needed
                    for cid in controller.supervisor.clustering.cluster_ids:
                        spike_ids = controller.supervisor.clustering.spikes_per_cluster[cid]
                        if len(spike_ids) < 2:
                            logger.warning(f"Cluster {cid} has too few spikes for Mahalanobis distance calculation.")
                            continue
                        spike_ids = np.array(spike_ids)  # Ensure this is a NumPy array
                        data = controller.model._load_features().data[spike_ids]
                        reshaped_data = np.reshape(data, (data.shape[0], -1))
                        if reshaped_data.shape[0] < reshaped_data.shape[1]:
                            logger.warning(f"Cluster {cid} has too few spikes for Mahalanobis distance calculation.")
                            continue

                        try:
                            covariance_matrix = np.cov(reshaped_data, rowvar=False)
                            if np.linalg.det(covariance_matrix) == 0:
                                logger.warning(f"Cluster {cid}: Singular covariance matrix, skipping.")
                                continue
                            inv_covariance_matrix = np.linalg.inv(covariance_matrix)
                            mean_vec = np.mean(reshaped_data, axis=0)
                            mahalanobis_distances = np.array([
                                np.sqrt((x - mean_vec).T @ inv_covariance_matrix @ (x - mean_vec))
                                for x in reshaped_data
                            ])
                            outlier_indices = np.where(mahalanobis_distances > threshold)[0]
                            if len(outlier_indices) > 0:
                                labels = np.ones(len(spike_ids), dtype=np.int64)
                                labels[outlier_indices] = 2
                                controller.supervisor.actions.split(spike_ids.tolist(), labels.tolist())
                                logger.info(f"Split {len(outlier_indices)} Mahalanobis outliers in cluster {cid}.")
                        except Exception as e:
                            logger.error(f"Mahalanobis distance calculation failed for cluster {cid}: {e}")
                            continue

                    logger.info("Automated pipeline steps completed.")

                except Exception as e:
                    logger.error(f"Error during automated pipeline: {e}")
