"""Automated Pipeline Plugin for Phy (ISI and Mahalanobis distance)."""

from phy import IPlugin, connect
import numpy as np
import logging

logger = logging.getLogger('phy')


class AutoPipelinePlugin(IPlugin):
    def attach_to_controller(self, controller):
        """Attach the plugin actions to the controller."""
        @connect
        def on_gui_ready(sender, gui):
            @controller.supervisor.actions.add(shortcut='alt+a')
            def run_auto_pipeline():
                """Run the automated pipeline for splitting clusters."""
                logger.info("Starting automated pipeline...")

                try:
                    # Step 1: Split short-ISI spikes.
                    logger.info("Step 1: Splitting short-ISI spikes.")
                    short_isi_sec = 0.002
                    for cid in controller.supervisor.clustering.cluster_ids:
                        spike_ids = controller.supervisor.clustering.spikes_per_cluster[cid]
                        if len(spike_ids) < 2:
                            continue
                        spike_ids = np.array(spike_ids)
                        spike_times = controller.model.spike_times[spike_ids]
                        isi = np.diff(np.sort(spike_times))
                        short_idx = np.where(isi < short_isi_sec)[0] + 1
                        if len(short_idx) > 0:
                            labels = np.ones(len(spike_ids), dtype=np.int64)
                            labels[short_idx] = 2
                            controller.supervisor.actions.split(spike_ids.tolist(), labels.tolist())
                            logger.info(f"Split {len(short_idx)} short-ISI spikes in cluster {cid}.")

                    # Step 2: Split using Mahalanobis distance.
                    logger.info("Step 2: Splitting clusters using Mahalanobis distance.")
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

                        def MahalanobisDistCalc(X):
                            """Mahalanobis distance calculation."""
                            mean_vec = np.mean(X, axis=0)
                            cov_matrix = np.cov(X, rowvar=False)
                            inv_cov_matrix = np.linalg.inv(cov_matrix)
                            diff = X - mean_vec
                            md = np.sqrt(np.sum(diff @ inv_cov_matrix * diff, axis=1))
                            return md

                        MD = MahalanobisDistCalc(reshaped_data)
                        outlier_indices = np.where(MD > threshold)[0]  # Apply threshold
                        if len(outlier_indices) > 0:
                            labels = np.ones(len(spike_ids), dtype=np.int64)
                            labels[outlier_indices] = 2
                            controller.supervisor.actions.split(spike_ids.tolist(), labels.tolist())
                            logger.info(f"Split {len(outlier_indices)} Mahalanobis outliers in cluster {cid}.")

                    logger.info("Automated pipeline steps completed.")

                except Exception as e:
                    logger.error(f"Error during automated pipeline: {e}")
