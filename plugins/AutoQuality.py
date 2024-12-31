"""AutoLabel Plugin for Phy 2.0b6."""

from phy import IPlugin, connect
import numpy as np
import logging

logger = logging.getLogger('phy')


class AutoQuality(IPlugin):
    def attach_to_controller(self, controller):
        """Attach the AutoLabel action to the controller."""
        @connect
        def on_gui_ready(sender, gui):
            @controller.supervisor.actions.add(shortcut='shift+l')
            def Auto_Quality():
                """Automatically label clusters as 'good' or 'noise' based on ISI and spike count."""
                logger.info("Starting automatic cluster labeling...")

                try:
                    # Define criteria
                    short_isi_sec = 0.002  # 2 ms
                    isi_threshold = 0.01    # 1%
                    min_spike_count = 1000  # Minimum number of spikes to consider as 'good'

                    # Retrieve all unique cluster IDs
                    cluster_ids = controller.supervisor.clustering.cluster_ids
                    unique_clusters = np.unique(cluster_ids)

                    logger.debug(f"Found {len(unique_clusters)} unique clusters.")

                    # Initialize a dictionary to store cluster IDs and their new quality labels
                    quality_labels = {}

                    # Iterate over each cluster
                    for cid in unique_clusters:
                        spike_ids = controller.supervisor.clustering.spikes_per_cluster[cid]
                        spike_count = len(spike_ids)

                        if spike_count < min_spike_count:
                            # Label as 'noise' due to insufficient spike count
                            quality_labels[cid] = 'noise'
                            logger.info(f"Cluster {cid} has {spike_count} spikes (< {min_spike_count}). Marking as 'noise'.")
                            continue

                        spike_times = controller.model.spike_times[spike_ids]

                        if spike_count < 2:
                            # Not enough spikes to compute ISI
                            quality_labels[cid] = 'noise'
                            logger.warning(f"Cluster {cid} has fewer than 2 spikes. Marking as 'noise'.")
                            continue

                        # Calculate Interspike Intervals (ISI)
                        isi = np.diff(np.sort(spike_times))
                        if len(isi) == 0:
                            short_isi_ratio = 0
                        else:
                            short_isi_count = np.sum(isi < short_isi_sec)
                            short_isi_ratio = short_isi_count / len(isi)

                        # Determine cluster quality based on ISI ratio
                        if short_isi_ratio > isi_threshold:
                            quality_labels[cid] = 'noise'
                            logger.info(f"Cluster {cid} marked as 'noise' ({short_isi_ratio*100:.2f}% short ISIs).")
                        else:
                            quality_labels[cid] = 'good'
                            logger.info(f"Cluster {cid} marked as 'good' ({short_isi_ratio*100:.2f}% short ISIs).")

                    # Apply labels to clusters by updating the 'quality' field in metadata
                    if quality_labels:
                        # Initialize 'quality' in metadata if it doesn't exist
                        if 'quality' not in controller.model.metadata:
                            controller.model.metadata['quality'] = {}
                            logger.debug("Initialized 'quality' field in metadata.")

                        # Update the 'quality' dictionary with new labels
                        controller.model.metadata['quality'].update(quality_labels)
                        logger.info(f"Assigned 'quality' labels to {len(quality_labels)} clusters.")

                        # Persist the changes by saving the 'quality' metadata
                        controller.model.save_metadata('quality', controller.model.metadata['quality'])
                        logger.info("Metadata saved successfully.")

                    logger.info("Automatic cluster labeling completed.")

                except AttributeError as ae:
                    logger.error(f"Attribute error during automatic cluster labeling: {ae}")
                except ZeroDivisionError as zde:
                    logger.error(f"Division by zero error during automatic cluster labeling: {zde}")
                except Exception as e:
                    logger.error(f"Unexpected error during automatic cluster labeling: {e}")
