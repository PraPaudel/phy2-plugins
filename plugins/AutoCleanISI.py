"""ISI Splitter Plugin for Phy."""

from phy import IPlugin, connect
import numpy as np
import logging

logger = logging.getLogger('phy')


class AutoCleanISI(IPlugin):
    def attach_to_controller(self, controller):
        """Attach the ISI splitter action to the controller."""
        @connect
        def on_gui_ready(sender, gui):
            @controller.supervisor.actions.add(shortcut='shift+i')
            def Auto_Clean_ISI():
                """Split clusters based on short ISI spikes."""
                logger.info("Starting ISI-based splitting...")

                try:
                    short_isi_sec = 0.0015
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

                    logger.info("ISI-based splitting completed.")

                except Exception as e:
                    logger.error(f"Error during ISI-based splitting: {e}")
