# import from plugins/action_status_bar.py
"""Show how to create new actions in the GUI.

The first action just displays a message in the status bar.

The second action selects the first N clusters, where N is a parameter that is entered by
the user in a prompt dialog.

Additionally, added two new actions:
- Ctrl+F: Selects the first (minimum) cluster ID.
- Ctrl+Shift+L: Selects the last (maximum) cluster ID.
"""

from phy import IPlugin, connect
import numpy as np
import logging

logger = logging.getLogger('phy')


class CustomActionPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_gui_ready(sender, gui):

            @controller.supervisor.actions.add(shortcut='ctrl+c')
            def select_first_unsorted():
                # Existing action to select the first unsorted cluster

                @controller.supervisor.cluster_view.get_ids
                def find_unsorted(cluster_ids):
                    """This function is called when the ordered list of cluster ids is returned
                    by the Javascript view."""
                    groups = controller.supervisor.cluster_meta.get('group', list(range(max(cluster_ids) + 1)))
                    for ii in cluster_ids:
                        if groups[ii] is None or groups[ii] == 'unsorted':
                            s = controller.supervisor.clustering.spikes_in_clusters([ii])
                            if len(s) > 0:
                                firstclu = ii
                                break

                    if 'firstclu' in locals():
                        controller.supervisor.select(firstclu)

                    return

            @controller.supervisor.actions.add(shortcut='ctrl+v')
            def move_selected_to_end():
                # Existing action to move selected cluster to end
                logger.warn("Moving selected cluster to end")
                selected = controller.supervisor.selected
                s = controller.supervisor.clustering.spikes_in_clusters(selected)
                outliers2 = np.ones(len(s), dtype=int)
                controller.supervisor.actions.split(s, outliers2)

            @controller.supervisor.actions.add(shortcut='ctrl+b')
            def move_similar_to_end():
                # Existing action to move similar clusters to end
                logger.warn("Moving selected similar cluster to end")
                sim = controller.supervisor.selected_similar
                s = controller.supervisor.clustering.spikes_in_clusters(sim)
                outliers2 = np.ones(len(s), dtype=int)
                controller.supervisor.actions.split(s, outliers2)


            @controller.supervisor.actions.add(shortcut='ctrl+shift+f')
            def select_first_cluster():
                """Selects the cluster with the minimum cluster ID."""
                logger.info("Selecting the first (minimum) cluster ID.")

                @controller.supervisor.cluster_view.get_ids
                def highlight_first(cluster_ids):
                    if not cluster_ids:
                        logger.warn("No clusters available to select.")
                        return

                    first_id = min(cluster_ids)
                    logger.info(f"First cluster ID to select: {first_id}")
                    controller.supervisor.select(first_id)

            @controller.supervisor.actions.add(shortcut='ctrl+shift+l')
            def select_last_cluster():
                """Selects the cluster with the maximum cluster ID."""
                logger.info("Selecting the last (maximum) cluster ID.")

                @controller.supervisor.cluster_view.get_ids
                def highlight_last(cluster_ids):
                    if not cluster_ids:
                        logger.warn("No clusters available to select.")
                        return

                    last_id = max(cluster_ids)
                    logger.info(f"Last cluster ID to select: {last_id}")
                    controller.supervisor.select(last_id)
