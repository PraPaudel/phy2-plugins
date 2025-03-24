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
            def select_first_good_cluster():
                """Selects the good cluster with the minimum cluster ID."""
                logger.info("Selecting the first (minimum) good cluster ID.")

                @controller.supervisor.cluster_view.get_ids
                def highlight_first_good(cluster_ids):
                    if not cluster_ids:
                        logger.warn("No clusters available to select.")
                        return

                    # Filter good clusters
                    good_clusters = []
                    for cl in cluster_ids:
                        group_label = controller.supervisor.cluster_meta.get('group', cl)
                        if group_label == 'good':
                            good_clusters.append(cl)
                    
                    if not good_clusters:
                        logger.warn("No good clusters available to select.")
                        return
                        
                    first_good_id = min(good_clusters)
                    logger.info(f"First good cluster ID to select: {first_good_id}")
                    controller.supervisor.select(first_good_id)

            @controller.supervisor.actions.add(shortcut='ctrl+shift+l')
            def select_last_good_cluster():
                """Selects the good cluster with the maximum cluster ID."""
                logger.info("Selecting the last (maximum) good cluster ID.")

                @controller.supervisor.cluster_view.get_ids
                def highlight_last_good(cluster_ids):
                    if not cluster_ids:
                        logger.warn("No clusters available to select.")
                        return

                    # Filter good clusters
                    good_clusters = []
                    for cl in cluster_ids:
                        group_label = controller.supervisor.cluster_meta.get('group', cl)
                        if group_label == 'good':
                            good_clusters.append(cl)
                    
                    if not good_clusters:
                        logger.warn("No good clusters available to select.")
                        return
                        
                    last_good_id = max(good_clusters)
                    logger.info(f"Last good cluster ID to select: {last_good_id}")
                    controller.supervisor.select(last_good_id)

                    
            @controller.supervisor.actions.add(shortcut='ctrl+shift+v')
            def show_good_clusters_info():
                """Shows information about 'good' clusters in a popup window."""
                logger.info("Displaying information about 'good' clusters.")
                
                # Get all cluster IDs
                cluster_ids = controller.supervisor.clustering.cluster_ids
                
                # Filter good clusters
                good_clusters = []
                for cl in cluster_ids:
                    # Check the 'group' meta field for each cluster
                    group_label = controller.supervisor.cluster_meta.get('group', cl)
                    if group_label == 'good':
                        good_clusters.append(cl)
                
                # Create message with the information
                len_good = len(good_clusters)
                message = f"Number of 'good' clusters: {len_good}\nGood cluster IDs: {good_clusters}"
                
                # Display in a popup window using Qt
                from PyQt5.QtWidgets import QMessageBox
                box = QMessageBox()
                box.setWindowTitle("Good Clusters Info")
                box.setText(message)
                box.exec_()