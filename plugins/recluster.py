from phy import IPlugin, connect

import logging
import os
import numpy as np

import platform

from pathlib import Path
from subprocess import Popen

from phy.utils.tempdir import TemporaryDirectory
from scipy.cluster.vq import kmeans2, whiten

#logger = logging.getLogger(__name__)
logger = logging.getLogger('phy')

##try:
##    from klusta.launch import cluster2
##except ImportError:  # pragma: no cover
##    logger.warn("Package klusta not installed: the KwikGUI will not work.")
# Not used
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    logger.warn("Package pandas not installed.")
try:
    from phy.utils.config import phy_config_dir
except ImportError:  # pragma: no cover
    logger.warn("phy_config_dir not available.")

class Recluster(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        #@controller.supervisor.connect
        #def on_create_cluster_views():
        def on_gui_ready(sender,gui):
            @controller.supervisor.actions.add(shortcut='alt+k')
            def Recluster_KlustaKwik_PCAs():
                def write_fet(fet, filepath):
                    with open(filepath, 'w') as fd:
                        #header line: number of features
                        fd.write('%i\n' % fet.shape[1])
                        #next lines: one feature vector per line
                        for x in range(0,fet.shape[0]):
                            fet[x,:].tofile(fd, sep="\t", format="%i")
                            fd.write ("\n")
                        #np.savetxt(fd, fet[0], fmt="%i", delimiter=' ')

                def read_clusters(filename_clu):
                    clusters = load_text(filename_clu, np.int64)
                    return process_clusters(clusters)
                def process_clusters(clusters):
                    return clusters[1:]
                def load_text(filepath, dtype, skiprows=0, delimiter=' '):
                    if not filepath:
                        raise IOError("The filepath is empty.")
                    with open(filepath, 'r') as f:
                        for _ in range(skiprows):
                            f.readline()
                        x = pd.read_csv(f, header=None,
                            sep=delimiter).values.astype(dtype).squeeze()
                    return x
                
                """Relaunch KlustaKwik on selected clusters."""
                # Selected clusters.
                cluster_ids = controller.supervisor.selected
                #spike_ids = controller.selector.select_spikes(cluster_ids)
                bunchs = controller._amplitude_getter(cluster_ids, name='template', load_all=True)
                spike_ids = bunchs[0].spike_ids
                logger.info("Running KlustaKwik on %d spikes.", len(spike_ids))
                # s = controller.supervisor.clustering.spikes_in_clusters(cluster_ids)
                data3 = controller.model._load_features().data[spike_ids]
                fet2 = np.reshape(data3,(data3.shape[0],data3.shape[1]*data3.shape[2]))

                dtype = np.int64
                factor = 2.**60
                #dtype = np.int32
                #factor = 2.**31
                factor = factor/np.abs(fet2).max()
                fet2 = (fet2 * factor).astype(dtype)
                # logger.warn(str(fet2[0,:]))

                # Run KK2 in a temporary directory to avoid side effects.
                # n = 10
                # spike_times = controller.model.spike_times[spike_ids]*controller.model.sample_rate

                #spike_times = convert_dtype(spike_times, np.int32)
                # times = np.expand_dims(spike_times, axis =1)
                
                # fet = 1000*np.concatenate((fet2,times),axis = 1)
                fet = fet2

                name = 'tempClustering'
                shank = 3
                mainfetfile = os.path.join(name + '.fet.' + str(shank))
                write_fet(fet, mainfetfile)
                if platform.system() == 'Windows':
                    program = os.path.join(phy_config_dir(),'klustakwik.exe')
                else:
                    program = '~/klustakwik/KlustaKwik'
                cmd = [program, name, str(shank)]
                cmd +=["-UseDistributional",'0',"-MaxPossibleClusters",'20',"-MinClusters",'20'] #,"-MinClusters",'2',"-MaxClusters",'12'   ,"-MaxClusters",'12',"-MaxClusters",'12'

                # Run KlustaKwik
                p = Popen(cmd)
                p.wait()
                # Read back the clusters
                spike_clusters = read_clusters(name + '.clu.' + str(shank))
                controller.supervisor.actions.split(spike_ids, spike_clusters)
                logger.warn("Reclustering complete!")


            @controller.supervisor.actions.add(shortcut='alt+q', prompt=True, prompt_default=lambda: 2)
            def K_means_clustering(kmeanclusters):
                """Select number of clusters.

                Example: `2`

                """
                logger.warn("Running K-means clustering")

                cluster_ids = controller.supervisor.selected
                #spike_ids = controller.selector.select_spikes(cluster_ids)
                bunchs = controller._amplitude_getter(cluster_ids, name='template', load_all=True)
                spike_ids = bunchs[0].spike_ids
                s = controller.supervisor.clustering.spikes_in_clusters(cluster_ids)
                data = controller.model._load_features()
                data3 = data.data[spike_ids]
                data2 = np.reshape(data3,(data3.shape[0],data3.shape[1]*data3.shape[2]))
                whitened = whiten(data2)
                clusters_out,label = kmeans2(whitened,kmeanclusters)
                controller.supervisor.actions.split(s,label)
                logger.warn("K means clustering complete")
