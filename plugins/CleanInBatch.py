"""Mahalanobis Distance Splitter Plugin for Phy."""

from phy import IPlugin, connect
import numpy as np
import logging
from scipy.cluster.vq import kmeans2, whiten
from scipy.spatial.distance import cdist

logger = logging.getLogger('phy')


class CleanInBatch(IPlugin):
    def attach_to_controller(self, controller):
        """Attach the Mahalanobis splitter action to the controller."""
        @connect
        def on_gui_ready(sender, gui):
            @controller.supervisor.actions.add(shortcut='shift+x', prompt=True, prompt_default=lambda: "9")
            def Batch_Mahalanobis(threshold_str):
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

            @controller.supervisor.actions.add(shortcut='shift+k', prompt=True, prompt_default=lambda: "2")
            def Batch_KMeans(kmeanclusters_str):
                """Run K-means clustering on all 'review' clusters in batch.
                
                Provide number of clusters (default: 2)
                """
                logger.info("Starting batch K-means clustering on review clusters...")
                
                # Parse number of clusters from input
                try:
                    n_clusters = int(kmeanclusters_str)
                    logger.info(f"Using {n_clusters} clusters for K-means")
                except ValueError:
                    n_clusters = 2
                    logger.info(f"Invalid input. Using default: {n_clusters} clusters")
                
                try:
                    # Get all cluster IDs and filter by 'review' group
                    all_cluster_ids = controller.supervisor.clustering.cluster_ids
                    review_clusters = []
                    
                    for cid in all_cluster_ids:
                        group_label = controller.supervisor.cluster_meta.get('group', cid)
                        if group_label == 'review':
                            review_clusters.append(cid)
                    
                    if not review_clusters:
                        logger.warn("No 'review' clusters found")
                        return
                    
                    logger.info(f"Processing {len(review_clusters)} review clusters with K-means (k={n_clusters})")
                    
                    # Process each review cluster
                    clusters_processed = 0
                    
                    for cid in review_clusters:
                        # Get spikes for this cluster
                        spike_ids = controller.supervisor.clustering.spikes_in_clusters([cid])
                        
                        if len(spike_ids) < n_clusters:
                            logger.info(f"Cluster {cid} has too few spikes ({len(spike_ids)}), skipping")
                            continue
                        
                        # Load features
                        data = controller.model._load_features().data[spike_ids]
                        data2 = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))
                        
                        # Whiten and cluster
                        whitened = whiten(data2)
                        clusters_out, label = kmeans2(whitened, n_clusters)
                        
                        # Split the cluster
                        controller.supervisor.actions.split(spike_ids, label)
                        logger.info(f"K-means split cluster {cid} into {n_clusters} groups")
                        clusters_processed += 1
                    
                    logger.info(f"Completed: K-means clustering on {clusters_processed} review clusters")
                
                except Exception as e:
                    logger.error(f"Error in batch K-means: {str(e)}")

            @controller.supervisor.actions.add(shortcut='shift+i')
            def Batch_ViolatedISI():
                """Analyze and split short ISI violations on all 'review' clusters in batch.
                
                Uses spike times, amplitudes, and waveforms to detect suspicious spikes.
                """
                logger.info("Starting batch short ISI analysis on review clusters...")
                
                def analyze_suspicious_spikes(spike_times, spike_amps, waveforms, isi_threshold=0.0015):
                    """Analyze spikes with multiple metrics"""
                    n_spikes = len(spike_times)
                    suspicious = np.zeros(n_spikes, dtype=bool)
                    
                    isi_prev = np.diff(spike_times, prepend=spike_times[0] - 1)
                    isi_next = np.diff(spike_times, append=spike_times[-1] + 1)
                    
                    for i in range(n_spikes):
                        if isi_prev[i] < isi_threshold or isi_next[i] < isi_threshold:
                            # Check amplitude changes
                            amp_window = slice(max(0, i - 1), min(n_spikes, i + 2))
                            amp_variation = np.std(spike_amps[amp_window])
                            
                            # Check waveform changes
                            wave_window = slice(max(0, i - 1), min(n_spikes, i + 2))
                            waves = waveforms[wave_window]
                            wave_distances = cdist(waves, waves, metric='correlation')
                            wave_variation = np.mean(wave_distances)
                            
                            if (amp_variation > np.std(spike_amps) * 1.5 or wave_variation > 0.1):
                                suspicious[i] = True
                    
                    return suspicious
                
                try:
                    # Get all cluster IDs and filter by 'review' group
                    all_cluster_ids = controller.supervisor.clustering.cluster_ids
                    review_clusters = []
                    
                    for cid in all_cluster_ids:
                        group_label = controller.supervisor.cluster_meta.get('group', cid)
                        if group_label == 'review':
                            review_clusters.append(cid)
                    
                    if not review_clusters:
                        logger.warn("No 'review' clusters found")
                        return
                    
                    logger.info(f"Processing {len(review_clusters)} review clusters for short ISI analysis")
                    
                    # Process each review cluster
                    clusters_processed = 0
                    total_suspicious = 0
                    
                    for cid in review_clusters:
                        # Get spikes for this cluster
                        spike_ids = controller.supervisor.clustering.spikes_in_clusters([cid])
                        
                        if len(spike_ids) < 10:
                            continue
                        
                        # Get spike times
                        spike_times = controller.model.spike_times[spike_ids]
                        
                        # Get amplitudes
                        bunchs = controller._amplitude_getter([cid], name='template', load_all=True)
                        spike_amps = bunchs[0].amplitudes
                        
                        # Get waveforms
                        data = controller.model._load_features().data[spike_ids]
                        waveforms = np.reshape(data, (data.shape[0], -1))
                        
                        # Analyze
                        suspicious = analyze_suspicious_spikes(spike_times, spike_amps, waveforms)
                        n_suspicious = np.sum(suspicious)
                        
                        # Split if found enough suspicious spikes
                        if n_suspicious >= 10 and n_suspicious <= len(spike_ids) * 0.5:
                            labels = np.ones(len(spike_ids), dtype=int)
                            labels[suspicious] = 2
                            controller.supervisor.actions.split(spike_ids, labels)
                            logger.info(f"Cluster {cid}: split {n_suspicious} suspicious spikes ({n_suspicious/len(spike_ids)*100:.1f}%)")
                            clusters_processed += 1
                            total_suspicious += n_suspicious
                    
                    logger.info(f"Completed: Analyzed ISI on {len(review_clusters)} clusters, split {clusters_processed} clusters ({total_suspicious} suspicious spikes)")
                
                except Exception as e:
                    logger.error(f"Error in batch short ISI: {str(e)}")