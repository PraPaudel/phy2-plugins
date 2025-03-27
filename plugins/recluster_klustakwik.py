"""
Reclustering Tool for Phy

This script identifies units with high firing rates and automatically reclusters them
using KlustaKwik, allowing for better separation of potential merged units.

Run this script in your Kilosort output folder (the same folder where you would run 'phy template-gui').

example usage:
cd /path/to/your/Kilosort/output
python C:/Users/praveen/.phy/plugins/Recluster_Klustakwik.py
"""

import os
import sys
import numpy as np
import logging
import platform
import pandas as pd
import argparse
import traceback
from pathlib import Path
from subprocess import Popen

from phy.utils.config import phy_config_dir
from phylib.io.model import TemplateModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])
logger = logging.getLogger('recluster')

def write_fet(fet, filepath):
    with open(filepath, 'w') as fd:
        fd.write('%i\n' % fet.shape[1])
        for x in range(0, fet.shape[0]):
            fet[x, :].tofile(fd, sep="\t", format="%i")
            fd.write("\n")

def read_clusters(filename_clu):
    clusters = load_text(filename_clu, np.int64)
    return clusters[1:]  # Remove the first line (cluster count)

def load_text(filepath, dtype, skiprows=0, delimiter=' '):
    if not filepath:
        raise IOError("The filepath is empty.")
    with open(filepath, 'r') as f:
        for _ in range(skiprows):
            f.readline()
        x = pd.read_csv(f, header=None, sep=delimiter).values.astype(dtype).squeeze()
    return x

def check_params(params_path):
    if os.path.isdir(params_path):
        params_file = os.path.join(params_path, 'params.py')
    else:
        params_file = params_path
        params_path = os.path.dirname(params_file)
    
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"params.py not found at {params_file}")
    
    sample_rate = 20000.0  # Default
    with open(params_file, 'r') as f:
        params_content = f.read()
        
    for line in params_content.split('\n'):
        if 'sample_rate' in line and '=' in line:
            try:
                sample_rate = float(line.split('=')[1].strip().rstrip('.'))
            except ValueError:
                pass
    
    return params_path, sample_rate

def recluster_high_firing_rate(session_path, firing_rate_thresh=2.0, isi_violation_thresh=0.01, 
                              isi_window_ms=2.0, sample_rate=None, no_prompt=False):
    """
    Automatically reclusters units with high firing rates or ISI violations.
    
    This function:
    1. Loads a spike-sorted dataset
    2. Identifies units with firing rates above the threshold OR ISI violations above threshold
    3. For each problematic unit, extracts features and runs KlustaKwik to subdivide it
    4. Updates spike_clusters.npy with the new cluster assignments
    
    Parameters:
    -----------
    session_path : str
        Path to the Kilosort/Phy output directory
    firing_rate_thresh : float
        Firing rate threshold in Hz (default: 3.0)
    isi_violation_thresh : float
        ISI violation threshold as proportion (default: 0.01 = 1%)
    isi_window_ms : float
        ISI violation window in milliseconds (default: 2.0)
    sample_rate : float or None
        Sampling rate in Hz. If None, read from params.py
        
    Returns:
    --------
    bool
        True if successful, False if failed
    """
    
    print(f"\n=== PROCESSING SESSION: {session_path} ===")
    print(f"Firing rate threshold: {firing_rate_thresh} Hz")
    print(f"ISI violation threshold: {isi_violation_thresh*100:.2f}% at {isi_window_ms} ms")
    
    try:
        session_path, params_sample_rate = check_params(session_path)
        
        if sample_rate is None:
            sample_rate = params_sample_rate
            print(f"Using sample rate from params.py: {sample_rate} Hz")
        else:
            print(f"Using specified sample rate: {sample_rate} Hz")
            
            if abs(params_sample_rate - sample_rate) > 0.1 and not no_prompt:
                print(f"WARNING: Sample rate in params.py ({params_sample_rate} Hz) "
                      f"differs from specified rate ({sample_rate} Hz)")
                response = input("Use sample rate from params.py instead? [Y/n]: ")
                if response.lower() != 'n':
                    sample_rate = params_sample_rate
                    print(f"Using sample rate from params.py: {sample_rate} Hz")
    except Exception as e:
        print(f"ERROR checking params: {e}")
        return False
    
    try:
        print("Loading spike data...")
        model = TemplateModel(dir_path=session_path)
        
        spike_clusters = model.spike_clusters
        spike_times = model.spike_times
        
        cluster_ids = np.unique(spike_clusters)
        print(f"Found {len(cluster_ids)} unique clusters")
        
        clusters_to_recluster = []
        print("\nAnalyzing clusters...")
        
        isi_window_samples = int((isi_window_ms / 1000.0) * sample_rate)
        
        for cluster_id in cluster_ids:
            cluster_mask = spike_clusters == cluster_id
            cluster_spikes = np.where(cluster_mask)[0]
            
            if len(cluster_spikes) > 1:
                cluster_times = spike_times[cluster_spikes]
                
                duration_seconds = (cluster_times[-1] - cluster_times[0]) / sample_rate
                firing_rate = 0
                if duration_seconds > 0:
                    firing_rate = len(cluster_times) / duration_seconds
                
                isi = np.diff(cluster_times)
                isi_violations = np.sum(isi < isi_window_samples)
                isi_violation_rate = isi_violations / (len(isi) + 1e-10)
                
                needs_reclustering = False
                reason = []
                
                if firing_rate > firing_rate_thresh:
                    needs_reclustering = True
                    reason.append(f"high firing rate ({firing_rate:.2f} Hz)")
                
                if isi_violation_rate > isi_violation_thresh:
                    needs_reclustering = True
                    reason.append(f"ISI violations ({isi_violation_rate*100:.2f}%)")
                
                if needs_reclustering:
                    print(f"Cluster {cluster_id}: {', '.join(reason)}")
                    clusters_to_recluster.append(cluster_id)
        
        if not clusters_to_recluster:
            print(f"No clusters found that need reclustering")
            return True
        
        print(f"\nFound {len(clusters_to_recluster)} clusters to recluster: {clusters_to_recluster}")
        
        for cluster_id in clusters_to_recluster:
            print(f"\nReclustering cluster {cluster_id}...")
            
            spike_ids = np.where(spike_clusters == cluster_id)[0]
            
            features = model.features
            data3 = features[spike_ids]
            fet2 = np.reshape(data3, (data3.shape[0], data3.shape[1] * data3.shape[2]))
            
            dtype = np.int64
            factor = 2.**60
            if np.abs(fet2).max() > 0:
                factor = factor / np.abs(fet2).max()
            fet2 = (fet2 * factor).astype(dtype)
            
            name = 'tempClustering'
            shank = 3
            mainfetfile = os.path.join(session_path, name + '.fet.' + str(shank))
            write_fet(fet2, mainfetfile)
            
            if platform.system() == 'Windows':
                program = os.path.join(phy_config_dir(), 'klustakwik.exe')
            else:
                program = '~/klustakwik/KlustaKwik'
            
            if not os.path.exists(program):
                print(f"ERROR: KlustaKwik not found at {program}")
                return False
            
            cmd = [program, name, str(shank)]
            cmd += ["-UseDistributional", '0', "-MaxPossibleClusters", '20', "-MinClusters", '20']
            
            print("Running KlustaKwik...")
            p = Popen(cmd, cwd=session_path)
            p.wait()
            
            clu_file = os.path.join(session_path, name + '.clu.' + str(shank))
            if not os.path.exists(clu_file):
                print(f"ERROR: KlustaKwik output not found: {clu_file}")
                continue
                
            spike_clusters_new = read_clusters(clu_file)
            
            unique_new_clusters = np.unique(spike_clusters_new)
            max_cluster = np.max(spike_clusters)
            
            remapped_clusters = {}
            next_id = max_cluster + 1
            
            if 0 in unique_new_clusters and len(unique_new_clusters) > 1:
                remapped_clusters[0] = cluster_id
                for c in unique_new_clusters:
                    if c != 0:
                        remapped_clusters[c] = next_id
                        next_id += 1
            else:
                remapped_clusters = {c: max_cluster + 1 + i for i, c in enumerate(unique_new_clusters)}
            
            print(f"Split into {len(unique_new_clusters)} subclusters")
            
            for i, spike_id in enumerate(spike_ids):
                if i < len(spike_clusters_new):
                    new_cluster = spike_clusters_new[i]
                    spike_clusters[spike_id] = remapped_clusters[new_cluster]
        
        spike_clusters_path = os.path.join(session_path, 'spike_clusters.npy')
        np.save(spike_clusters_path, spike_clusters)
        print(f"Saved new clustering to spike_clusters.npy")
        
        print(f"\nReclustering complete!")
        return True
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Recluster high firing rate or ISI violation units in Phy/Kilosort data',
        epilog='Run in Kilosort output directory containing params.py'
    )
    parser.add_argument('--sessions', nargs='+', default=['.'], 
                        help='Paths to session directories (default: current directory)')
    parser.add_argument('--threshold', type=float, default=2.0, 
                        help='Firing rate threshold in Hz (default: 2.0)')
    parser.add_argument('--isi-threshold', type=float, default=0.01, 
                        help='ISI violation threshold as proportion (default: 0.01 = 1%)')
    parser.add_argument('--isi-window', type=float, default=2.0, 
                        help='ISI violation window in milliseconds (default: 2.0)')
    parser.add_argument('--sample-rate', type=float, 
                        help='Sampling rate in Hz (default: from params.py)')
    parser.add_argument('--no-prompt', action='store_true', 
                        help='Run without prompting for user input')
    
    args = parser.parse_args()
    
    print("\n====== PHY Cluster Reclustering Tool ======")
    
    if args.sample_rate is None:
        try:
            first_session = args.sessions[0]
            _, default_sample_rate = check_params(first_session)
            
            if not args.no_prompt:
                prompt_message = f"Enter sample rate in Hz (default: {default_sample_rate}): "
                user_input = input(prompt_message).strip()
                if user_input:
                    try:
                        args.sample_rate = float(user_input)
                    except ValueError:
                        args.sample_rate = default_sample_rate
                else:
                    args.sample_rate = default_sample_rate
            else:
                args.sample_rate = default_sample_rate
                print(f"Using sample rate from params.py: {default_sample_rate} Hz")
        except Exception:
            default_sample_rate = 20000.0
            if not args.no_prompt:
                user_input = input("Enter sample rate in Hz (default: 20000.0): ").strip()
                if user_input:
                    try:
                        args.sample_rate = float(user_input)
                    except ValueError:
                        args.sample_rate = default_sample_rate
                else:
                    args.sample_rate = default_sample_rate
            else:
                args.sample_rate = default_sample_rate
    
    print(f"\nProcessing {len(args.sessions)} session(s)")
    print(f"Firing rate threshold: {args.threshold} Hz")
    print(f"ISI violation threshold: {args.isi_threshold*100:.2f}% at {args.isi_window} ms")
    
    for session_path in args.sessions:
        success = recluster_high_firing_rate(
            session_path, 
            args.threshold, 
            args.isi_threshold,
            args.isi_window,
            args.sample_rate,
            args.no_prompt
        )