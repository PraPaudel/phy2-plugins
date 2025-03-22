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
from pathlib import Path
from subprocess import Popen

from phy.utils.config import phy_config_dir
from phylib.io.model import TemplateModel


logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])
logger = logging.getLogger('recluster')

def write_fet(fet, filepath):
    """Write features to KlustaKwik format"""
    with open(filepath, 'w') as fd:
        fd.write('%i\n' % fet.shape[1])
        for x in range(0, fet.shape[0]):
            fet[x, :].tofile(fd, sep="\t", format="%i")
            fd.write("\n")

def read_clusters(filename_clu):
    """Read clusters from KlustaKwik output"""
    clusters = load_text(filename_clu, np.int64)
    return process_clusters(clusters)

def process_clusters(clusters):
    """Process clusters by removing the first line (cluster count)"""
    return clusters[1:]

def load_text(filepath, dtype, skiprows=0, delimiter=' '):
    """Load text data from file"""
    if not filepath:
        raise IOError("The filepath is empty.")
    with open(filepath, 'r') as f:
        for _ in range(skiprows):
            f.readline()
        x = pd.read_csv(f, header=None, sep=delimiter).values.astype(dtype).squeeze()
    return x

def check_params(params_path):
    """
    Check for params.py and extract sample rate
    
    Parameters:
    -----------
    params_path : str
        Path to folder containing params.py or direct path to params.py
        
    Returns:
    --------
    tuple
        (directory_path, sample_rate)
    """
    # If params_path is a directory, look for params.py inside it
    if os.path.isdir(params_path):
        params_file = os.path.join(params_path, 'params.py')
    else:
        # If it's pointing to params.py directly
        params_file = params_path
        params_path = os.path.dirname(params_file)
    
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"params.py not found at {params_file}")
    
    # Extract sample rate if present in params.py
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

def recluster_high_firing_rate(session_path, firing_rate_thresh=3.0, sample_rate=None):
    """
    Automatically reclusters units with high firing rates.
    
    This function:
    1. Loads a spike-sorted dataset
    2. Identifies units with firing rates above the threshold
    3. For each high firing unit, extracts features and runs KlustaKwik to subdivide it
    4. Updates spike_clusters.npy with the new cluster assignments
    
    Parameters:
    -----------
    session_path : str
        Path to the Kilosort/Phy output directory
    firing_rate_thresh : float
        Firing rate threshold in Hz (default: 3.0)
    sample_rate : float or None
        Sampling rate in Hz. If None, read from params.py
        
    Returns:
    --------
    bool
        True if successful, False if failed
    """
    print(f"\n=== PROCESSING SESSION: {session_path} ===")
    print(f"Firing rate threshold: {firing_rate_thresh} Hz")
    
    # Check params.py and get sample rate if needed
    try:
        session_path, params_sample_rate = check_params(session_path)
        
        # Use provided sample rate or fall back to params.py value
        if sample_rate is None:
            sample_rate = params_sample_rate
            print(f"Using sample rate from params.py: {sample_rate} Hz")
        else:
            print(f"Using specified sample rate: {sample_rate} Hz")
            
            # Confirm if different from params.py
            if abs(params_sample_rate - sample_rate) > 0.1:
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
        # Load the model
        print("Loading spike data...")
        model = TemplateModel(dir_path=session_path)
        
        # Load spike clusters and times
        spike_clusters = model.spike_clusters
        spike_times = model.spike_times
        
        # Get all clusters
        cluster_ids = np.unique(spike_clusters)
        print(f"Found {len(cluster_ids)} unique clusters")
        
        # Calculate firing rate for each cluster and identify high firing rate ones
        high_fr_clusters = []
        print("\nAnalyzing firing rates...")
        
        for cluster_id in cluster_ids:
            # Get spikes in this cluster
            cluster_mask = spike_clusters == cluster_id
            cluster_spikes = np.where(cluster_mask)[0]
            
            if len(cluster_spikes) > 1:
                # Calculate firing rate (spikes/sec)
                cluster_times = spike_times[cluster_spikes]
                # Convert from sample units to seconds
                duration_samples = cluster_times[-1] - cluster_times[0]
                duration_seconds = duration_samples / sample_rate
                if duration_seconds > 0:
                    firing_rate = len(cluster_times) / duration_seconds
                    # Only print high firing rate clusters to reduce output
                    if firing_rate > firing_rate_thresh:
                        print(f"Cluster {cluster_id}: {firing_rate:.2f} Hz")
                        high_fr_clusters.append(cluster_id)
        
        if not high_fr_clusters:
            print(f"No clusters found with firing rate > {firing_rate_thresh} Hz")
            return True  # Still consider this a success, just nothing to do
        
        print(f"\nFound {len(high_fr_clusters)} clusters with firing rate > {firing_rate_thresh} Hz: {high_fr_clusters}")
        
        # For each high firing rate cluster, perform reclustering
        for cluster_id in high_fr_clusters:
            print(f"\nReclustering cluster {cluster_id}...")
            
            # Get spike indices for this cluster
            spike_ids = np.where(spike_clusters == cluster_id)[0]
            
            # Extract features
            features = model.features
            data3 = features[spike_ids]
            fet2 = np.reshape(data3, (data3.shape[0], data3.shape[1] * data3.shape[2]))
            
            # Rescale for int64
            dtype = np.int64
            factor = 2.**60
            if np.abs(fet2).max() > 0:  # Avoid division by zero
                factor = factor / np.abs(fet2).max()
            fet2 = (fet2 * factor).astype(dtype)
            
            # Write features to file
            name = 'tempClustering'
            shank = 3
            mainfetfile = os.path.join(session_path, name + '.fet.' + str(shank))
            write_fet(fet2, mainfetfile)
            
            # Run KlustaKwik
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
            p = Popen(cmd, cwd=session_path)  # Run in session directory
            p.wait()
            
            # Read back the clusters
            clu_file = os.path.join(session_path, name + '.clu.' + str(shank))
            if not os.path.exists(clu_file):
                print(f"ERROR: KlustaKwik output not found: {clu_file}")
                continue
                
            spike_clusters_new = read_clusters(clu_file)
            
            # Create unique IDs for new clusters
            unique_new_clusters = np.unique(spike_clusters_new)
            max_cluster = np.max(spike_clusters)
            
            # Create mapping for new clusters
            remapped_clusters = {}
            next_id = max_cluster + 1
            
            # Special case for noise cluster (0)
            if 0 in unique_new_clusters and len(unique_new_clusters) > 1:
                remapped_clusters[0] = cluster_id  # Keep original ID for noise
                for c in unique_new_clusters:
                    if c != 0:
                        remapped_clusters[c] = next_id
                        next_id += 1
            else:
                # Simple offset for all clusters if no noise cluster
                remapped_clusters = {c: max_cluster + 1 + i for i, c in enumerate(unique_new_clusters)}
            
            print(f"Split into {len(unique_new_clusters)} subclusters")
            
            # Apply the new clustering
            for i, spike_id in enumerate(spike_ids):
                if i < len(spike_clusters_new):
                    new_cluster = spike_clusters_new[i]
                    spike_clusters[spike_id] = remapped_clusters[new_cluster]
        
        spike_clusters_path = os.path.join(session_path, 'spike_clusters.npy')
        # Save the new clustering
        np.save(spike_clusters_path, spike_clusters)
        print(f"Saved new clustering to spike_clusters.npy")
        
        print(f"\nHigh firing rate reclustering complete!")
        print(f"You can now run 'phy template-gui' to review the reclustered units")
        return True
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Automatically recluster high firing rate units in Phy/Kilosort data',
        epilog='Run this script in your Kilosort output directory containing params.py'
    )
    parser.add_argument('--sessions', nargs='+', help='List of paths to session directories (default: current directory)', default=['.'])
    parser.add_argument('--threshold', type=float, default=3.0, help='Firing rate threshold in Hz (default: 3.0)')
    parser.add_argument('--sample-rate', type=float, help='Sampling rate in Hz (default: read from params.py)')
    
    args = parser.parse_args()
    
    print("\n====== PHY High Firing Rate Reclustering Tool ======")
    print("This tool automatically reclusters high firing rate units")
    print("See https://github.com/cortex-lab/phy for more information about Phy")
    
    # If no sample rate provided via command line, prompt user
    if args.sample_rate is None:
        # Try to get default from first session's params.py
        try:
            first_session = args.sessions[0]
            _, default_sample_rate = check_params(first_session)
            prompt_message = f"Enter sample rate in Hz (default: {default_sample_rate}): "
        except Exception:
            default_sample_rate = 20000.0
            prompt_message = "Enter sample rate in Hz (default: 20000.0): "
        
        # Prompt user for sample rate
        user_input = input(prompt_message).strip()
        if user_input:
            try:
                args.sample_rate = float(user_input)
                print(f"Using user-provided sample rate: {args.sample_rate} Hz")
            except ValueError:
                print(f"Invalid sample rate '{user_input}', using default")
                args.sample_rate = default_sample_rate
        else:
            print(f"Using default sample rate: {default_sample_rate} Hz")
            args.sample_rate = default_sample_rate
    
    print(f"\nProcessing {len(args.sessions)} session(s)")
    print(f"Using threshold: {args.threshold} Hz")
    print(f"Using sample rate: {args.sample_rate} Hz")
    
    # Process each session
    for session_path in args.sessions:
        success = recluster_high_firing_rate(session_path, args.threshold, args.sample_rate)
        if success:
            print(f"Successfully processed {session_path}")
        else:
            print(f"Failed to process {session_path}")
    
    print("\nAll sessions processed.")