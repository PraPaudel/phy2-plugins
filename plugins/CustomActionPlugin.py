"""
The following actions are implemented:
- Ctrl+C: Selects the first unsorted cluster.
- Ctrl+V: Moves the selected cluster to the end.
- Ctrl+B: Moves similar clusters to the end.
- Ctrl+Shift+F: Selects the first (minimum) good cluster ID.
- Ctrl+Shift+L: Selects the last (maximum) good cluster ID.
- Ctrl+Shift+V: Displays detailed information about 'good' clusters, including firing rate analysis.
"""

from phy import IPlugin, connect
import numpy as np
import logging

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QTextEdit, QHBoxLayout,
                            QWidget, QSplitter, QPushButton, QTabWidget, QComboBox,
                            QSpinBox, QFormLayout, QGroupBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

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
                """Shows detailed information about 'good' clusters with firing rate analysis."""
                logger.info("Displaying detailed information about 'good' clusters.")

                cluster_ids = controller.supervisor.clustering.cluster_ids
                
                # Filter good clusters
                good_clusters = []
                for cl in cluster_ids:
                    group_label = controller.supervisor.cluster_meta.get('group', cl)
                    if group_label == 'good':
                        good_clusters.append(cl)
                
                if not good_clusters:
                    from PyQt5.QtWidgets import QMessageBox
                    box = QMessageBox()
                    box.setWindowTitle("Good Clusters Info")
                    box.setText("No 'good' clusters found.")
                    box.exec_()
                    return
                
                # Get recording duration and spike times
                total_duration = controller.model.duration  # Recording duration in seconds
                spike_times = controller.model.spike_times
                
                # Get firing rates for good clusters
                firing_rates = {}
                spike_times_per_cluster = {}
                high_firing = []
                low_firing = []
                firing_threshold = 2.0  # Hz
                
                for cl in good_clusters:
                    # Get spikes for this cluster
                    spike_ids = controller.supervisor.clustering.spikes_per_cluster[cl]
                    n_spikes = len(spike_ids)
                    
                    # Get spike times for temporal analysis
                    if len(spike_ids) > 0:
                        spike_times_per_cluster[cl] = spike_times[spike_ids]
                    else:
                        spike_times_per_cluster[cl] = np.array([])
                    
                    # Calculate firing rate (Hz)
                    rate = n_spikes / total_duration
                    firing_rates[cl] = rate
                    
                    # Classify as high or low firing
                    if rate >= firing_threshold:
                        high_firing.append(cl)
                    else:
                        low_firing.append(cl)
                
                # Sort clusters by ID for consistent ordering
                sorted_clusters = sorted(good_clusters)
                
                # Create dialog with plots
                dialog = QDialog()
                dialog.setWindowTitle("Good Clusters Analysis")
                dialog.resize(1200, 700)
                
                # Enable maximize button
                dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowMaximizeButtonHint)
                
                main_layout = QVBoxLayout()
                
                # Add simple font size control
                font_control = QHBoxLayout()
                font_label = QLabel("GUI Font Size:")
                font_control.addWidget(font_label)
                
                font_spinner = QSpinBox()
                font_spinner.setRange(8, 16)
                font_spinner.setValue(10)
                font_control.addWidget(font_spinner)
                
                font_control.addStretch()
                main_layout.addLayout(font_control)
                
                # Add summary text at top
                summary = (f"Number of 'good' clusters: {len(good_clusters)}\n"
                        f"High firing (≥{firing_threshold} Hz): {len(high_firing)} clusters\n"
                        f"Low firing (<{firing_threshold} Hz): {len(low_firing)} clusters\n")
                summary_label = QLabel(summary)
                main_layout.addWidget(summary_label)
                
                # Create tab widget for different visualizations
                tab_widget = QTabWidget()
                main_layout.addWidget(tab_widget)
                
                # ----- OVERVIEW TAB -----
                # Tab 1: Overview (Text info + Bar chart + Heatmap)
                overview_tab = QWidget()
                overview_layout = QVBoxLayout(overview_tab)
                
                # Create HORIZONTAL splitter for side-by-side content in overview
                overview_splitter = QSplitter(Qt.Horizontal)
                overview_layout.addWidget(overview_splitter)
                
                # Left side: Text info
                left_widget = QWidget()
                left_layout = QVBoxLayout(left_widget)
                
                # Create detailed table of clusters and their firing rates
                detailed_info = "Cluster ID | Firing Rate (Hz)\n"  # Removed Category column
                detailed_info += "-" * 30 + "\n"  # Shortened separator
                
                for cl in sorted_clusters:
                    rate = firing_rates[cl]
                    detailed_info += f"{cl} | {rate:.2f} Hz\n"  # Simplified format
                
                text_box = QTextEdit()
                text_box.setPlainText(detailed_info)
                text_box.setReadOnly(True)
                left_layout.addWidget(text_box)
                
                # Add sorting controls to left panel
                sort_group = QGroupBox("Sorting Options")
                sort_layout = QHBoxLayout()
                sort_options = QComboBox()
                sort_options.addItem("Sort by Cluster ID (Ascending)", "id_asc")
                sort_options.addItem("Sort by Cluster ID (Descending)", "id_desc")
                sort_options.addItem("Sort by Firing Rate (Ascending)", "rate_asc")
                sort_options.addItem("Sort by Firing Rate (Descending)", "rate_desc")
                sort_layout.addWidget(sort_options)
                sort_group.setLayout(sort_layout)
                left_layout.addWidget(sort_group)
                
                # Colorbar range control (will affect heatmap)
                colorbar_group = QGroupBox("Heatmap Color Scale")
                colorbar_layout = QFormLayout()
                colorbar_range = QComboBox()
                colorbar_range.addItem("Auto (95th percentile)", "auto")
                colorbar_range.addItem("Max Value", "max")
                colorbar_range.addItem("Custom Range", "custom")
                colorbar_layout.addRow("Range:", colorbar_range)
                
                max_scale_spinner = QSpinBox()
                max_scale_spinner.setRange(1, 100)
                max_scale_spinner.setValue(10)  # Default value (Hz)
                max_scale_spinner.setEnabled(False)
                colorbar_layout.addRow("Max Value (Hz):", max_scale_spinner)
                colorbar_group.setLayout(colorbar_layout)
                left_layout.addWidget(colorbar_group)
                
                left_widget.setLayout(left_layout)
                overview_splitter.addWidget(left_widget)
                
                # Right side: Figures (Bar chart on top, Heatmap on bottom)
                right_widget = QWidget()
                right_layout = QVBoxLayout(right_widget)
                
                # Create bar plot
                figure1 = plt.figure(figsize=(7, 3))
                canvas1 = FigureCanvas(figure1)
                toolbar1 = NavigationToolbar(canvas1, dialog)  # Add zoom functionality
                right_layout.addWidget(toolbar1)
                right_layout.addWidget(canvas1)
                
                # Heatmap plot
                figure2 = plt.figure(figsize=(7, 4))
                canvas2 = FigureCanvas(figure2)
                toolbar2 = NavigationToolbar(canvas2, dialog)  # Add zoom functionality
                right_layout.addWidget(toolbar2)
                right_layout.addWidget(canvas2)
                
                right_widget.setLayout(right_layout)
                overview_splitter.addWidget(right_widget)
                
                # Complete overview tab layout
                overview_layout.addWidget(overview_splitter)
                overview_tab.setLayout(overview_layout)
                
                # Add overview tab to tab widget
                tab_widget.addTab(overview_tab, "Overview")
                
                # ----- INDIVIDUAL NEURONS TAB -----
                # Tab 2: Individual Neuron View
                individual_tab = QWidget()
                individual_layout = QVBoxLayout(individual_tab)
                
                # Navigation controls
                nav_widget = QWidget()
                nav_layout = QHBoxLayout(nav_widget)
                
                # Previous button
                prev_btn = QPushButton("Previous")
                prev_btn.setMinimumWidth(100)  # Set consistent width
                nav_layout.addWidget(prev_btn)
                
                # Dropdown for direct cluster selection with consistent width
                cluster_selector = QComboBox()
                cluster_selector.setMinimumWidth(200)  # Make dropdown wider to match button size
                
                for cl in sorted_clusters:
                    cluster_selector.addItem(f"Cluster {cl} ({firing_rates[cl]:.1f} Hz)", cl)
                nav_layout.addWidget(cluster_selector)
                
                # Next button
                next_btn = QPushButton("Next")
                next_btn.setMinimumWidth(100)  # Set consistent width
                nav_layout.addWidget(next_btn)
                
                # Add "Mark for Review" button
                review_btn = QPushButton("Mark for Review")
                review_btn.setCheckable(True)  # Make it toggleable
                review_btn.setMinimumWidth(130)  # Set consistent width
                nav_layout.addWidget(review_btn)
                
                # Add navigation controls
                nav_widget.setLayout(nav_layout)
                individual_layout.addWidget(nav_widget)
                
                # Neuron info label
                neuron_label = QLabel("Current Neuron: ")
                individual_layout.addWidget(neuron_label)
                
                # Add review status indication below neuron label
                review_status_label = QLabel("")
                review_status_label.setStyleSheet("color: #FF6600; font-weight: bold;")
                individual_layout.addWidget(review_status_label)
                
                # Create figure for individual neuron visualization
                neuron_figure = plt.figure(figsize=(9, 7))
                neuron_canvas = FigureCanvas(neuron_figure)
                neuron_toolbar = NavigationToolbar(neuron_canvas, dialog)  # Add zoom functionality
                individual_layout.addWidget(neuron_toolbar)
                individual_layout.addWidget(neuron_canvas)
                
                # Complete individual tab layout
                individual_layout.addStretch()
                individual_tab.setLayout(individual_layout)
                
                # Add individual tab to tab widget
                tab_widget.addTab(individual_tab, "Individual Neurons")
                
                # Collect all UI elements that need font updating
                font_controlled_elements = [
                    # Labels
                    font_label, summary_label, neuron_label, review_status_label,
                    # Buttons
                    prev_btn, next_btn, review_btn,
                    # Comboboxes
                    sort_options, colorbar_range, cluster_selector,
                    # Text boxes
                    text_box,
                    # Spinners
                    font_spinner, max_scale_spinner,
                    # GroupBoxes
                    sort_group, colorbar_group
                ]
                
                # Get all form labels in colorbar layout
                for row in range(colorbar_layout.rowCount()):
                    label_item = colorbar_layout.itemAt(row, QFormLayout.LabelRole)
                    if label_item and label_item.widget():
                        font_controlled_elements.append(label_item.widget())
                
                # update all fonts
                def update_fonts(size):
                    font = QFont()
                    font.setPointSize(size)
                    
                    # Update all collected elements
                    for element in font_controlled_elements:
                        element.setFont(font)
                        
                    # Update tab widget fonts
                    for i in range(tab_widget.count()):
                        tab_text = tab_widget.tabText(i)
                        tab_widget.setTabText(i, tab_text)  # This forces the font to update
                        
                    # Special handling for TabWidget to update tab text
                    tab_font = QFont()
                    tab_font.setPointSize(size)
                    tab_widget.setFont(tab_font)
                
                # Connect font spinner
                font_spinner.valueChanged.connect(update_fonts)
                
                # Initialize needs_review metadata if it doesn't exist
                if 'needs_review' not in controller.model.metadata:
                    controller.model.metadata['needs_review'] = {}
                    logger.info("Initialized 'needs_review' metadata dictionary")
                
                # Safe method to check if a cluster needs review
                def needs_review(cluster_id):
                    try:
                        # Check if the cluster ID exists in the metadata dictionary
                        if cluster_id in controller.model.metadata['needs_review']:
                            return controller.model.metadata['needs_review'][cluster_id] == "yes"
                        return False
                    except Exception as e:
                        logger.error(f"Error checking review status: {e}")
                        return False
                
                # toggle review status
                def toggle_review_status():
                    try:
                        cluster_id = sorted_clusters[current_neuron_idx[0]]
                        current_status = needs_review(cluster_id)
                        
                        # Update the review status ("yes" = needs review, "" = doesn't need review)
                        new_status = "" if current_status else "yes"
                        
                        # Update the metadata dictionary
                        controller.model.metadata['needs_review'][cluster_id] = new_status
                        
                        # Save the changes to disk
                        controller.model.save_metadata('needs_review', controller.model.metadata['needs_review'])
                        
                        # Update button appearance and label
                        update_review_status_display(cluster_id)
                        
                        # Log the change
                        if new_status == "yes":
                            logger.info(f"Marked cluster {cluster_id} for review")
                        else:
                            logger.info(f"Unmarked cluster {cluster_id} from review")
                    except Exception as e:
                        logger.error(f"Error toggling review status: {e}")
                
                # update review status display
                def update_review_status_display(cluster_id):
                    try:
                        if needs_review(cluster_id):
                            review_btn.setChecked(True)
                            review_btn.setText("Unmark Review")
                            review_status_label.setText("⚠️ This cluster is marked for later review/curation")
                        else:
                            review_btn.setChecked(False)
                            review_btn.setText("Mark for Review")
                            review_status_label.setText("")
                    except Exception as e:
                        logger.error(f"Error updating review display: {e}")
                        
                # Connect review button
                review_btn.clicked.connect(toggle_review_status)
                # update the overview plots based on sorting options
                def update_overview_plots():
                    try:
                        # Get sorting option
                        sort_option = sort_options.currentData()
                        
                        # Create temporary cluster list for sorting
                        clusters_to_sort = [(cl, firing_rates[cl]) for cl in sorted_clusters]
                        
                        # Sort based on selected option
                        if sort_option == "id_asc":
                            sorted_items = sorted(clusters_to_sort, key=lambda x: x[0])
                        elif sort_option == "id_desc":
                            sorted_items = sorted(clusters_to_sort, key=lambda x: x[0], reverse=True)
                        elif sort_option == "rate_asc":
                            sorted_items = sorted(clusters_to_sort, key=lambda x: x[1])
                        else:  # "rate_desc"
                            sorted_items = sorted(clusters_to_sort, key=lambda x: x[1], reverse=True)
                            
                        # Extract sorted cluster IDs and rates
                        sorted_cl_ids = [item[0] for item in sorted_items]
                        sorted_cl_rates = [item[1] for item in sorted_items]
                            
                        # Clear previous plots
                        figure1.clear()
                        figure2.clear()
                        
                        # Plot firing rate distribution (bar chart)
                        ax1 = figure1.add_subplot(111)
                        
                        # Color bars based on firing rate
                        colors = ['#FF9999' if r >= firing_threshold else '#9999FF' for r in sorted_cl_rates]
                        
                        # Bar plot of firing rates
                        ax1.bar(range(len(sorted_cl_ids)), sorted_cl_rates, tick_label=sorted_cl_ids, color=colors)
                        ax1.axhline(y=firing_threshold, color='r', linestyle='--', label=f'Threshold ({firing_threshold} Hz)')
                        
                        # Add minor ticks for better visualization of low rates - more frequent minor ticks
                        ax1.xaxis.set_minor_locator(AutoMinorLocator(4))  # More minor ticks (4 minor ticks per major)
                        ax1.yaxis.set_minor_locator(AutoMinorLocator(5))  # More minor ticks on y-axis
                        
                        # Highlight x-axis to emphasize zero firing
                        ax1.axhline(y=0, color='k', linewidth=1.5)
                        
                        ax1.set_xlabel('Cluster ID')
                        ax1.set_ylabel('Firing Rate (Hz)')
                        ax1.set_title('Firing Rates of Good Clusters')
                        ax1.legend()
                        
                        if len(sorted_cl_ids) > 10:
                            ax1.set_xticklabels(sorted_cl_ids, rotation=45)
                            
                        # Create heatmap
                        ax2 = figure2.add_subplot(111)
                        
                        # Create time bins for heatmap analysis
                        bin_size_ms = 200  # 200 ms bins
                        n_bins = int(total_duration / (bin_size_ms/1000))
                        
                        # Limit bins to a reasonable number for display
                        if n_bins > 500:
                            n_bins = 500
                            bin_size_ms = (total_duration * 1000) / n_bins
                        
                        # Prepare data for heatmap based on sorted clusters
                        heatmap_data = np.zeros((len(sorted_cl_ids), n_bins))
                        
                        for i, cl in enumerate(sorted_cl_ids):
                            # Get spike times for this cluster
                            times = spike_times_per_cluster[cl]
                            
                            # Count spikes in each time bin
                            if len(times) > 0:
                                hist, _ = np.histogram(times, bins=n_bins, range=(0, total_duration))
                                # Convert to firing rate (Hz)
                                heatmap_data[i, :] = hist / (bin_size_ms/1000)
                        
                        # Get colorbar range setting
                        colorbar_setting = colorbar_range.currentData()
                        
                        if colorbar_setting == "auto":
                            # Find max firing rate for color scaling (exclude outliers)
                            vmax = np.percentile(heatmap_data[heatmap_data > 0], 95)
                        elif colorbar_setting == "max":
                            vmax = np.max(heatmap_data)
                        else:  # custom
                            vmax = max_scale_spinner.value()
                            
                        norm = Normalize(vmin=0, vmax=vmax)
                        
                        # Create heatmap with VIRIDIS colormap
                        im = ax2.imshow(heatmap_data, aspect='auto', interpolation='none', 
                                    cmap='viridis', norm=norm)
                        cbar = figure2.colorbar(im, ax=ax2)
                        cbar.set_label('Firing Rate (Hz)')
                        
                        ax2.set_xlabel(f'Time (bin size: {bin_size_ms:.1f} ms)')
                        ax2.set_ylabel('Cluster ID')
                        ax2.set_title('Temporal Firing Pattern of Good Clusters')
                        
                        # Set y-ticks to cluster IDs
                        ax2.set_yticks(range(len(sorted_cl_ids)))
                        ax2.set_yticklabels(sorted_cl_ids)
                        
                        # Add time markers on x-axis with minor ticks
                        x_ticks = np.linspace(0, n_bins-1, 5).astype(int)
                        x_labels = [f"{int((x/n_bins) * total_duration / 60)}m" for x in x_ticks]
                        ax2.set_xticks(x_ticks)
                        ax2.set_xticklabels(x_labels)
                        
                        # Add more frequent minor tick marks for better zooming
                        ax2.xaxis.set_minor_locator(AutoMinorLocator(4))  # 4 minor ticks between major ticks
                        ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
                        
                        # Redraw the figures
                        figure1.tight_layout()
                        canvas1.draw()
                        figure2.tight_layout()
                        canvas2.draw()
                    except Exception as e:
                        logger.error(f"Error updating overview plots: {str(e)}")
                        import traceback
                        traceback.print_exc()
                
                # Current neuron index
                current_neuron_idx = [0]  # Use list for mutable reference
                
                #  update individual neuron view
                def update_neuron_view():
                    try:
                        neuron_figure.clear()
                        
                        # Get current cluster ID
                        cluster_id = sorted_clusters[current_neuron_idx[0]]
                        neuron_label.setText(f"Current Neuron: Cluster {cluster_id} " +
                                        f"(Rate: {firing_rates[cluster_id]:.2f} Hz)")
                        
                        # Update dropdown to match current cluster
                        cluster_selector.setCurrentIndex(sorted_clusters.index(cluster_id))
                        
                        # Update review status display
                        update_review_status_display(cluster_id)
                        
                        # Create a 2-row subplot layout
                        ax_line = neuron_figure.add_subplot(211)  # Top plot - line plot with smoothing
                        ax_trend = neuron_figure.add_subplot(212)  # Bottom plot - trend line plot
                        
                        # Get spike times for this cluster
                        spike_times_array = spike_times_per_cluster[cluster_id]
                        
                        # Calculate max_rate for setting better y-axis limits
                        max_rate = max(firing_rates.values()) * 1.5  # Some headroom above max firing rate
                        
                        if len(spike_times_array) > 0:
                            # 1. Line plot with smoothing (100ms bins)
                            bin_size_line = 0.1  # 100ms bins
                            n_bins_line = int(total_duration / bin_size_line)
                            
                            # Calculate histogram
                            hist, bin_edges = np.histogram(
                                spike_times_array, bins=n_bins_line, range=(0, total_duration))
                            
                            # Convert to firing rate (Hz) and smooth
                            firing_rate = hist / bin_size_line
                            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                            
                            # Plot raw firing rate with reduced opacity
                            ax_line.plot(bin_centers, firing_rate, linewidth=0.5, color='gray', alpha=0.3)
                            
                            # Add smoothed line with GREEN color as requested
                            smooth_rate = gaussian_filter1d(firing_rate, sigma=2)
                            ax_line.plot(bin_centers, smooth_rate, linewidth=1.5, color='#00AA00')
                            
                            # Add threshold reference line
                            ax_line.axhline(y=firing_threshold, color='r', linestyle='--', 
                                        label=f"Threshold ({firing_threshold} Hz)")
                            
                            # Add y=0 line with emphasis
                            ax_line.axhline(y=0, color='k', linewidth=1.5)
                            
                            # Set up axes
                            ax_line.set_title(f"Firing Rate - Cluster {cluster_id} (bin: {int(bin_size_line*1000)}ms)")
                            ax_line.set_ylabel("Firing Rate (Hz)")
                            
                            # Add more minor ticks for better zooming
                            ax_line.yaxis.set_minor_locator(AutoMinorLocator(5))  # 5 minor ticks between major
                            ax_line.xaxis.set_minor_locator(AutoMinorLocator(4))  # 4 minor ticks between major
                            
                            # Set reasonable y-axis limits based on data
                            ax_line.set_ylim(0, min(max_rate, max(smooth_rate)*1.5))
                            
                            # Set x-axis ticks (every 10 minutes)
                            minute_intervals = 10
                            minutes_major = np.arange(0, total_duration + 60*minute_intervals, 60*minute_intervals)
                            
                            if len(minutes_major) > 0:
                                ax_line.set_xticks(minutes_major)
                                ax_line.set_xticklabels([f"{int(m/60)}m" for m in minutes_major])
                                
                            # Add minute markers as vertical lines (reduced number for performance)
                            minutes = np.arange(0, total_duration, 300)  # Every 5 minutes
                            for m in minutes:
                                ax_line.axvline(x=m, color='gray', linestyle='--', alpha=0.3)
                                
                            ax_line.set_xlim(0, total_duration)
                            ax_line.legend(loc='upper right')
                            
                            # 2. Trend line plot (1-minute bins)
                            trend_bin_size = 60  # 1-minute bins
                            n_bins_trend = int(total_duration / trend_bin_size)
                            
                            if n_bins_trend > 0:
                                trend_hist, trend_edges = np.histogram(
                                    spike_times_array, bins=n_bins_trend, range=(0, total_duration))
                                
                                # Convert to firing rate
                                trend_rate = trend_hist / trend_bin_size
                                trend_centers = (trend_edges[:-1] + trend_edges[1:]) / 2
                                
                                # Calculate moving average
                                window_size = max(3, int(n_bins_trend / 20))  # Adapt window to recording length
                                trend_smooth = np.convolve(trend_rate, 
                                                        np.ones(window_size)/window_size, 
                                                        mode='same')
                                
                                # Plot trend with LAVENDER color as requested
                                ax_trend.plot(trend_centers, trend_rate, 'o', markersize=3, alpha=0.5, color='#9966CC')
                                ax_trend.plot(trend_centers, trend_smooth, linewidth=2, color='#9966CC')
                                
                                # Add reference line at mean firing rate
                                mean_rate = firing_rates[cluster_id]
                                ax_trend.axhline(y=mean_rate, color='k', linestyle='-', 
                                            label=f"Mean rate: {mean_rate:.2f} Hz")
                                
                                # Add threshold reference
                                ax_trend.axhline(y=firing_threshold, color='r', linestyle='--', 
                                            label=f"Threshold ({firing_threshold} Hz)")
                                
                                # Add y=0 line with emphasis
                                ax_trend.axhline(y=0, color='k', linewidth=1.5)
                                
                                # Add more minor ticks to axes for better zoom experience
                                ax_trend.yaxis.set_minor_locator(AutoMinorLocator(5))
                                ax_trend.xaxis.set_minor_locator(AutoMinorLocator(4))
                                
                                # Set up axes
                                ax_trend.set_title(f"Firing Rate Trend - Cluster {cluster_id} (1-minute bins)")
                                ax_trend.set_xlabel("Time (minutes)")
                                ax_trend.set_ylabel("Firing Rate (Hz)")
                                
                                # Set reasonable y-axis limits based on data
                                ax_trend.set_ylim(0, min(max_rate, max(trend_smooth)*1.5))
                                
                                # Set x-ticks
                                trend_ticks = np.arange(0, total_duration + 600, 600)  # Every 10 minutes
                                if len(trend_ticks) > 0:
                                    ax_trend.set_xticks(trend_ticks)
                                    ax_trend.set_xticklabels([f"{int(x/60)}" for x in trend_ticks])
                                
                                ax_trend.legend(loc='upper right')
                        else:
                            # Handle case with no spikes
                            for ax in [ax_line, ax_trend]:
                                ax.text(0.5, 0.5, "No spikes in this cluster", 
                                    horizontalalignment='center', verticalalignment='center')
                        
                        neuron_figure.tight_layout()
                        neuron_canvas.draw()
                    except Exception as e:
                        logger.error(f"Error updating neuron view: {str(e)}")
                        import traceback
                        traceback.print_exc()
                
                # Button handlers
                def on_prev_button():
                    current_neuron_idx[0] = (current_neuron_idx[0] - 1) % len(sorted_clusters)
                    update_neuron_view()
                    #  sync with Phy GUI
                    controller.supervisor.select([sorted_clusters[current_neuron_idx[0]]])
                
                def on_next_button():
                    current_neuron_idx[0] = (current_neuron_idx[0] + 1) % len(sorted_clusters)
                    update_neuron_view()
                    #  sync with Phy GUI
                    controller.supervisor.select([sorted_clusters[current_neuron_idx[0]]])
                
                # Dropdown handler
                def on_cluster_selected(index):
                    selected_cluster = cluster_selector.itemData(index)
                    try:
                        selected_idx = sorted_clusters.index(selected_cluster)
                        current_neuron_idx[0] = selected_idx
                        update_neuron_view()
                        #  sync with Phy GUI
                        controller.supervisor.select([selected_cluster])
                    except (ValueError, TypeError):
                        pass
                
                # Connect buttons and dropdown
                prev_btn.clicked.connect(on_prev_button)
                next_btn.clicked.connect(on_next_button)
                cluster_selector.currentIndexChanged.connect(on_cluster_selected)
                
                # Connect colorbar range dropdown and spinner
                def on_colorbar_range_change():
                    if colorbar_range.currentData() == "custom":
                        max_scale_spinner.setEnabled(True)
                    else:
                        max_scale_spinner.setEnabled(False)
                    update_overview_plots()
                
                colorbar_range.currentIndexChanged.connect(on_colorbar_range_change)
                max_scale_spinner.valueChanged.connect(update_overview_plots)
                
                # Connect sorting dropdown to update function
                sort_options.currentIndexChanged.connect(update_overview_plots)
                
                # Set dialog layout and show
                dialog.setLayout(main_layout)
                
                # Initialize view with first neuron
                update_neuron_view()
                
                # Initial plot update
                update_overview_plots()
                
                # Apply initial font size
                update_fonts(font_spinner.value())
                
                # Add a note about review tags
                review_note = QLabel("Note: Clusters marked for review will be saved in the 'needs_review' metadata field.\n"
                                    "You can filter these in the Phy GUI using: needs_review == \"yes\"")
                review_note.setStyleSheet("color: #555555; font-style: italic;")
                main_layout.addWidget(review_note)
                
                # Explicitly set to first tab (Overview)
                tab_widget.setCurrentIndex(0)
                
                # Execute dialog
                dialog.exec_()