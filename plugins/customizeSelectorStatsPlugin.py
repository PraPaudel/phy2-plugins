# import from plugins/custom_columns.py
"""Show how to customize the columns in the cluster and similarity views."""

from phy import IPlugin, connect


class customizeSelectorStatsPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_controller_ready(sender):
            controller.supervisor.columns = ['id', 'fr', 'amp', 'sh', 'ch','n_spikes', 'quality']

            # controller.supervisor.columns = ['id', 'depth', 'fr', 'Amplitude', 'n_spikes', 'quality',
            # 'fractionRPV', 'percentSpikesMissing'] # can be used if used bombcell during preprocessing
