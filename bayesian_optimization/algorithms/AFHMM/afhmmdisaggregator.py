from __future__ import print_function, division
from hmmlearn import hmm
import numpy as np
import pandas as pd
from nilmtk.legacy.disaggregate import Disaggregator

class AFHMMDisaggregator(Disaggregator):

    def __init__(self):
        super().__init__()
        self.models = {}
        self.appliances = []

    def train(self, train_elec, sample_period=60):
        self.appliances = [meter.appliances[0].type['type']
                           for meter in train_elec.submeters().meters
                           if meter.appliances]

        for appliance in self.appliances:
            meter = train_elec.submeters().select_using_appliances(type=appliance)
            power_series = meter.total_energy().values.flatten()
            power_series = power_series[~np.isnan(power_series)]

            if len(power_series) == 0:
                print(f"No data for appliance: {appliance}, skipping...")
                continue

            if len(power_series) >= 2:
                try:
                    model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)
                    model.fit(power_series.reshape(-1, 1))
                    self.models[appliance] = model
                except ValueError as e:
                    print(f"Error training HMM for appliance {appliance}: {e}. Skipping.")
            else:
                print(f"Not enough data for appliance {appliance}. Using fallback model.")
                self.models[appliance] = {'fallback': np.mean(power_series)}

    def disaggregate(self, mains_signal, output_datastore):
        """
        Perform disaggregation on mains signal and store results in output_datastore
        """
        print("========== DISAGGREGATE ===========")
        print(f"Type of mains_signal: {type(mains_signal)}")

        if isinstance(mains_signal, np.ndarray):
            mains_signal = pd.Series(mains_signal)
        elif not isinstance(mains_signal, (pd.Series, pd.DataFrame)):
            raise ValueError("mains_signal must be a pandas Series or DataFrame.")

        disaggregated_data = self.disaggregate_chunk(mains_signal)

        for appliance in disaggregated_data.columns:
            output_datastore.append(key=appliance, value=disaggregated_data[appliance])

        print("✅ Disaggregation completed and stored in output_datastore.")

    def disaggregate_chunk(self, mains_chunk, **load_kwargs):
        print("🔍 AFHMM disaggregate_chunk called.")

        if not isinstance(mains_chunk, (pd.Series, pd.DataFrame)):
            raise ValueError("mains_chunk must be a pandas Series or DataFrame.")

        disaggregated = pd.DataFrame(index=mains_chunk.index)

        for appliance, model in self.models.items():
            if isinstance(model, dict):  # Fallback
                disaggregated[appliance] = model['fallback']
            else:
                X = mains_chunk.values.reshape(-1, 1)
                hidden_states = model.predict(X)
                means = model.means_.flatten()
                predicted_power = np.array([means[state] for state in hidden_states])
                disaggregated[appliance] = predicted_power

        return disaggregated
