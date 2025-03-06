import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

"""The units of the data are not specified. 

I do not know the unit of the data.

- Energy_price_forecast: Likely in currency per unit of energy (e.g., $/kWh).
- pv_forecast: Likely in power units (e.g., p.u. for per unit).
- wind_forecast: Likely in power units (e.g., p.u. for per unit).
- Load_forecast: Likely in power units (e.g., p.u. for per unit).
- DG: Represents the distributed generation parameters.
- ES: Represents the energy storage parameters.
"""





def equipment_info():
    '''simulation data '''
    Energy_price_forecast = [16.07, 14.96, 14.28, 13.26, 13.21, 14.08, 15, 16.93, 18.6, 20.41, 22.11, 24.5, 26.57, 28.07, 29.97, 30.66, 35.5, 35.58, 29.54, 26.73, 23.91, 22.16, 19.93, 17.93]
    pv_forecast = [0, 0, 0, 0, 0, 0, 0.0341275572, 0.2688327316, 0.4466907341, 0.7883032491, 0.803898917, 1.0, 0.9681829122, 0.8001925391, 0.659253911, 0.34811071, 0.2309025271, 0.0895788207, 0.0074127557, 0, 0, 0, 0, 0]
    wind_forecast = [1.00, 1.00, 0.98, 0.95, 0.98, 0.95, 0.90, 0.78,0.75, 0.78, 0.80, 0.60, 0.35, 0.30, 0.55, 0.50,0.45, 0.50, 0.60, 0.70, 0.80, 0.85, 0.95, 1.00]
    pv_forecast = np.array(pv_forecast)
    wind_forecast = np.array(wind_forecast)
    Res_forecast = pv_forecast+wind_forecast
    Load_forecast = [0.605585931, 0.607214238, 0.608842546, 0.656974725, 0.709757292, 0.790594894,0.877666948,0.913087467, 0.92913866, 0.931251234, 0.925398882, 0.918255032, 0.910034934, 0.909848113,0.919301226, 0.934087481, 0.958206733, 0.974063106, 0.967885132, 0.94853486, 0.869909243,0.792083167, 0.722252488, 0.65242181]
    Load_forecast = np.array(Load_forecast)*2.5
    DG={'maxp':2,'minp':0.5,} # page 88 Dr.yi
    ES={'soc0':2,'cap':2.5,'maxmum_Ps':1,'cdr':0.95,'esr':0.95}
    # RES_forecast = savgol_filter(RES_forecast, window_length=3, polyorder=2) #
    # power_load = savgol_filter(power_load, window_length=3, polyorder=2) #
    # Energy_price_forecast = savgol_filter(Energy_price_forecast, window_length=3, polyorder=2) #
    # Load_forecast = savgol_filter(Load_forecast, window_length=3, polyorder=2) #
    return Res_forecast,Load_forecast,Energy_price_forecast,DG,ES

def plot_forecast_data():
    """
    Plot forecast data including renewable energy forecast, load forecast, and energy price forecast
    """
    Res_forecast, Load_forecast, Energy_price_forecast, _, _ = equipment_info()
    
    # Create time axis
    hours = list(range(24))
    
    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot renewable energy forecast
    ax1.plot(hours, Res_forecast, 'g-', marker='o', linewidth=2, label='Renewable Energy Forecast')
    ax1.set_title('Renewable Energy Forecast')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Power (p.u.)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot load forecast
    ax2.plot(hours, Load_forecast, 'b-', marker='s', linewidth=2, label='Load Forecast')
    ax2.set_title('Load Forecast')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Power (p.u.)')
    ax2.grid(True)
    ax2.legend()
    
    # Plot energy price forecast
    ax3.plot(hours, Energy_price_forecast, 'r-', marker='^', linewidth=2, label='Energy Price Forecast')
    ax3.set_title('Energy Price Forecast')
    ax3.set_xlabel('Hour')
    ax3.set_ylabel('Price ($/MWh)')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig



plot_forecast_data()



