
import numpy as np
from scipy.interpolate import interp1d

class ResampleTransform:
    def __init__(self, timeLength, timeWindow=200, method="nearest"):
        """
        Initialize the ResampleTransform class.
        
        Args:
        timeLength (int): The target length of the time series data after resampling.
        timeWindow (float): The actual time period of the given sample.
        method (str): The type of interpolation method. Default to 'linear'.
                      Supported methods: 'linear', 'cubic', 'nearest', 'identity'.
        """
        self.timeLength = timeLength
        self.timeWindow = timeWindow
        self.method = method
        # Create new time points for the resampled data only once during initialization
        self.new_t = np.linspace(0, self.timeWindow, self.timeLength)


    def __call__(self, t, x):
        """
        Apply resampling to the input time series data.
        
        Args:
        t_x (list of tuples): Time series data as a list of (time, data) tuples.
                              Data can be a scalar or an array/vector of features.
        
        Returns:
        np.ndarray: Resampled data array.
        """
        # Unzip the time and data from the input list of tuples
        # print(t_x)
        
        # t, x = zip(*t_x)
        x = np.array(x)
        indx_ = np.any(~np.isnan(x), axis=1)
        x = x[indx_]
        t = np.array(t) - t[0]
        t = t[indx_]
        # print(x.shape)
        # Handle the 'identity' case where no interpolation is needed
        if self.method == "identity":
            if len(x) == self.timeLength:
                return x
            else:
                raise ValueError("Identity method requires input length to match timeLength.")

        # Create new time points for the resampled data
        # new_t = np.linspace(0, self.timeWindow, self.timeLength)
        new_t = self.new_t

        # Determine the number of features in the data
        num_features = x.shape[1] if x.ndim > 1 else 1
        # new_x = np.zeros((self.timeLength, num_features))
        
        t_min, t_max = t.min(), t.max()
        # assert t_min == 0
        if t_min != 0:
            t = np.concatenate(([0],       t))
            x = np.concatenate(([x[0, :]], x))
        if t_max < self.timeWindow:
            t = np.concatenate((t, [self.timeWindow]))
            x = np.concatenate((x, [x[-1, :]]))
        
        interpolator = interp1d(t, x, kind=self.method, bounds_error=False, fill_value="extrapolate", axis=0)
        
        new_x = interpolator(new_t)

        # for i in range(num_features):
        #     # Extract the feature column if multiple features, else the whole array
        #     feature_data = x[:, i] if num_features > 1 else x
            
        #     # Create the interpolator function
        #     if self.method in ["linear", "cubic", "nearest"]:
        #         # Ensure that the interpolation covers the entire new time range
        #         if t.shape != feature_data.shape:
        #             print(t.shape, feature_data.shape)
        #         interpolator = interp1d(t, feature_data, kind=self.method, bounds_error=False, fill_value="extrapolate")
        #     else:
        #         raise ValueError(f"Unsupported interpolation method: {self.method}")
            
        #     # Use the interpolator to resample the data for the current feature
        #     new_x[:, i] = interpolator(new_t)

        return new_x
    