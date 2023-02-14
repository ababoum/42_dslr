import numpy as np

class Scaler():
    def __init__(self):
        pass

    @staticmethod
    def minmax(x):
        """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
        Args:
        x: has to be an numpy.ndarray, a vector.
        Returns:
        x' as a numpy.ndarray.
        None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
        Raises:
        This function shouldn't raise any Exception.
        """
        try:
            if not isinstance(x, np.ndarray) or x.size <= 1:
                return None

            min = np.min(x)
            max = np.max(x)
            if min == max:
                return None
            return (x - min) / (max - min)

        except:
            return None

    @staticmethod
    def zscore(x):
        """Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
        Args:
        x: has to be an numpy.ndarray, a vector.
        Returns:
        x' as a numpy.ndarray.
        None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
        Raises:
        This function shouldn't raise any Exception.
        """
        try:
            if not isinstance(x, np.ndarray) or x.size <= 1:
                return None

            mean = np.mean(x)
            std = np.std(x)
            if std == 0:
                return None
            return (x - mean) / std

        except:
            return None