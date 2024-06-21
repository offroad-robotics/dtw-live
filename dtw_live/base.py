#!/usr/bin/env python3

import json
from abc import ABCMeta, abstractmethod

import numpy as np
# from numpy.lib.arraysetops import isin


class ModelBase:
    __metaclass__ = ABCMeta

    @abstractmethod
    def _to_dict(self):
        """Return a dict representation of the model (for serialization).
        Implemented by each model, based on the parameters required.

        NOTE: This method should check if the model has been fitted before
        attempting to return parameters.

        Returns
        -------
        params : dict
            Fitter model parameters required for serialization.
        """
        pass

    def to_json(self, path):
        """Save a copy of the model parameters required for re-fitting,
        prediction, and scoring to the json file at path.

        Parameters
        ----------
        path : str
            Output filepath.
        """
        with open(path, 'w') as f:
            json.dump(self._to_dict(), f)
    
    @classmethod
    def from_json(cls, path):
        """Load a copy of the model parameters required for re-fitting,
        prediction, and scoring to the json file at path.

        Parameters
        ----------
        path : str
            Valid input filepath.

        Returns
        -------
        model : object
            Instance of the model that inherits :class:`ModelBase`
        """
        with open(path, 'r') as f:
            data = json.load(f)

        model = cls()
        for k, v in data.items():
            if isinstance(v, list):
                v0 = v[0]
                # convert to uniform numpy array if possible
                if isinstance(v0, list):
                    if all(len(v0) == len(s) for s in v):
                        v = np.array(v)
                    else:
                        v = [np.array(s) for s in v]
                else:
                    v = np.array(v)

            setattr(model, k, v)
        
        model.reset()
        return model

    @staticmethod
    def _dict_to_json(d):
        """Recursive conversion of dict to JSON serializable object.
        """
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = ModelBase._dict_to_json(v)
            elif isinstance(v, (list, np.ndarray)):
                d[k] = ModelBase._array_to_json(v)
        
        return d

    @staticmethod
    def _array_to_json(a):
        """Recursive conversion of array-like to JSON serializable object.
        """
        if isinstance(a, np.ndarray):
            return a.tolist()
        elif isinstance(a, list) and isinstance(a[0], np.ndarray):
            return [ModelBase._array_to_json(s) for s in a]
        else:
            return a
