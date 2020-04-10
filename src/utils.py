"""Utils functions module"""

from pathlib import Path
import json
import pickle


def persist_local_artifact(artifact, filename):
    """
    Persist pickle or json Python objects locally

    Parameters
    ----------
    artifact : pickle or dict
        Python object to be persisted
    filename : str
        Local file location
    """
    extension = Path(filename).suffix
    assert extension in ['.pkl', '.json'], 'the filename extension must be pkl or json'
    if extension == '.pkl':
        with open(filename, 'wb') as f:
            pickle.dump(artifact, f, pickle.HIGHEST_PROTOCOL)
    elif extension == '.json':
        with open(filename, 'wt') as f:
            json.dump(artifact, f)
