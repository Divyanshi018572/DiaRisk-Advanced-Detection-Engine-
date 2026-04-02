import os

def get_project_root():
    """Returns the root path of the project."""
    return os.path.dirname(os.path.abspath(__file__))

def get_data_path(subdir="raw", filename=None):
    """Returns the path to the data directory."""
    path = os.path.join(get_project_root(), "data", subdir)
    if filename:
        path = os.path.join(path, filename)
    return path

def get_models_path(filename=None):
    """Returns the path to the models directory."""
    path = os.path.join(get_project_root(), "models")
    if filename:
        path = os.path.join(path, filename)
    return path

def get_outputs_path(filename=None):
    """Returns the path to the outputs directory."""
    path = os.path.join(get_project_root(), "outputs")
    if filename:
        path = os.path.join(path, filename)
    return path

def get_pipeline_path(filename=None):
    """Returns the path to the pipeline directory."""
    path = os.path.join(get_project_root(), "pipeline")
    if filename:
        path = os.path.join(path, filename)
    return path
