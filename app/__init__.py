from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ragr")
except PackageNotFoundError:
    __version__ = "dev"