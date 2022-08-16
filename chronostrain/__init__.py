from .logging import create_logger
logger = create_logger(__name__)

# Note: don't load anything else here --- encourage lazy loading of configuration (incase bare logging is required).
