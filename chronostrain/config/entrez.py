from .base import AbstractConfig
from chronostrain.logging import create_logger
logger = create_logger(__name__)


class EntrezConfig(AbstractConfig):
    def __init__(self, cfg: dict):
        super().__init__("Entrez", cfg)
        self.enabled = self.get_bool("ENABLED", False)
        self.email = self.get_str("EMAIL")
        self.entrez_api_initialized = False

    def ensure_enabled(self):
        if not self.enabled:
            raise RuntimeError(
                f"To enable Entrez API access, please set [{self.name}] ENABLED=True "
                "in the config INI file (Ensure that the credentials are correct.)."
            )
        if not self.entrez_api_initialized:
            self.initialize_api()
            self.entrez_api_initialized = True

    def initialize_api(self):
        from Bio import Entrez
        if len(self.email) == 0:
            raise RuntimeError(
                "To meet NCBI's Entrez API conventions [https://www.ncbi.nlm.nih.gov/books/NBK25497/#chapter2.Usage_Guidelines_and_Requiremen], "
                "please specify an Email address in the configuration (We do not store or use your e-mail for any other purpose)."
            )
        logger.info(f"Using email `{self.email}` for Entrez.")
        Entrez.email = self.email
        Entrez.tool = "chronostrain"
