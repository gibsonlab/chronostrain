"""
    initialize_database.py
    Author: Younhun Kim

    Initialize a database instance to download the necessary files.
"""
from chronostrain import cfg, logger


def main():
    logger.info("Initializing database.")
    db = cfg.database_cfg.get_database(force_refresh=True)

    logger.info("Finished initializing database.")


if __name__ == "__main__":
    main()
