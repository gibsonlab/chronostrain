"""
    initialize_database.py
    Author: Younhun Kim

    Initialize a database instance to download the necessary files.
"""
from chronostrain import cfg, logger


def main():
    logger.info("Initializing database.")
    cfg.database_cfg.get_database()
    logger.info("Finished initializing database.")


if __name__ == "__main__":
    main()
