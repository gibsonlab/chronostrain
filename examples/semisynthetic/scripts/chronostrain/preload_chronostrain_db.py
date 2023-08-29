from chronostrain.config import cfg


def main():
    _ = cfg.database_cfg.get_database()


if __name__ == "__main__":
    main()
