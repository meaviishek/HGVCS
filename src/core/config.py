class ConfigManager:
    def __init__(self, config):
        self.config = config

    def get(self, key, default=None):
        return self.config.get(key, default)
