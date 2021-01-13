# -*- coding: utf-8 -*-

import json

class Settings(object):
    
    _settings = None
    _configFile = 'code/config.json'
    
    @staticmethod
    def getInstance():
        if not Settings._settings:
            json_data=open(Settings._configFile)
            Settings._settings = json.load(json_data)
            print("Config file: "+ Settings._configFile)
        assert Settings._settings
        return Settings._settings

    @staticmethod
    def get(key):
        assert key
        value = Settings.getInstance()[key]
        assert value
        return value
