# config_reader.py
import yaml
import os

yaml_file = "config.yml"

def ConfigReadYaml():
    global yaml_file
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cwd,yaml_file), "r") as fp:
        return yaml.safe_load(fp)