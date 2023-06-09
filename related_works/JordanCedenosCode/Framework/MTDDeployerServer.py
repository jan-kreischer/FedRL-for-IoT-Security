#!/usr/bin/env python3

import json
import os
import socket
import subprocess
import pkg_resources

pkg_resources.require("jsonschema==3.0.0")
import jsonschema
import logging

log = logging.getLogger("main")
logging.basicConfig(level=logging.INFO)

SCRIPT_NAME = 'ScriptName'
PATH = 'AbsolutePath'
PRIORITY = 'Priority'
TYPE = 'Type'
MTD_SOLUTIONS = 'MTDSolutions'
ATTACK_TYPES = 'AttackTypes'
DEPL_POLICY = 'DeploymentPolicy'
ALLOW_EXT = 'AllowAllExternalReports'
WHITE_LIST = 'WhiteListForExternalReports'
PARAMS = 'Params'
RUN_PREFIX = 'RunWithPrefix'
PORT = 'PortToUse'


def main():
    with open('config.json') as json_file:
        data = json.load(json_file)
        validate_config_file(data)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('0.0.0.0', data[PORT]))
    s.listen(10)

    external_allowed = data[ALLOW_EXT]

    white_list = data[WHITE_LIST]

    while True:
        client_socket, address = s.accept()
        log.info('Recieved attack report from ' + address[0])
        if address[0] == client_socket.getsockname()[0] or external_allowed or address[0] in white_list:
            log.info('Attack report is being processed.')
            attack_type = client_socket.recv(1024).decode('utf-8')
            log.info('Attack type: ' + attack_type)
            json_data_attack_types = data[ATTACK_TYPES]
            mtd_solutions = browse_configs(attack_type, json_data_attack_types)
            if mtd_solutions and len(mtd_solutions) > 0:
                os.chdir(mtd_solutions[0][2])
                try:
                    if mtd_solutions[0][3]:
                        os.system(mtd_solutions[0][4] + ' ' + str(mtd_solutions[0][1]) + ' ' + str(mtd_solutions[0][3]))
                    else:
                        os.system(mtd_solutions[0][4] + ' ' + str(mtd_solutions[0][1]))
                    log.info('MTD Solution ' + str(mtd_solutions[0][1]) + ' has been deployed.')
                except Exception:
                    log.error(Exception)
        else:
            log.info('Attack report ignored.')
        client_socket.close()


def browse_configs(attack_type, json_data):
    mtd_solutions = []

    for att_type in json_data:
        if att_type[TYPE] == attack_type:
            # First check if some special deployment rule applies
            output = None
            try:
                os.chdir(att_type[DEPL_POLICY][PATH])
                output = subprocess.check_output(
                    [att_type[DEPL_POLICY][RUN_PREFIX], att_type[DEPL_POLICY][SCRIPT_NAME]]).decode('utf-8')
            except KeyError:
                pass
            if output:
                mtd_id = int(output)
                for mtd_solution in att_type[MTD_SOLUTIONS]:
                    if mtd_solution[PRIORITY] == int(mtd_id):
                        mtd_solutions.append((mtd_solution[PRIORITY], mtd_solution[SCRIPT_NAME], mtd_solution[PATH],
                                              mtd_solution[PARAMS], mtd_solution[RUN_PREFIX]))
                return mtd_solutions
            else:
                for mtd_solution in att_type[MTD_SOLUTIONS]:
                    mtd_params = None
                    try:
                        mtd_params = mtd_solution[PARAMS]
                    except KeyError:
                        pass
                    mtd_solutions.append(
                        (mtd_solution[PRIORITY], mtd_solution[SCRIPT_NAME], mtd_solution[PATH], mtd_params,
                         mtd_solution[RUN_PREFIX]))
                    mtd_solutions.sort()
                return mtd_solutions


def validate_config_file(json_data):
    with open('config-schema.json') as json_schema_file:
        json_schema = json.load(json_schema_file)
        try:
            jsonschema.validate(instance=json_data, schema=json_schema, cls=jsonschema.Draft4Validator)
        except jsonschema.ValidationError as err:
            err.message = 'The config file does not follow the json schema and is therefore not compatible!'
            raise err
        return True


if __name__ == '__main__':
    main()
