import os
import re
import subprocess
import random
import time
from urllib.request import urlopen


def main():
    additional_ips_to_discard = []

    while True:
        gateway_ip = get_gateway_ip()
        ip = generate_ip_address(gateway_ip, additional_ips_to_discard)
        additional_ips_to_discard.append(ip)
        cmd_change_ip_address = 'ifconfig eth0 ' + ip
        cmd_add_route = 'route add default gw {} eth0'.format(gateway_ip)
        cmd_restart_electro_sense_service = 'sudo systemctl restart electrosense-sensor-mqtt'

        os.system(cmd_change_ip_address)
        os.system(cmd_add_route)
        time.sleep(5)

        if connection_is_live():
            os.system(cmd_restart_electro_sense_service)
            break


def generate_ip_address(gateway_ip, additional_ips_to_discard):
    list_of_possible_ip_endings = list(range(0, 255))

    gateway_ending = gateway_ip.split('.')[3]
    list_of_possible_ip_endings.remove(int(gateway_ending))

    endings_to_remove = get_ip_endings_to_remove('0'.join(gateway_ip.rsplit(gateway_ending, 1)))
    
    for ip in additional_ips_to_discard:
        endings_to_remove.append(ip)

    for i in endings_to_remove:
        ending_to_remove = i.split('.')[3]
        try:
            list_of_possible_ip_endings.remove(int(ending_to_remove))
        except ValueError:
            pass

    return str(random.choice(list_of_possible_ip_endings)).join(gateway_ip.rsplit(gateway_ending, 1))


def get_gateway_ip():
    cmd = ['ip', 'route']
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0]
    regex_search = re.search('(?<=default via )([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})(?= dev)', str(output))
    return str(regex_search.group(1))


def get_ip_endings_to_remove(ip_with_zero_at_end):
    cmd = ['arp-scan', '--interface=eth0', ip_with_zero_at_end + '/24']
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0]
    list_of_ips = re.findall('([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})', str(output))
    return list_of_ips


def connection_is_live():
    try:
        urlopen("https://www.google.com", timeout=2)
        return True
    except:
        return False


if __name__ == '__main__':
    main()
