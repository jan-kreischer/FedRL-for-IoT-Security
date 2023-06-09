import socket
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', required=True, help='Please set the ip address of the MTDDeployerServer.')
    parser.add_argument('--port', required=True, help='Please set the port of the MTDDeployerServer.', type=int)
    parser.add_argument('--attack', required=True, help='Add the attack keyword you want to send')

    args = parser.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((str(args.ip), args.port))

    s.send(bytes(str(args.attack), 'utf-8'))


if __name__ == '__main__':
    main()
