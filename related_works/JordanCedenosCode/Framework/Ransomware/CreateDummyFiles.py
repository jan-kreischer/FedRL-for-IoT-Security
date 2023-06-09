import os
import glob
import re
import argparse
import random
import setproctitle
import multiprocessing
import KillProcess


def main():
    setproctitle.setproctitle('CreateDummyFiles')

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--p', help='This specifies the path, in which the dummy files should be created.')
    parser.add_argument('--numberOfDummyFiles', '--ndf',
                        help='Chooses the number of concurrent dummy files that are allowed.', required=False, type=int,
                        default=30)
    parser.add_argument('--numberOfDummyFilesPerSubdirectory', '--ndfpsd',
                        help='Chooses the number of dummy files, that are created before moving to the next '
                             'subdirectory.',
                        required=False, type=int, default=15)
    parser.add_argument('--size', '--s', help='Chooses the file size of the dummy files.', required=False, type=int,
                        default=10)
    parser.add_argument('--extension', '--e', help='Chooses the extension of the dummy files.', required=False,
                        default='pdf')
    args = parser.parse_args()

    main_directory = args.path

    if not os.path.exists(main_directory):
        raise Exception('Given path is an invalid path')

    FILENAME = "honey_"
    EXTENSION = '.' + args.extension
    SIZE_1_MB = 1048576
    SIZE = args.size * SIZE_1_MB
    i = 1
    h = 1
    flag_value = 0

    current_directory = find_random_sub_directory(main_directory)
    next_directory = os.path.join(current_directory, 'honey-directory-' + str(h))
    create_next_directory(next_directory)
    h += 1

    sub_proc = multiprocessing.Process(target=KillProcess.main)
    sub_proc.daemon = True
    sub_proc.start()

    while sub_proc.is_alive():
        file_list = glob.glob(os.path.join(main_directory, '**/honey_*'), recursive=True)

        file_list.sort(key=get_num_for_natural_sort)

        if i % (args.numberOfDummyFilesPerSubdirectory + 1) == 0 and i != flag_value:
            current_directory, next_directory, h = move_current_directory_to_next_level(next_directory, h)
            flag_value = i

        if args.numberOfDummyFiles <= len(file_list):
            for file in file_list:
                if not file.endswith('pdf'):
                    os.remove(file)
        else:
            try:
                with open(os.path.join(current_directory, FILENAME + str(i) + EXTENSION), "wb") as file:
                    file.seek(SIZE - 1)
                    file.write(b"\0")
                    i += 1
                    print('New file created: ' + file.name)
            except FileExistsError:
                pass


def find_random_sub_directory(directory):
    dirs = next(os.walk(directory))[1]
    if dirs:
        return os.path.join(directory, random.choice(dirs))
    else:
        create_next_directory(os.path.join(directory, 'start-honey'))
        return os.path.join(directory, 'start-honey')


def get_num_for_natural_sort(file_name):
    return int(re.findall('\d+', file_name)[-1])


def move_current_directory_to_next_level(next_directory, level_of_dirs):
    new_level_of_dirs = level_of_dirs + 1
    new_next_directory = os.path.join(next_directory, 'honey-directory-' + str(new_level_of_dirs))
    create_next_directory(new_next_directory)
    print('Created new directory: ' + new_next_directory)
    return next_directory, new_next_directory, new_level_of_dirs


def create_next_directory(next_directory):
    try:
        os.mkdir(next_directory)
    except FileExistsError:
        pass


if __name__ == '__main__':
    main()
