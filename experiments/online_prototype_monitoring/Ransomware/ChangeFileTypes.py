import multiprocessing
import random
import string
import os
import glob
import argparse

import setproctitle

import KillProcess

characters_to_use = string.ascii_letters + string.digits

extensions = [
    'jpg', 'jpeg', 'bmp', 'gif', 'png', 'svg', 'psd', 'raw',  # images
    'mp3', 'mp4', 'm4a', 'aac', 'ogg', 'flac', 'wav', 'wma', 'aiff', 'ape',  # music and sound
    'avi', 'flv', 'm4v', 'mkv', 'mov', 'mpg', 'mpeg', 'wmv', 'swf', '3gp',  # Video and movies

    'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx',  # Microsoft office
    'odt', 'odp', 'ods', 'txt', 'rtf', 'tex', 'pdf', 'epub', 'md',  # OpenOffice, Adobe, Latex, Markdown, etc
    'yml', 'yaml', 'json', 'xml', 'csv',  # structured data
    'db', 'sql', 'dbf', 'mdb', 'iso',  # databases and disc images

    'html', 'htm', 'xhtml', 'php', 'asp', 'aspx', 'js', 'jsp', 'css',  # web technologies
    'c', 'cpp', 'cxx', 'h', 'hpp', 'hxx',  # C source code
    'java', 'class', 'jar',  # java source code
    'ps', 'bat', 'vb',  # windows based scripts
    'awk', 'sh', 'cgi', 'pl', 'ada', 'swift',  # linux/mac based scripts
    'go', 'py', 'pyc', 'bf', 'coffee',  # other source code files

    'zip', 'tar', 'tgz', 'bz2', '7z', 'rar', 'bak',  # compressed formats
]


def main():
    setproctitle.setproctitle('ChangeFileTypes')

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--p', help='This specifies the path, in which the dummy files should be created.')
    parser.add_argument('--extensions', '--e', help='This specifies which file types should have their extensions '
                                                    'changed. Provide file type extensions as a list e.g.: --e pdf doc '
                                                    'docx', required=False, default=[], nargs='+')
    parser.add_argument('--recursive', '--r', help='This boolean value specifies if file extensions should be changed '
                                                   'for subdirectories as well in a recursive manner.', default=False, action='store_true')
    args = parser.parse_args()

    main_directory = args.path

    if not os.path.exists(main_directory):
        raise Exception('Given path is an invalid path')

    sub_proc = multiprocessing.Process(target=KillProcess.main)
    sub_proc.daemon = True
    sub_proc.start()

    extensionsToConsider = []

    for extension in args.extensions:
        extensionsToConsider.append(str(extension))

    if not extensionsToConsider:
        extensionsToConsider = extensions

    print(extensionsToConsider)

    dictionary_with_mappings = create_dictionary_for_extensions(extensionsToConsider)
    change_file_extensions(main_directory, dictionary_with_mappings, args.recursive)

    while True:
        if not sub_proc.is_alive():
            inverse_dictionary = {v: k for k, v in dictionary_with_mappings.items()}
            change_file_extensions(main_directory, inverse_dictionary, args.recursive)
            break


def create_dictionary_for_extensions(extensionsToConsider):
    dictionary_extension_to_pseudo_extension = {}
    pseudo_extensions = []
    for extension in extensionsToConsider:
        while True:
            num_of_letters_for_speudo_ext = random.randrange(3, 7)
            pseudo_extension = ''
            for i in range(num_of_letters_for_speudo_ext):
                pseudo_extension = pseudo_extension + random.choice(characters_to_use)
            if pseudo_extension not in extension and pseudo_extension not in pseudo_extensions:
                break
        dictionary_extension_to_pseudo_extension[extension] = pseudo_extension
        pseudo_extensions.append(pseudo_extension)
    return dictionary_extension_to_pseudo_extension


def change_file_extensions(directory, dictionary_ext_to_pext, doRecursive):
    files = []
    dirs = []
    for fileOrDir in glob.glob(os.path.join(directory,'*')):
        if os.path.isdir(fileOrDir):
            dirs.append(fileOrDir)
        else:
            files.append(fileOrDir)

    for file in files:
        file_split = file.rsplit('.')
        if len(file_split) > 1 and file_split[1] in dictionary_ext_to_pext.keys():
            try:
                os.rename(file, file_split[0] + "." + dictionary_ext_to_pext[file_split[1]])
            except:
                pass

    if doRecursive:
        for subdir in dirs:
            change_file_extensions(subdir, dictionary_ext_to_pext, doRecursive)


if __name__ == '__main__':
    main()
