import os
import shutil

ETC_LD_SO_PRELOAD = '/etc/ld.so.preload'
ETC_LD_SO_PRELOAD_NEW = '/etc/ld.so.preload.new'
LD_SO_PRELOAD = 'backupLSP'
LD_SO = '/lib/arm-linux-gnueabihf/ld-2.24.so'
LD_SO_NEW = '/lib/arm-linux-gnueabihf/ld-2.24.so.new'
LD_SO_OLD = '/lib/arm-linux-gnueabihf/ld-2.24.so.old'

def main():
    if check_if_ld_so_preload_was_unhooked_by_malware():
        unlink_fake_etc_ld_so_preload()
        replace_with_known_good_preload_file()
    else:
        replace_with_known_good_preload_file()


def unlink_fake_etc_ld_so_preload():
    with open(LD_SO, 'r', encoding='latin-1') as file:
        flag = False
        for line in file:
            if flag:
                flag = False
                splitted_line = line.split('\x00')
                for elem in splitted_line:
                    if not elem:
                        splitted_line.remove(elem)
                elem_to_remove = splitted_line[4]
            elif 'prelink checking: %s' in line:
                flag = True
    os.system("sed -i 's|"+elem_to_remove+"|"+ETC_LD_SO_PRELOAD+"|g' "+LD_SO)


def check_if_ld_so_preload_visible():
    return os.path.exists(ETC_LD_SO_PRELOAD)


def replace_with_known_good_preload_file():
    shutil.copy(os.path.join(os.getcwd(), LD_SO_PRELOAD), ETC_LD_SO_PRELOAD_NEW)

    os.system("chmod --reference=" + ETC_LD_SO_PRELOAD + ' ' + ETC_LD_SO_PRELOAD_NEW)
    os.system("chown --reference=" + ETC_LD_SO_PRELOAD + ' ' + ETC_LD_SO_PRELOAD_NEW)
    os.system('rm -f ' + ETC_LD_SO_PRELOAD)

    os.rename(ETC_LD_SO_PRELOAD_NEW, ETC_LD_SO_PRELOAD)


def check_if_ld_so_preload_was_unhooked_by_malware():
    with open(LD_SO, errors='ignore') as f:
        for line in f:
            if ETC_LD_SO_PRELOAD in line:
                return False
    return True


if __name__ == '__main__':
    main()
