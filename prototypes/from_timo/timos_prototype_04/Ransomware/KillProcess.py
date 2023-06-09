import psutil
import time


def find_most_intensive_processes_pids():
    list_of_processes = []
    tuple_per_process = []
    core_count = psutil.cpu_count()
    for proc in psutil.process_iter():
        try:
            # Get process name & pid from process object.
            proc.cpu_percent()
            list_of_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    time.sleep(2)
    for proc in list_of_processes:
        try:
            proc_percent = proc.cpu_percent() / core_count
            proc_name = proc.name()
            proc_pid = proc.pid
            safe_processes = ['es_sensor', 'CreateDummyFiles', 'ChangeFileTypes']
            if proc_name not in safe_processes and proc_percent >= 10:
                tuple_per_process.append((proc_percent, proc_pid, proc_name))
        except:
            pass

    tuple_per_process.sort(reverse=True)

    return tuple_per_process


def check_if_process_is_ransomware(pid):
    curr_time = time.time()

    list_of_files = set([])
    try:
        process = psutil.Process(pid)

        while time.time() < curr_time + 60:
            for file in process.open_files():
                list_of_files.add(file.path)

        print('Suspect process modified these files: ')
        print(list_of_files)

        if len(list_of_files) >= 5:
            return True
        else:
            return False
    except psutil.NoSuchProcess:
        return False


def main():
    counter = 0
    while True:
        if counter <= 10:
            suspect_processes_pids = find_most_intensive_processes_pids()
            counter += 1
            for tuples in suspect_processes_pids:
                process_pid = tuples[1]
                if process_pid:
                    if check_if_process_is_ransomware(process_pid):
                        proc = psutil.Process(process_pid)
                        name = proc.name()
                        proc.terminate()
                        print("Process: " + name + " with pid [" + str(process_pid) + "] was terminated.")
                        break
        else:
            print('No suspected ransomware could be found, and therefore no process was finished.')
            break


if __name__ == '__main__':
    main()
