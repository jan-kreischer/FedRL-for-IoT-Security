import matplotlib.pyplot as plt


nmon = "ransomware_dirtrap_resourcepi_nmon_220915_1052.nmon"

if __name__ == '__main__':
    with open(nmon, 'r') as f:
        lines = f.readlines()

    ram_lines = []
    cpu_lines = []
    for line in lines:
        if line.startswith("CPU_ALL"):
            cpu_lines.append(line)
        if line.startswith("MEM"):
            ram_lines.append(line)

    print(ram_lines)
    print(cpu_lines)

    memactive = [l.split(",")[-5] for l in ram_lines][1:]
    cpuuser = [l.split(",")[2] for l in cpu_lines][1:]

    print(memactive)
    print(cpuuser)

    print(len(memactive))
    plt.plot(range(len(memactive)), memactive, "-b", label="mem active")
    plt.plot(range(len(cpuuser)), cpuuser, "-r", label="cpu user")
    #plt.yscale("linear")
    plt.title("CPU and RAM Usage")
    plt.legend()
    plt.show()
