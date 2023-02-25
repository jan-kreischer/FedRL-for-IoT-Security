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
    memtotal = [l.split(",")[6] for l in ram_lines][1:]
    cpuuser = [l.split(",")[2] for l in cpu_lines][1:]
    cpusystem = [l.split(",")[-4] for l in cpu_lines][1:]

    memactive = [float(e) for e in memactive][:400]
    memtotal = [float(e) for e in memtotal][:400]
    cpuuser = [float(e) for e in cpuuser][:400]
    cpusystem = [float(e) for e in cpusystem][:400]

    fig, axs = plt.subplots(nrows=2, ncols=2)
    #fig.subplots_adjust(bottom=0.1, top=0.8)
    axs = axs.ravel().tolist()
    #fig.suptitle("Resource Evaluation")
    fig.set_figheight(8)
    fig.set_figwidth(18)
    fig.tight_layout(pad=5.0)

    axs[1].plot(range(len(memactive)), memactive, "-b", label="Mem Active")
    axs[1].set_ylabel("MB")
    axs[1].set_xlabel("seconds")
    axs[1].set_title("Mem Active", fontsize='xx-large')

    axs[0].plot(range(len(memtotal)), memtotal, "-b", label="Mem Free")
    axs[0].set_ylabel("% Free")
    axs[0].set_xlabel("seconds")
    axs[0].set_title("Mem Free", fontsize='xx-large')

    axs[2].plot(range(len(cpuuser)), cpuuser, "-r", label="CPU User")
    axs[2].set_ylabel("% User")
    axs[2].set_xlabel("seconds")
    axs[2].set_title("CPU User", fontsize='xx-large')

    axs[3].plot(range(len(cpusystem)), cpusystem, "-r", label="CPU System")
    axs[3].set_ylabel("% System")
    axs[3].set_xlabel("seconds")
    axs[3].set_title("CPU System", fontsize='xx-large')


    fig.savefig(f'res_eval.png', dpi=100)


    #plt.title("CPU and RAM Usage")
    #plt.legend()
    #plt.show()
