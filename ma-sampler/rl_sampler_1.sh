#!/bin/bash -u

##############################################################
#############		SCRIPT CONFIGURATION		##############
##############################################################
#	Set language to make sure same separator (, and .) config is being used
export LC_ALL=C.UTF-8
#	Events to monitor using perf
targetEvents="alarmtimer:alarmtimer_fired,alarmtimer:alarmtimer_start,block:block_bio_backmerge,block:block_bio_remap,block:block_dirty_buffer,block:block_getrq,block:block_touch_buffer,block:block_unplug,cachefiles:cachefiles_create,cachefiles:cachefiles_lookup,cachefiles:cachefiles_mark_active,clk:clk_set_rate,cpu-migrations,cs,dma_fence:dma_fence_init,fib:fib_table_lookup,filemap:mm_filemap_add_to_page_cache,gpio:gpio_value,ipi:ipi_raise,irq:irq_handler_entry,irq:softirq_entry,jbd2:jbd2_handle_start,jbd2:jbd2_start_commit,kmem:kfree,kmem:kmalloc,kmem:kmem_cache_alloc,kmem:kmem_cache_free,kmem:mm_page_alloc,kmem:mm_page_alloc_zone_locked,kmem:mm_page_free,kmem:mm_page_pcpu_drain,mmc:mmc_request_start,net:net_dev_queue,net:net_dev_xmit,net:netif_rx,page-faults,pagemap:mm_lru_insertion,preemptirq:irq_enable,qdisc:qdisc_dequeue,random:get_random_bytes,random:mix_pool_bytes_nolock,random:urandom_read,raw_syscalls:sys_enter,raw_syscalls:sys_exit,rpm:rpm_resume,rpm:rpm_suspend,sched:sched_process_exec,sched:sched_process_free,sched:sched_process_wait,sched:sched_switch,sched:sched_wakeup,signal:signal_deliver,signal:signal_generate,skb:consume_skb,skb:kfree_skb,skb:skb_copy_datagram_iovec,sock:inet_sock_set_state,task:task_newtask,tcp:tcp_destroy_sock,tcp:tcp_probe,timer:hrtimer_start,timer:timer_start,udp:udp_fail_queue_rcv_skb,workqueue:workqueue_activate_work,writeback:global_dirty_state,writeback:sb_clear_inode_writeback,writeback:wbc_writepage,writeback:writeback_dirty_inode,writeback:writeback_dirty_inode_enqueue,writeback:writeback_dirty_page,writeback:writeback_mark_inode_dirty,writeback:writeback_pages_written,writeback:writeback_single_inode,writeback:writeback_write_inode,writeback:writeback_written"
#	Resource monitoring
resourceMonitor=false

#	ransomware
ransomwareMonitor=false

#	Time window per sample
timeWindowSeconds=50
#	Number of samples to take (Monitored time will be: timeWindowSeconds*desiredSamples)
desiredSamples=99999
#	Total time monitored (NOT TAKING IN CONSIDERATION TIME BETWEEN SCREENSHOTS)
timeAcumulative=0

#	Final and temporal output files
finalOutput="/tmp/samples_$1_$(date +'%Y-%m-%d-%H-%M')_${timeWindowSeconds}s"
tempOutput=temp

##############################################################
#############		  OUTPUT FORMATTING  		##############
##############################################################

#	Perf execution to get headers/placeholders for the monitored data (perf might output them in a different order than we defined them in $targetEvents)
perf stat -e "$targetEvents" --o $tempOutput -a sleep 1

#	We calculate the number of events (for output formatting)
numberEvents=$(($(wc -l < $tempOutput)-7))

#	Resources to monitor manually (without perf)
resourcePlaceholder=""
if [ "$resourceMonitor" = true ]
then
	cpuMetrics="cpuUser,cpuSystem,cpuNice,cpuIdle,cpuIowait,cpuHardIrq,cpuSoftIrq,"
	taskMetrics="tasks,tasksRunning,tasksSleeping,tasksStopped,tasksZombie,"
	ramMetrics="ramFree,ramUsed,ramCache,"
	swapMetrics="memAvail,"
	resourcePlaceholder="${cpuMetrics}${taskMetrics}${ramMetrics}${swapMetrics}iface0RX,iface0TX,iface1RX,iface1TX,"
fi

if [ "$ransomwareMonitor" = true ]
then
	resourcePlaceholder="${resourcePlaceholder}numEncrypted,"
fi



#	Output header/placeholder to output file
echo -n "time,timestamp,seconds,connectivity,${resourcePlaceholder}" > "$finalOutput"

placeholder=$(cat $tempOutput | cut -b 25- | cut -d " " -f 1 | tail -n +6 | head -n -3 | tr "\n" "," | sed 's/.$//')

echo "$placeholder" >> "$finalOutput"

##############################################################
#############		   MONITORING LOOP			##############
##############################################################
while :
do
	##############################################################
	#############		   DATA COLLECTION			##############
	##############################################################
	#	Internet connection check via ping
	if ping -q -c 1 -W 1.5 8.8.8.8 >/dev/null; then
		connectivity="1"
	else
		connectivity="0"
	fi
	timestamp=$(($(date +%s%N)/1000000))
	
	#	First capture for network resources, results will be calculated as the difference between this capture and the one taken later
	if [ "$resourceMonitor" = true ]
	then
		oldNetworkTraffic=$(ifconfig | grep -oP -e "bytes \K\w+" | head -n 4)
	fi
	
	#	Perf will monitor the events and also act as a "sleep" between both network captures
	perf stat -e "$targetEvents" --o "$tempOutput" -a sleep "$timeWindowSeconds"


	if [ "$resourceMonitor" = true ]
	then
		#	Second capture of network resources
		newNetworkTraffic=$(ifconfig | grep -oP -e "bytes \K\w+" | head -n 4)
		#	Capture with top for CPU usage, tasks and RAM usage
		#	This is not part of "DATA EXTRACTION/CALCULATION" but it's temporaly moved here
		topResults=$(top -bn 2 -d 1)
	fi

	##############################################################
	#############	DATA EXTRACTION/CALCULATION	  ##############
	##############################################################
	resourceSample=""
	if [ "$resourceMonitor" = true ]
	then
		#	Network data calculation (newer capture - older capture)
		networkTraffic="$(paste <(echo "$newNetworkTraffic") <(echo "$oldNetworkTraffic") | awk 'BEGIN { ORS = "," }{ print $1 - $2 }')"
		networkTraffic=${networkTraffic::-1}
		#	Data extraction from top results
		taskSamples=$(echo "$topResults" | grep "Tasks:" | tail -n 1 | tr -s " " | cut -d " " -f 2,4,6,8,10 --output-delimiter=",")
		cpuSamples=$(echo "$topResults" | grep "%Cpu" | tail -n 1 | tr -s " " | tr "," "." | cut -d " " -f 2,4,6,8,10,12,14 --output-delimiter=",")
		ramSamples=$(echo "$topResults" | grep "KiB Mem" | tail -n 1 | tr -s " " | cut -d " " -f 6,8,10 --output-delimiter=",")
		swapSamples=$(echo "$topResults" | grep "KiB Swap:" | tail -n 1 | tr -s " " | cut -d " " -f 9 --output-delimiter=",")
		resourceSample="${cpuSamples},${taskSamples},${ramSamples},${swapSamples},${networkTraffic},"
	fi

	if [ "$ransomwareMonitor" = true ]
	then
		numEncrypted=$(find "TestFileStructure/" -type f | grep "wasted$" | wc -l)

		resourceSample="${resourceSample}${numEncrypted},"
	fi

	
	#	Data extraction from perf results
	sample=$(cat "$tempOutput" | cut -c -20 | tr -s " " | tail -n +6 | head -n -3 | tr "\n" "," | sed 's/ //g'| sed 's/.$//')
	seconds=$(cat "$tempOutput" | tr -s " " | cut -d " " -f 2 | tail -n 2 | head -n 1| tr "," ".")
	

	#	Cumulative sum of seconds calculation
	timeAcumulative=$(awk "BEGIN{ print $timeAcumulative + $seconds }")
		
	##############################################################
	#############			   OUTPUT				##############
	##############################################################
	
	#	Output to file
	echo "$timeAcumulative,$timestamp,$seconds,$connectivity,${resourceSample}$sample" >> "$finalOutput"
done

#rm -f "$tempOutput"
