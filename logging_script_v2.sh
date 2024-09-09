#!/bin/bash

#create folder to store logged data in
#date_of_creation=`date '+%s'`
#folder_name="performance_log_$date_of_creation"
folder_name=$1
mkdir -p $folder_name

#toggle optional logs
log_sensors=0

#check for GPUs
#0 = unknown; 1 = no gpu of this vendor; 2 = gpu of this vendor exists
gpu_is_intel=0
gpu_is_amd=0
gpu_is_nvidia=0
system_is_tegra=0
system_is_rpi=0
lm_sensors_installed=0 #read temperatures
top_installed=0 #top command does not exist in windows
iotop_installed=0 #read disk usage
top_1_supported=0 #check if top for individual cores is supported
scaling_cur_freq_exists=0 #place where clock speeds are
timeout_time=$2 #logging time

check_for_gpu() {
if command -v hwinfo &> /dev/null
then
	echo "hwinfo installed" | tee -a ./"$folder_name"/console_log.txt
	echo "printing GPU Name to file" | tee -a ./"$folder_name"/console_log.txt
	hwinfo --gfxcard --short >> ./"$folder_name"/gpu_name.txt 
fi
if ! command -v intel_gpu_top &> /dev/null
then
	echo "intel_gpu_top not found" | tee -a ./"$folder_name"/console_log.txt
	gpu_is_intel=1
else
	echo "intel_gpu_top installed, searching for intel gpu" | tee -a ./"$folder_name"/console_log.txt
	#-o - means no graphical mode, output lines to stdout; there is no option 
	if sudo timeout 5 intel_gpu_top -o - | grep 'blitt\|Freq'; then
		echo "Intel GPU found" | tee -a ./"$folder_name"/console_log.txt
		gpu_is_intel=2
	else
		echo "no Intel GPU found" | tee -a ./"$folder_name"/console_log.txt
		gpu_is_intel=1
	fi
	pid=$!
	wait $pid #wait for timeout to finish
fi
if ! command -v radeontop &> /dev/null
then
	echo "radeontop not found" | tee -a ./"$folder_name"/console_log.txt
	gpu_is_amd=1
else
	echo "radeontop installed, searching for amd gpu" | tee -a ./"$folder_name"/console_log.txt
	#-d - means no graphical mode, output lines to stdout; there is no option 
	#-l 1 means only dump one line
	if radeontop -d - -l 1 | grep sclk; then
		echo "AMD GPU found" | tee -a ./"$folder_name"/console_log.txt
		gpu_is_amd=2
	else
		echo "no AMD GPU found" | tee -a ./"$folder_name"/console_log.txt
		gpu_is_amd=1
	fi
fi
if ! command -v nvidia-smi &> /dev/null
then
	echo "nvidia-smi not found" | tee -a ./"$folder_name"/console_log.txt
	gpu_is_nvidia=1
else
	echo "nvidia-smi installed, searching for nvidia gpu" | tee -a ./"$folder_name"/console_log.txt
	if nvidia-smi | grep Memory-Usage; then
		echo "Nvidia GPU found" | tee -a ./"$folder_name"/console_log.txt
		echo "printing nvidia GPU information to file" | tee -a ./"$folder_name"/console_log.txt
		nvidia-smi -f ./"$folder_name"/nvidia_gpu_information.csv --query-gpu=name,driver_version,memory.total,pcie.link.gen.max --format=csv,nounits
		gpu_is_nvidia=2
	else
		echo "no Nvidia GPU found" | tee -a ./"$folder_name"/console_log.txt
		gpu_is_nvidia=1
	fi
fi
if ! command -v tegrastats &> /dev/null
then
	echo "tegrastats not found" | tee -a ./"$folder_name"/console_log.txt
	system_is_tegra=1
else
	echo "tegrastats installed, assuming jetson hardware" | tee -a ./"$folder_name"/console_log.txt
	system_is_tegra=2
#	if timeout 2 tegrastats --interval 1000 | grep thermal; then
#		echo "Tegra data found" | tee -a ./"$folder_name"/console_log.txt
#		system_is_tegra=2
#	else
#		echo "no Tegra data found" | tee -a ./"$folder_name"/console_log.txt
#		system_is_tegra=1
#	fi
fi
#vcgencmd is for Raspberry Pi
if ! command -v vcgencmd &> /dev/null
then
	echo "vcgencmd not found" | tee -a ./"$folder_name"/console_log.txt
	system_is_rpi=1
else
	echo "vcgencmd installed, searching for rpi data" | tee -a ./"$folder_name"/console_log.txt
	if vcgencmd measure_temp | grep temp; then
		echo "rpi data found" | tee -a ./"$folder_name"/console_log.txt
		system_is_rpi=2
	else
		echo "no rpi data found" | tee -a ./"$folder_name"/console_log.txt
		system_is_rpi=1
	fi
fi
}

check_for_gpu
#output GPU data to files
if (( $gpu_is_amd==2 )); then
	timeout $timeout_time radeontop -d ./"$folder_name"/amd_gpu_log.txt -i 1 & #-d - means print to console; -l 1 means only dump one line
fi
if (( $gpu_is_intel==2 )); then
	timeout $timeout_time intel_gpu_top -o ./"$folder_name"/intel_gpu_log.txt -s 1000 &
fi
if (( $gpu_is_nvidia==2 )); then
	timeout $timeout_time nvidia-smi -f ./"$folder_name"/nvidia_gpu_log.csv -lms 1000 --query-gpu=timestamp,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,clocks.sm,clocks.mem,clocks.gr,pstate --format=csv,nounits &
fi
#nvidia jetson (or other system with nvidia tegra)
if (( $system_is_tegra==2 )); then
	#log nvpmodel to file
	nvpmodel --query >> ./"$folder_name"/tegra_nvpmodel.txt
	#log tegrastats to file
	timeout $timeout_time tegrastats --logfile ./"$folder_name"/tegra_stats.txt --interval 1000 &
fi

#check if top command does exist (should exist on all linux systems, but not windows)
if ! command -v top &> /dev/null
then
	echo "top not installed" | tee -a ./"$folder_name"/console_log.txt
else
	echo "top installed" | tee -a ./"$folder_name"/console_log.txt
	top_installed=1
fi
#check if directory for clock speeds exists
if [ -d "/sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq" ]
then
    echo "scaling_cur_freq exists" | tee -a ./"$folder_name"/console_log.txt
	scaling_cur_freq_exists=1
else
    echo "scaling_cur_freq does not exist" | tee -a ./"$folder_name"/console_log.txt
fi
#check if lm-sensors is installed to read CPU temp
if ! command -v sensors &> /dev/null
then
	echo "lm-sensors not installed" | tee -a ./"$folder_name"/console_log.txt
else
	echo "lm-sensors installed" | tee -a ./"$folder_name"/console_log.txt
	if (( $log_sensors==1 )); then
		echo "log_sensors true" | tee -a ./"$folder_name"/console_log.txt
		lm_sensors_installed=1
	else
		echo "log_sensors false" | tee -a ./"$folder_name"/console_log.txt
	fi
fi
#check if iotop is installed to read disk usage
if ! command -v iotop &> /dev/null
then
	echo "iotop not installed" | tee -a ./"$folder_name"/console_log.txt
else
	echo "iotop installed" | tee -a ./"$folder_name"/console_log.txt
	iotop_installed=1
fi
#check if top -1 is supported
if top -b -n 1 -1 | grep Cpu
then 
	echo "top -1 supported" | tee -a ./"$folder_name"/console_log.txt
	top_1_supported=1
else
	echo "top -1 not supported" | tee -a ./"$folder_name"/console_log.txt
	echo "run top command; press 1; then press W and then Enter to switch to per-core-usage" | tee -a ./"$folder_name"/console_log.txt
fi
#output CPU data to file
grep -m 1 'model name' /proc/cpuinfo >> ./"$folder_name"/cpu_name.txt 
core_count=$( cat /proc/cpuinfo | grep processor | wc -l )
echo "core_count ${core_count}" >> ./"$folder_name"/cpu_name.txt
echo "starting CPU and disk logging" | tee -a ./"$folder_name"/console_log.txt

#create timeout for loop, sleep would wait for over 2 days, but gets cancelled after logging time ends by timeout
timeout $timeout_time sleep 200000 & timeout_pid=$!
while ps -p $timeout_pid > /dev/null #check if timeout has cancelled sleep
do
	#log data every 1 second
	sleep 1 &
	pid=$!
	if (( $top_installed==1 )); then #top does not exist under windows
        if (( $top_1_supported==1 )); then
            echo `top -b -1 -n 1 | grep 'Cpu\|MiB \Mem'` >> ./"$folder_name"/cpu_top.txt 
        else
            echo `top -b -n 1 | grep 'Cpu\|MiB \Mem'` >> ./"$folder_name"/cpu_top.txt 
        fi
    fi
    if (( $scaling_cur_freq_exists==1 )); then
		echo `cat /sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq` >> ./"$folder_name"/cpu_freq.txt 
	fi
    if (( $lm_sensors_installed==1 )); then
		#sensors | grep CPU: >> ./"$folder_name"/cpu_lmsensors.txt
		sensors -u --no-adapter >> ./"$folder_name"/lmsensors.txt
	fi
	if (( $iotop_installed==1 )); then
		iotop -n 1 -P -b -o | grep Total >>  ./"$folder_name"/disk_iotop.txt
	fi
	#raspberry pi
	if (( $system_is_rpi==2 )); then
		echo "temp:$(vcgencmd measure_temp) arm_clock:$(vcgencmd measure_clock arm) core_clock:$(vcgencmd measure_clock core) v3d_clock:$(vcgencmd measure_clock v3d)" >> ./"$folder_name"/rpi_vcgencmd.txt 
	fi
	wait $pid #wait for sleep to finish
done

echo "Logging finished, Imma head out"


