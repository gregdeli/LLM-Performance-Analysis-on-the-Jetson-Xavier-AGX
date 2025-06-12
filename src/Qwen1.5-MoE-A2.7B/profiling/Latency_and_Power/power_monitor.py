"""
PowerMonitor Utility for GPU and CPU Power Measurement

This module provides the PowerMonitor class, which enables background monitoring of GPU and CPU
power consumption by reading INA3221 sensor files from the system. It is designed for use in
profiling scripts that require accurate power measurements during model inference.

Requirements:
- This script requires sudo privileges to access the hardware monitoring files in /sys.
"""

from collections import deque
import threading


class PowerMonitor:
    def __init__(self):
        # Paths to INA3221 sensor files for current and voltage readings
        self.gpu_current_path = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon4/curr1_input"
        self.gpu_voltage_path = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon4/in1_input"
        self.cpu_current_path = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon4/curr2_input"
        self.cpu_voltage_path = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon4/in2_input"

        # Deque to store recent power readings (maxlen limits memory usage)
        self.data = deque(maxlen=10000)
        self.running = False
        self.thread = None

    def _read_power(self):
        """
        Reads instantaneous GPU and CPU power from sensor files.
        Returns:
            (gpu_power_mw, cpu_power_mw): Tuple of power values in milliwatts.
        """
        try:
            with open(self.gpu_current_path, "r") as f:
                gpu_current = int(f.read())
            with open(self.gpu_voltage_path, "r") as f:
                gpu_voltage = int(f.read())
            with open(self.cpu_current_path, "r") as f:
                cpu_current = int(f.read())
            with open(self.cpu_voltage_path, "r") as f:
                cpu_voltage = int(f.read())

            return (
                gpu_current * gpu_voltage / 1000,  # GPU power in mW
                cpu_current * cpu_voltage / 1000,  # CPU power in mW
            )
        except Exception as e:
            return (0, 0)

    def run(self):
        """
        Continuously read and store power values while monitoring is active.
        """
        print("Power monitor running...")
        while self.running:
            gpu_pwr, cpu_pwr = self._read_power()
            self.data.append((gpu_pwr, cpu_pwr))

    def start(self):
        """
        Start background power monitoring in a separate thread.
        """
        self.clear()
        self.running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def stop(self):
        """
        Stop background power monitoring and wait for the thread to finish.
        """
        print("Power monitor stopping...")
        self.running = False
        if self.thread is not None:
            self.thread.join()

    def clear(self):
        """
        Clear all stored power readings.
        """
        self.data.clear()

    def get_avg_gpu_power(self):
        """
        Compute the average GPU power (in mW) over the recorded samples.
        """
        power_data = list(self.data)
        gpu_power = [d[0] for d in power_data]
        return sum(gpu_power) / len(gpu_power) if len(gpu_power) > 0 else 0

    def get_avg_cpu_power(self):
        """
        Compute the average CPU power (in mW) over the recorded samples.
        """
        power_data = list(self.data)
        cpu_power = [d[1] for d in power_data]
        return sum(cpu_power) / len(cpu_power) if len(cpu_power) > 0 else 0


# Example usage:
# power_monitor = PowerMonitor()
# power_monitor.start()
# ... run workload ...
# power_monitor.stop()
# print(power_monitor.get_avg_gpu_power(), power_monitor.get_avg_cpu_power())
