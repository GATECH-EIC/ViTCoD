
import math
class SRAM:
    def __init__(self):
        # SRAM
        # FIXME: too small
        # self.max_Q = 53 * 1024 * 8 # 53KB
        # self.max_K = 53 * 1024 * 8 # 53KB
        # self.max_V = 53 * 1024 * 8 # 53KB
        # self.max_index = 53 * 1024 * 8 # 53KB
        # self.max_output = 53 * 1024 * 8 # 53KB
        self.max_Q = 53 * 1024 * 8 # 53KB
        self.max_K = 53 * 1024 * 8 # 53KB
        self.max_V = 53 * 1024 * 8 # 53KB
        self.max_index = 20 * 1024 * 8 # 20KB
        self.max_output = 108 * 1024 * 8 # 108KB

        # HBM to SRAM
        self.bandwidth = 76.8 * 1024 * 1024 * 1024 * 8 # 76.8GB/s
        self.clock_frequency = 500 * 1e6 # 500MHz

    
    def preload_decoder(self, nums=0, bits=32, bandwidth_ratio=1):
        if nums * bits > self.max_Q:
            print('Error: loading Q from DRAM to SRAM')
            # exit()
        else:
            latency = nums * bits / (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)

        return cycle

    def preload_encoder(self, nums=0, bits=32, bandwidth_ratio=1):
        if nums * bits > self.max_Q:
            print('Error: loading Q from DRAM to SRAM')
            # exit()
        else:
            latency = nums * bits / (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)

        return cycle
    
    def preload_Q(self, nums=0, bits=32, bandwidth_ratio=1):
        if nums * bits > self.max_Q:
            print('Error: loading Q from DRAM to SRAM')
            # exit()
        else:
            latency = nums * bits / (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)

        return cycle
    
    def data_cycle(self, num, bit=8, bandwidth_ratio=1):
        latency = (num*bit) / (self.bandwidth * bandwidth_ratio)
        cycle = math.ceil(latency * self.clock_frequency)
        return cycle

    def preload_K(self, nums=0, bits=32, bandwidth_ratio=1):
        if nums * bits > self.max_K:
            print('Error: loading K from DRAM to SRAM')
            # exit()
        else:
            latency = nums * bits / (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)

        return cycle

    def preload_V(self, nums=0, bits=32, bandwidth_ratio=1):
        if nums * bits > self.max_V:
            print('Error: loading V from DRAM to SRAM')
            # exit()
        else:
            latency = nums * bits / (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)

        return cycle

    def preload_index(self, nums=0, bits=32, bandwidth_ratio=1):
        if nums * bits > self.max_index:
            print('Error: loading index from DRAM to SRAM')
            # exit()
        else:
            latency = nums * bits / (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)

        return cycle

    def store_out(self, nums=0, bits=32, bandwidth_ratio=1):
        if nums * bits > self.max_output:
            print('Error: storing back intermediate results from PE to SRAM')
            # exit()
        else:
            latency = nums * bits / (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)

        return cycle

    
    def preload_weight(self, nums=0, bits=32, bandwidth_ratio=1):
        if nums * bits > self.max_output:
            print('Error: storing back intermediate results from PE to SRAM')
            # exit()
        else:
            latency = nums * bits / (self.bandwidth * bandwidth_ratio)
            cycle = math.ceil(latency * self.clock_frequency)

        return cycle