import multiprocessing as mp
import time


class VanillaMonteCarloTesting:
    def __init__(self, configurations, init_params, simulation_class, stop_criteria):
        self.configurations = configurations
        self.init_params = init_params
        self.simulation_class = simulation_class
        self.stop_criteria = stop_criteria

    def run_testing(self, n_threads=1, t_limit=60):
        """
        Run all the configurations in parallel.
        t_limit is the time limit for each configuration.
        For each config, create a new instance of simulation_class(config, init_params)
        Then run repeatedly_run_simulation(t_limit) and record the results.
        """
        start_time = time.time()
        self.t_limit = t_limit
        pool = mp.Pool(processes=n_threads)
        results = pool.map(self.run_single_config_one, self.configurations)
        pool.close()
        pool.join()
        print("Total time used: {} seconds".format(time.time() - start_time))
        return results
    
    def run_single_config_one(self, config):
        config_id = config[0]
        configuration = config[1]
        simulation = self.simulation_class(configuration, self.init_params, self.stop_criteria)
        return config_id, simulation.repeatedly_run_simulation(self.t_limit)
    


class VanillaMonteCarloTestingSingle:
    def __init__(self, configurations, init_params, simulation_class, stop_criteria):
        self.configurations = configurations
        self.init_params = init_params
        self.simulation_class = simulation_class
        self.stop_criteria = stop_criteria
    
    def run_testing(self, n_threads=1, t_limit=60):
        """
        The difference from VanillaMonteCarloTesting is that
        This class uses all threads for simuulating a single configuration in parallel while
        VanillaMonteCarloTesting uses each thread to simulate a different configurations.
        """
        start_time = time.time()
        self.t_limit = t_limit
        self.n_processes = n_threads
        results = []
        for config in self.configurations:
            res = self.run_single_config_one(config)
            results.append(res)
        print("Total time used: {} seconds".format(time.time() - start_time))
        return results
        
    def run_single_config_one(self, config):
        config_id = config[0]
        configuration = config[1]
        simulation = self.simulation_class(configuration, self.init_params, self.stop_criteria)
        return config_id, simulation.repeatedly_run_simulation_parallel(self.t_limit, self.n_processes)
            


