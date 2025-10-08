import logging
import os
import time
import random
import numpy as np
from config import get_common_search_config
import torch
import torch.backends.cudnn as cudnn

from utils import *
from process_data.data_loader import *
from model.SuperNet import SuperNet
from EMO_public import P_generator, NDsort, F_distance, F_mating, F_EnvironmentSelect
import matplotlib.pyplot as plt

class Individual:
    """
    Represents an individual in the population with a specific architecture encoding.
    """
    def __init__(self, dec, length_info=None):
        self.length_info = length_info
        if len(dec)==len(self.length_info):
            self.dec = dec
        else:
            self.deal_IndividualDec_2_NASDec(dec)
        self.get_ndarryDec()
        self.fitness = np.array([0.0, 0.0, 0.0])

    def get_decF(self):
        # Calculate fitness based on the number of model parameters
        A = [12, 12, 16]
        B = [sum(x) for i, x in enumerate(self.dec) if i in (0, 1)] + [sum(sum(self.dec[4]))]
        return 1 - np.mean([y / x for x, y in zip(A, B)])

    def deal_IndividualDec_2_NASDec(self, individualDec):
        # Convert individual decision vector to architecture encoding
        dec = []
        start = 0
        for idx, item in enumerate(self.length_info):
            end = start + np.prod(item)
            dec.append(np.reshape(individualDec[start:end], newshape=item))
            start = end

        self.dec = []
        for idx, encoding in enumerate(dec):
            if idx == 0:
                encoding[0] = 1
            elif idx == 1:
                encoding[2] = 1
            elif idx == 4:
                for item_i in range(encoding.shape[0]):
                    if encoding[item_i].sum() == 0:
                        encoding[item_i, np.random.choice(encoding.shape[1], 1)] = 1
                if encoding[:, -1].sum() == 0:
                    encoding[np.random.choice(encoding.shape[0], 1), -1] = 1
            self.dec.append(encoding)

    def get_ndarryDec(self):
        ndarryDec = []
        for item in self.dec:
            ndarryDec.extend(np.array(item).reshape(-1,))
        self.ndarryDec = np.array(ndarryDec)

    def evaluation(self, setting):
        # Evaluate the individual's performance
        acc, auc, latency = evaluation(self.dec, setting)
        self.fitness = np.array([auc, 1 - acc, latency])


class EvolutionaryAlgorithm:
    """
    Evolutionary algorithm for neural architecture search.
    """
    def __init__(self, config):
        self.config = config
        self.load_super_model()
        self.load_validation_dataset()
        self.Popsize = 20
        self.get_Boundary()
        self.Maxi_Gen = 25
        self.gen = 0

    def load_super_model(self):
        # Load the supernet model
        self.model = SuperNet(self.config)
        state_dict = torch.load(self.config.pre_train_path)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.config.device)

    def load_validation_dataset(self):
        # Load the validation dataset
        fold_path = self.config.data_path[0]
        test_data = CTLSTMDataset(config=self.config, mode='test', fold_path=fold_path)
        self.test_dataloader = DataLoader(test_data, batch_size=self.config.batch_size*4, shuffle=False,
                                          drop_last=False, num_workers=self.config.num_workers, collate_fn=test_data.pad_batch_fn)

    def get_Boundary(self):
        # Define the search space boundaries
        Boundary_Up = [[1,1,1,1,1,1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1,1,1,1,1,1],
                    [2,2,4,2,2,4,2,2,4,2,2,4],
                    [2,2,4,2,2,4,2,2,4,2,2,4]]
        Boundary_Low = [[0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0]]

        self.dec_length_info = [np.array(item).shape for item in Boundary_Up]
        self.Boundary_Up = np.array([item for sublist in Boundary_Up for item in sublist])
        self.Boundary_Low = np.array([item for sublist in Boundary_Low for item in sublist])
        self.dec_length = sum([np.prod(x) for x in self.dec_length_info])

        self.SearchSpace = [np.linspace(j, i, i+1).tolist() for i, j in zip(self.Boundary_Up, self.Boundary_Low)]

    def population_initialization(self):
        # Initialize the population
        self.Population = []
        for i in range(0,self.Popsize):
            prob = (i+1)/(self.Popsize+1)
            dec_i = []
            for j,(up,low) in enumerate(zip(self.Boundary_Up,self.Boundary_Low)):
                # dec_i.extend(np.random.randint(low,up+1,1))

                if j<24:
                    dec_i.extend([int(np.random.rand()<0.1)])
                elif (j+1)%3==0:
                    if np.random.rand()<prob:
                        dec_i.extend([0])
                    else:
                        dec_i.extend(np.random.choice([1,2,3,4],1))
                else:
                    if np.random.rand()<prob:
                        dec_i.extend(np.random.choice([0,2],1))
                    else:
                        dec_i.extend([1])

            dec_i = np.array(dec_i)
            self.Population.append(Individual(dec_i,self.dec_length_info))


        self.Pop_fitness = self.Evaluation(self.Population)
        self.set_dir(path='initial')
        self.Save()

    def Evaluation(self, Population):
        # Evaluate the population
        Fitness = []
        for idx, individual in enumerate(Population):
            print(f'Evaluating solution {idx}: ')
            logging.info(f'Evaluating solution {idx}: ')
            individual.evaluation([self.model, self.test_dataloader])
            Fitness.append(individual.fitness)
        return np.array(Fitness)

    def MatingPoolSelection(self):
        self.MatingPool, self.tour_index = F_mating.F_mating(self.Population.copy(), self.FrontValue, self.CrowdDistance)

    def Genetic_operation(self):
        # Perform genetic operations (crossover and mutation) to generate offspring
        offspring_dec = P_generator.P_generator(self.MatingPool, Boundary=np.vstack([self.Boundary_Up.copy(), self.Boundary_Low.copy()]),
                                                Coding='Binary', MaxOffspring=self.Popsize, SearchSpace=self.SearchSpace)
        self.offspring = [Individual(x, self.dec_length_info) for x in offspring_dec]
        self.off_fitness = self.Evaluation(self.offspring)

    def SearchSpaceReduction(self, Population, Fitness):
        # Reduce the search space based on population statistics
        decs = np.array([indi.ndarryDec for indi in Population])

        self.spacefitness = []
        self.spacefitnessSTD = []
        self.spacefitnessMinimal = []
        self.spaceLength = []

        for idx, item in enumerate(self.SearchSpace):

            fitness = [np.mean(Fitness[decs[:, idx] == id][:, 0]) if np.sum(decs[:, idx] == id) > 0 else np.mean(Fitness[:, 0]) for id in item]
            self.spacefitness.append(fitness)
            self.spaceLength.append(len(item))
            self.spacefitnessSTD.append(np.std(fitness) if len(item) > 1 else 0)
            self.spacefitnessMinimal.append(np.max(fitness) if len(item) > 1 else 0)

        index_set = np.argmax(self.spacefitnessSTD)
        index_num = np.argmax(self.spacefitness[index_set])
        self.SearchSpace[index_set].pop(index_num)

        index_set_1 = np.argmax(self.spacefitnessMinimal)
        index_set_1 = np.argsort(self.spacefitnessMinimal)[-2] if index_set_1 == index_set else index_set_1
        index_num_1 = np.argmax(self.spacefitness[index_set_1])
        self.SearchSpace[index_set_1].pop(index_num_1)

    def EvironmentSelection(self):
        # Perform environmental selection to update the population
        Population = self.Population + self.offspring
        FunctionValue = np.vstack((self.Pop_fitness, self.off_fitness))

        self.SearchSpaceReduction(Population, FunctionValue)

        Population, FunctionValue, FrontValue, CrowdDistance, select_index = F_EnvironmentSelect.F_EnvironmentSelect(Population, FunctionValue, self.Popsize)

        self.Population = Population
        self.Pop_fitness = FunctionValue
        self.FrontValue = FrontValue
        self.CrowdDistance = CrowdDistance
        self.select_index = select_index

    def print_logs(self, since_time=None, initial=False):
        # Print logs during the evolutionary process
        if initial:
            logging.info('********************************************************************Initializing**********************************************')
            print('********************************************************************Initializing**********************************************')
        else:
            used_time = (time.time() - since_time) / 60
            logging.info(f'*******************************************************{self.gen + 1:>2d}/{self.Maxi_Gen:>2d} processing, time spent so far:{used_time:.2f} min***********************************************')
            print(f'*******************************************************{self.gen + 1:>2d}/{self.Maxi_Gen:>2d} processing, time spent so far:{used_time:.2f} min***********************************************')

    def set_dir(self, path=None):
        # Set the directory for saving results
        path = self.gen if path is None else path
        self.whole_path = f"{self.config.exp_name}/Gen_{path}/"
        os.makedirs(self.whole_path, exist_ok=True)

    def Save(self):
        # Save the population and search space information
        np.savetxt(self.whole_path + 'fitness.txt', self.Pop_fitness, delimiter=' ')
        with open(self.whole_path + 'Population.txt', "w") as file:
            for j, solution in enumerate(self.Population):
                file.write(f'solution {j}: {solution.fitness} \n {solution.dec} \n {solution.ndarryDec} \n')
        with open(self.whole_path + 'Space.txt', "w") as file:
            for j, solution in enumerate(self.SearchSpace):
                file.write(f' {j}: {solution} \n ')

    def Plot(self):
        # Plot the population fitness during the evolutionary process
        plt.clf()
        plt.plot(1 - self.Pop_fitness[:, 0], 1 - self.Pop_fitness[:, 1], 'o')
        plt.xlabel('ACC')
        plt.ylabel('AUC')
        plt.title(f'Generation {self.gen + 1}/{self.Maxi_Gen} \n best ACC: {max(1 - self.Pop_fitness[:, 0]):.4f}, best AUC: {max(1 - self.Pop_fitness[:, 1]):.4f}, best Latency: {max(self.Pop_fitness[:, 2]):.4f}')
        plt.pause(0.2)
        plt.savefig(self.whole_path + 'figure.jpg')

    def main_loop(self):
        # Main loop of the evolutionary algorithm
        plt.ion()
        since_time = time.time()
        self.print_logs(initial=True)
        self.population_initialization()
        self.Plot()

        self.FrontValue = NDsort.NDSort(self.Pop_fitness, self.Popsize)[0]

        self.CrowdDistance = F_distance.F_distance(self.Pop_fitness, self.FrontValue)

        while self.gen < self.Maxi_Gen:
            self.set_dir()
            self.print_logs(since_time=since_time)

            self.MatingPoolSelection()
            self.Genetic_operation()
            self.EvironmentSelection()

            self.Save()
            self.Plot()
            self.gen += 1
            
        best_individual, best_fitness = self.get_best_individual()

        with open(f"{self.config.exp_name}/best_configuration.txt", "w") as f:
            f.write(f"Best Individual Found:\n")
            f.write(f"Accuracy: {1 - best_fitness[1]:.4f}\n")
            f.write(f"AUC: {1 - best_fitness[0]:.4f}\n")
            f.write(f"Latency: {best_fitness[2]:.4f}\n")
            f.write("\nArchitecture Configuration:\n")
            f.write(str(best_individual.dec))
        plt.ioff()


    def get_best_individual(self):
        """
        获取最佳个体的配置
        Returns:
            best_individual: 最佳个体
            best_fitness: 最佳适应度值
        """
        # 获取所有个体的适应度值
        fitness_values = self.Pop_fitness
        
        # 使用非支配排序找到第一前沿面的个体
        front_values = NDsort.NDSort(fitness_values, len(self.Population))[0]
        first_front_indices = np.where(front_values == 1)[1]
        
        # 从第一前沿面中选择一个最佳个体
        # 这里可以根据您的需求选择不同的标准
        # 例如：选择AUC最高的
        best_idx = first_front_indices[np.argmax(1 - fitness_values[first_front_indices, 0])]
        
        best_individual = self.Population[best_idx]
        best_fitness = self.Pop_fitness[best_idx]
        
        # 打印最佳个体信息
        print(f"Best Individual Configuration:")
        print(f"Accuracy: {1 - best_fitness[1]:.4f}")
        print(f"AUC: {1 - best_fitness[0]:.4f}")
        print(f"Latency: {best_fitness[2]:.4f}")
        print("\nArchitecture Configuration:")
        print(best_individual.dec)
        
        return best_individual, best_fitness


def evaluation(NASdec, setting):
    # Evaluate the performance of a specific architecture encoding
    NASdec.append(np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]))
    model, val_dataloader = setting
    model.eval()
    test_str = 'Validation'

    start_time = time.time()
    output_dict_list = []
    epoch_val_loss = []

    with torch.no_grad():
        total = len(val_dataloader)
        for idx, item in enumerate(val_dataloader):
            output_dict = model.forward(item, NASdec)
            loss = model.loss(output_dict)
            print(f'\r              [{test_str} {idx + 1:>2d}/{total:>2d}, Loss: {loss:.5f}, used_time {time.time() - start_time:.2f}min({time.time() - start_time:.2f} s)]', end='')
            output_dict_list.append(output_dict)
            epoch_val_loss.append(loss.item())

    val_epoch_avg_loss = np.mean(epoch_val_loss)
    metrics = get_metrics(output_dict_list)
    epoch_acc = metrics['acc']
    epoch_auc = metrics['auc']
    latency = time.time() - start_time

    print_info = f" {test_str} loss: {val_epoch_avg_loss:.5f}, {test_str} time: {latency:.3f}s, metrics: {metrics}"
    print(print_info)
    logging.info(print_info)

    return epoch_acc, epoch_auc, latency


def main():
    # Main function to run the evolutionary algorithm
    config = get_common_search_config()
    config.device, config.device_ids = setup_device(config.n_gpu)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    EA = EvolutionaryAlgorithm(config)
    EA.main_loop()
    return None


if __name__ == '__main__':
    main()