import numpy as np
import matplotlib.pyplot as plt
import code

class Lineage():
	'''Class for containing and managing a sequence of generations'''
	def __init__(self, gen_size=20, network_shape=[1]):
		self.gen_size = gen_size
		self.network_shape = network_shape
		self.current_gen = Generation(gen_size, network_shape)
		self.best_max_gen = self.current_gen.copy()
		self.best_avg_gen = self.current_gen.copy()
		self.best_min_gen = self.current_gen.copy()
		self.avg_list = []
		self.min_list = []
		self.max_list = []
		self.num_gens = 0

	def plot(self, title='', xmin=None, xmax=None, ymin=None, ymax=None):
		'''Method for plotting the learning curve of the lineage'''
		plot = plt.figure()
		plt.title(title)
		plt.plot(self.min_list, 'b-')
		plt.plot(self.max_list, 'r-')
		plt.plot(self.avg_list, 'g-')
		plt.axhline(np.max(self.max_list), color='r')
		if(xmin != None):
			plt.xlim(xmin=xmin)
		if(xmax != None):
			plt.xlim(xmax=xmax)
		if(ymin != None):
			plt.ylim(ymin=ymin)
		if(ymax != None):
			plt.ylim(ymax=ymax)
		plt.show()
		plt.close(plot)

	def nextGeneration(self, mutation_rate=0.05, breed_method='tournament'):
		'''Method for advancing to the next generation'''
		next_gen = self.current_gen.breed(mutation_rate, breed_method)
		self.avg_list.append(self.current_gen.avg_fitness)
		self.min_list.append(self.current_gen.min_fitness)
		self.max_list.append(self.current_gen.max_fitness)
		self.num_gens = self.num_gens + 1

		if(self.current_gen.max_fitness > self.best_max_gen.max_fitness):
			self.best_max_gen = self.current_gen.copy()

		if(self.current_gen.avg_fitness > self.best_avg_gen.avg_fitness):
			self.best_avg_gen = self.current_gen.copy()

		if(self.current_gen.min_fitness > self.best_min_gen.min_fitness):
			self.best_min_gen = self.current_gen.copy()

		self.current_gen = next_gen

class Generation():
	def __init__(self, gen_size=20, network_shape=[1]):
		self.ranked = False
		self.network_shape = network_shape
		self.size = gen_size
		self.n_layers = len(network_shape)
		self.avg_fitness = 0
		self.min_fitness = 0
		self.max_fitness = 0
		self.genome = [None]*gen_size
		for i in range(self.size):
			self.genome[i] = Genome(network_shape)
			self.genome[i].randomize()

	def breed(self, mutation_rate=0.05, breed_method='tournament'):
		self.rank()
		new_generation = self.blankCopy()
		for i in range(new_generation.size):
			if(breed_method is 'tournament'):
				new_generation.genome[i] = self.genome[0].splice(self.genome[1])
			elif(breed_method is 'alpha'):
				if(i == 0):
					new_generation.genome[i] = self.genome[0].copy()
				else:
					new_generation.genome[i] = self.genome[0].splice(self.genome[i])
			elif(breed_method is 'dominance'):
				draw = i
				while(draw == i):
					draw = np.random.randint(0, new_generation.size)
				if(self.genome[draw].fitness+self.genome[i].fitness != 0):
					mate_dominance = self.genome[draw].fitness/(self.genome[draw].fitness+self.genome[i].fitness)
				else:
					mate_dominance = 0.5
				new_generation.genome[i] = self.genome[i].splice(self.genome[draw], mate_dominance)
			else:
				print('Invalid breed method')
			new_generation.genome[i].mutation(mutation_rate)

		return new_generation

	def eval(self, x):
		if(type(x) is np.ndarray and self.network_shape[0] == x.shape[1] and x.ndim == 2):
			h = np.array([self.genome[0].eval(x)])
			for i in range(1, self.size):
				h = np.concatenate((h, np.array([self.genome[i].eval(x)])), axis=0)
			return h
		else:
			print('Input must be a vector with the same size as the input layer')

	def updateFitness(self, new_fitness):
		if(len(new_fitness) == self.size):
			for i in range(self.size):
				self.genome[i].updateFitness(new_fitness[i])

	def rank(self, invert=False):
		if(not self.ranked):
			self.avg_fitness = self.genome[0].fitness
			self.min_fitness = self.genome[0].fitness
			self.max_fitness = self.genome[0].fitness
			for i in range(1, self.size):
				self.avg_fitness += self.genome[i].fitness
				if(self.genome[i].fitness < self.min_fitness):
					self.min_fitness = self.genome[i].fitness
				if(self.genome[i].fitness > self.max_fitness):
					self.max_fitness = self.genome[i].fitness
			self.avg_fitness = self.avg_fitness/self.size

			one_more = True
			while(one_more):
				one_more = False
				for i in range(self.size-1):
					if(self.genome[i].fitness < self.genome[i+1].fitness):
						one_more = True
						swap = self.genome[i]
						self.genome[i] = self.genome[i+1]
						self.genome[i+1] = swap

			self.ranked = True

	def copy(self):
		copy = Generation(self.size, self.network_shape)
		copy.ranked = self.ranked
		copy.avg_fitness = self.avg_fitness
		copy.min_fitness = self.min_fitness
		copy.max_fitness = self.max_fitness
		for i in range(copy.size):
			copy.genome[i] = self.genome[i].copy()
		return copy

	def blankCopy(self):
		copy = Generation(self.size, self.network_shape)
		for i in range(copy.size):
			copy.genome[i] = self.genome[i].blankCopy()
		return copy

class Genome():
	def __init__(self, network_shape=[1]):
		self.network_shape = network_shape
		self.fitness = 0
		self.n_layers = len(network_shape)
		self.Theta = [None]*(self.n_layers-1)

		if(self.n_layers > 1):
			self.Theta[0] = np.zeros([network_shape[1], network_shape[0]+1])

		for i in range(1, self.n_layers-1):
			self.Theta[i] = np.zeros([network_shape[i+1], network_shape[i]])

	def reshape(self, new_network_shape):
		self.network_shape = new_network_shape
		self.fitness = 0
		self.n_layers = len(self.network_shape)
		self.Theta = [None]*(self.n_layers-1)
		for i in range(self.n_layers-1):
			self.Theta[i] = np.zeros([self.network_shape[i+1], self.network_shape[i]])

	def randomize(self, sigma=1):
		self.fitness = 0
		for i in range(self.n_layers-1):
			self.Theta[i] = sigma*np.random.randn(self.Theta[i].shape[0], self.Theta[i].shape[1])

	def eval(self, x):
		if(type(x) is np.ndarray and self.network_shape[0] == x.shape[1] and x.ndim == 2):
			x = np.transpose(np.insert(x, 0, np.ones(x.shape[0]), axis=1))
			h = x
			for i in range(self.n_layers-1):
				h = sigmoid(np.dot(self.Theta[i],h))
			return np.transpose(h)
		else:
			print('Input must be a numpy array with the same number os columns as the input layer and have 2 dimensions')

	def splice(self, mate, mate_dominance=0.5):
		if(self.network_shape == mate.network_shape):
			child = self.blankCopy()
			child.fitness = ((1-mate_dominance)*self.fitness+mate_dominance*mate.fitness)
			for k in range(child.n_layers-1):
				draw = np.greater(np.random.random(child.Theta[k].shape), mate_dominance)
				child.Theta[k] = draw*self.Theta[k]+np.logical_not(draw)*mate.Theta[k]
			return child
		else:
			print('Incompatible Mate')

	def mutation(self, mutation_rate=0.01, mutation_type='Gaussian'):
		if(mutation_rate >= 0 and mutation_rate <= 1):
			for k in range(self.n_layers-1):
				draw = np.less(np.random.random(self.Theta[k].shape), mutation_rate)
				if(mutation_type == 'Gaussian'):
					self.Theta[k] += draw*np.random.randn(self.Theta[k].shape[0], self.Theta[k].shape[1])
				elif(mutation_type == 'Uniform'):
					self.Theta[k] += draw*np.random.uniform(size=self.Theta[k].shape, low=-1.0, high=1.0)
		else:
			print('Invalid Mutation Rate')

	def updateFitness(self, new_fitness):
		if(new_fitness < 0):
			new_fitness = 0
		self.fitness = new_fitness

	def copy(self):
		copy = Genome(self.network_shape)
		copy.fitness = self.fitness
		for i in range(copy.n_layers-1):
			copy.Theta[i] = np.copy(self.Theta[i])
		return copy

	def blankCopy(self):
		copy = Genome(self.network_shape)
		return copy

def sigmoid(num):
	sig = 1/(1+np.exp(-num))
	return sig
