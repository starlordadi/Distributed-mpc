import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class DMPC(object):
	"""docstring for DMPC"""
	def __init__(self, n_agents, n, d, horizon):
		super(DMPC, self).__init__()
		self.n_agents = n_agents
		self.n = n
		self.d = d
		self.horizon = horizon

	def set_params(self):
		# rho
		self.rho = 0.01

		# gamma_i
		self.gamma_z_list = []
		self.gamma_u_list = []

		for i in range(self.n_agents):
			self.gamma_z_list.append(np.zeros(self.n*(self.horizon+1)*self.n_agents).reshape([-1,1]))
			self.gamma_u_list.append(np.zeros(self.d*self.horizon*self.n_agents).reshape([-1,1]))

		# network_average
		self.z_average = np.zeros(self.n*(self.horizon+1)*self.n_agents).reshape([-1,1])
		self.u_average = np.zeros(self.d*self.horizon*self.n_agents).reshape([-1,1]) 


	def solve(self, agent_list, initial_guess):
		# initialize network average and multipliers
		self.set_params()
		constraint_val = 10e7
		k = 0 
		
		# while not converged
		while constraint_val >= 0.05 and k <= 30:
			# print(k)	
			z_avg = []
			u_avg = []
			z0 = np.vstack([agent.current_state for agent in agent_list])
			
			# loop over/parallel compute in all agents to solve local problem
			for i in range(self.n_agents):
				agent = agent_list[i]
				if initial_guess["condition"] == True:				
					agent.solve(z0, self.n_agents, self.z_average, self.u_average, self.gamma_z_list[i], self.gamma_u_list[i], self.rho, guess=initial_guess["value"])
				else:
					agent.solve(z0, self.n_agents, self.z_average, self.u_average, self.gamma_z_list[i], self.gamma_u_list[i], self.rho, guess=self.z_average)
				z_avg.append(agent.z_stack)
				u_avg.append(agent.u_stack)				

			# average local copies from all agents
			self.z_average = np.mean(z_avg, axis=0)
			self.u_average = np.mean(u_avg, axis=0)
					
			# update network average and multiplier
			constraint_val = 0
			for i in range(self.n_agents):
				self.gamma_z_list[i] = self.gamma_z_list[i] + self.rho*(agent_list[i].z_stack - self.z_average)
				self.gamma_u_list[i] = self.gamma_u_list[i] + self.rho*(agent_list[i].u_stack - self.u_average)
				constraint_val = constraint_val + np.mean(abs(agent_list[i].z_stack - self.z_average))
				constraint_val = constraint_val + np.mean(abs(agent_list[i].u_stack - self.u_average))
			
			# print(constraint_val)
			k = k+1

		# plot optimal solution

		# color = ['red', 'blue', 'green', 'black']

		# plt.cla()
		# ax = plt.axes(projection='3d')
		# ax.set_xlim([-5, 5])
		# ax.set_ylim([-5, 5])
		# ax.set_zlim([-5, 5])

		# for i in range(self.n_agents):
		# 	for j in range(self.horizon+1):
		# 		ax.plot3D(self.z_average[(i*(self.horizon+1)*self.n)+(j*self.n)],self.z_average[(i*(self.horizon+1)*self.n)+(j*self.n)+1],self.z_average[(i*(self.horizon+1)*self.n)+(j*self.n)+2], marker='.', color = color[i])

		# plt.pause(0.01)

		