import numpy as np
import casadi as ca

dT = 0.1

class Agent(object):
	"""docstring for Agent"""
	def __init__(self, current_state, goal_state, index):
		super(Agent, self).__init__()
		self.current_state = current_state
		self.goal_state = goal_state
		self.index = index
		# self.n = 6
		self.n = 4
		# self.d = 3
		self.d = 2
		self.horizon = 10
		self.Q = 0.1*np.eye(self.n)

		self.A = np.matrix([[0, 0, 1, 0],
						   [0, 0, 0, 1],
						   [0, 0, 0, 0],
						   [0, 0, 0, 0]])

		# self.A = np.matrix([[0, 0, 0, 1, 0, 0],
		# 				   [0, 0, 0, 0, 1, 0],
		# 				   [0, 0, 0, 0, 0, 1],
		# 				   [0, 0, 0, 0, 0, 0],
		# 				   [0, 0, 0, 0, 0, 0],
		# 				   [0, 0, 0, 0, 0, 0]])


		self.B = np.matrix([[0, 0],
						   [0, 0],
						   [1, 0],
						   [0, 1]])

		# self.B = np.matrix([[0, 0, 0],
		# 				   [0, 0, 0],
		# 				   [0, 0, 0],
		# 				   [1, 0, 0],
		# 				   [0, 1, 0],
		# 				   [0, 0, 1]])



		self.Ad = (self.A * dT) + np.eye(self.n)
		self.Bd = (self.B * dT)
		self.cost = 0


	def update_state(self, u):
		self.current_state = self.Ad@self.current_state + self.Bd@u + 0.01*np.random.rand(self.n).reshape([-1,1])
		# self.current_state = self.current_state + dT*np.array([u[0,0]*np.cos(self.current_state[2,0]), u[0,0]*np.sin(self.current_state[2,0]), u[1,0]]).reshape([-1,1])

	def dynamics(self, z, u):
		return self.Ad@z + self.Bd@u
		# return z + dT*np.array([u[0,0]*np.cos(z[2,0]), u[0,0]*np.sin(z[2,0]), u[1,0]]).reshape([-1,1])

	def solve(self, z0, n_agents, z_average, u_average, gamma_z, gamma_u, rho, guess):
		# Define Variables
		zi = ca.SX.sym('z', self.n*(self.horizon+1)*n_agents)
		ui = ca.SX.sym('u', self.d*self.horizon*n_agents)
		lam = ca.SX.sym('lam', self.horizon*(n_agents-1)*4)
		# lam = ca.SX.sym('lam', self.horizon*(n_agents-1)*6)
		s = ca.SX.sym('s', self.horizon*(n_agents-1))
		slack = ca.SX.sym('slack', self.horizon*(n_agents-1))
		# slack_mult = ca.SX.sym('slack_mult', self.horizon*(n_agents-1))
		
		
		# Get Cost Function
		agent_cost = 0
		network_cost = 0
		for i in range(self.horizon+1):
			agent_cost = agent_cost + (zi[self.index*(self.n)*(self.horizon+1)+self.n*i:self.index*(self.n)*(self.horizon+1)+self.n*(i+1)]-self.goal_state).T @ self.Q @ (zi[self.index*(self.n)*(self.horizon+1)+self.n*i:self.index*(self.n)*(self.horizon+1)+self.n*(i+1)]-self.goal_state)
		network_cost = network_cost + gamma_z.T @ (zi-z_average)
		network_cost = network_cost + gamma_u.T @ (ui-u_average)
		network_cost = network_cost + (rho/2)*(ca.sumsqr(zi-z_average)+ca.sumsqr(ui-u_average))
		cost = agent_cost + network_cost + 10e7*ca.sum1(s)
		# cost = agent_cost + network_cost

		# Get Constraints
		constraints = []
		# Dynamics
		for j in range(n_agents):
			for i in range(self.horizon):
				z_next = self.dynamics(zi[(j*(self.horizon+1)*self.n)+(i*self.n):(j*(self.horizon+1)*self.n)+((i+1)*self.n)],ui[(j*(self.horizon)*self.d)+(i*self.d):(j*(self.horizon)*self.d)+((i+1)*self.d)])
				for k in range(self.n):
					constraints = ca.vertcat(constraints, zi[(j*(self.horizon+1)*self.n)+((i+1)*self.n)+k]-z_next[k,0])

		# Obstacle
		A = np.vstack([np.eye(2), -np.eye(2)])
		# A = np.vstack([np.eye(3), -np.eye(3)])
		delz = 0.3*np.ones(4).reshape([-1,1])
		# delz = 0.5*np.ones(6).reshape([-1,1])
		m = np.matrix([[1,0,0,0],[0,1,0,0]])
		# m = np.matrix([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])
		# m = np.matrix([[1,0,0],[0,1,0]])

		k = 0
		for i in range(n_agents):
			if i == self.index:
				k = k+1
				continue 
			for j in range(self.horizon):
				pi = m @ zi[((self.index)*(self.horizon+1)*self.n)+(j+1)*(self.n):((self.index)*(self.horizon+1)*self.n)+(j+2)*self.n]
				pj = m @ zi[((i)*(self.horizon+1)*self.n)+(j+1)*(self.n):((i)*(self.horizon+1)*self.n)+(j+2)*(self.n)]
				dual_val = (A @ (pi-pj) - delz).T @ lam[((i-k)*4*self.horizon)+(j*4):((i-k)*4*self.horizon)+((j+1)*4)]
				# dual_val = (A @ (pi-pj) - delz).T @ lam[((i-k)*6*self.horizon)+(j*6):((i-k)*6*self.horizon)+((j+1)*6)]
				constraints = ca.vertcat(constraints, dual_val+s[((i-k)*self.horizon)+j]- slack[((i-k)*self.horizon)+j])
				# constraints = ca.vertcat(constraints, dual_val - slack[((i-k)*self.horizon)+j])
				mult_val = ca.sumsqr(A.T @ lam[((i-k)*4*self.horizon)+(j*4):((i-k)*4*self.horizon)+((j+1)*4)])
				# mult_val = ca.sumsqr(A.T @ lam[((i-k)*6*self.horizon)+(j*6):((i-k)*6*self.horizon)+((j+1)*6)])
				constraints = ca.vertcat(constraints, mult_val-1)
				# constraints = ca.vertcat(constraints, mult_val-1+slack_mult[((i-k)*self.horizon)+j])

		# lbg and ubg
		lbg = [0]*n_agents*self.horizon*self.n + [0]*2*(n_agents-1)*self.horizon
		ubg = [0]*n_agents*self.horizon*self.n + [0]*2*(n_agents-1)*self.horizon

		# Create NLP solver
		opts = {'verbose':False, 'ipopt.print_level':0, 'print_time':0}
		nlp = {'x':ca.vertcat(zi, ui, lam, s, slack), 'f':cost, 'g':constraints}
		# nlp = {'x':ca.vertcat(zi, ui, lam, s, slack, slack_mult), 'f':cost, 'g':constraints}
		solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

		# Solve
		z_aug_l = [-100]*self.n*(self.horizon+1)*n_agents
		z_aug_u = [100]*self.n*(self.horizon+1)*n_agents

		for i in range(n_agents):
			z_aug_u[(i*(self.horizon+1)*self.n):(i*(self.horizon+1)*self.n)+self.n] = z0[i*self.n:(i+1)*self.n]
			z_aug_l[(i*(self.horizon+1)*self.n):(i*(self.horizon+1)*self.n)+self.n] = z0[i*self.n:(i+1)*self.n]

		lbx = z_aug_l + [-1]*self.d*self.horizon*n_agents + [0]*(self.horizon*(n_agents-1)*4 + self.horizon*(n_agents-1) + self.horizon*(n_agents-1))
		ubx = z_aug_u + [1]*self.d*self.horizon*n_agents + [100]*(self.horizon*(n_agents-1)*4 + self.horizon*(n_agents-1) + self.horizon*(n_agents-1))

		# lbx = z_aug_l + [-1]*self.d*self.horizon*n_agents + [0]*(self.horizon*(n_agents-1)*6 + self.horizon*(n_agents-1) + self.horizon*(n_agents-1))
		# ubx = z_aug_u + [1]*self.d*self.horizon*n_agents + [100]*(self.horizon*(n_agents-1)*6 + self.horizon*(n_agents-1) + self.horizon*(n_agents-1))

		# lbx = z_aug_l + [-1]*self.d*self.horizon*n_agents + [0]*(self.horizon*(n_agents-1)*4 + self.horizon*(n_agents-1) + 2*self.horizon*(n_agents-1))
		# ubx = z_aug_u + [1]*self.d*self.horizon*n_agents + [100]*(self.horizon*(n_agents-1)*4 + self.horizon*(n_agents-1) + 2*self.horizon*(n_agents-1))

		# get guess value
		x0 = guess.reshape(-1).tolist() + (self.d*self.horizon*n_agents+self.horizon*(n_agents-1)*4+self.horizon*(n_agents-1)+self.horizon*(n_agents-1))*[0]
		# x0 = guess.reshape(-1).tolist() + (self.d*self.horizon*n_agents+self.horizon*(n_agents-1)*6+self.horizon*(n_agents-1)+self.horizon*(n_agents-1))*[0]
		# x0 = guess.reshape(-1).tolist() + (self.d*self.horizon*n_agents+self.horizon*(n_agents-1)*4+self.horizon*(n_agents-1)+2*self.horizon*(n_agents-1))*[0]

		sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=x0)
		sol_array = np.array(sol['x'])
		# self.cost = (zi-self.goal_state).T @ self.Q @ (zi-self.goal_state)
		self.z_stack = sol_array[:self.n*(self.horizon+1)*n_agents].reshape([-1,1])
		self.u_stack = sol_array[self.n*(self.horizon+1)*n_agents:self.n*(self.horizon+1)*n_agents+self.d*self.horizon*n_agents].reshape([-1,1])
