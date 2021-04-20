import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
from dmpc import DMPC

def start():
	# plotting params
	# fig, ax = plt.subplots()
	# ax.set_xlim([-50,50])
	# ax.set_ylim([-50,50])

	# initialize agents
	agent_list = []
	# agent_list.append(Agent(current_state=np.array([2,2,np.pi]).reshape([-1,1]), goal_state=np.array([-4,-4,np.pi]).reshape([-1,1]), index=0))
	# agent_list.append(Agent(current_state=np.array([-2,2,0]).reshape([-1,1]), goal_state=np.array([4,-4,0]).reshape([-1,1]), index=1))
	# agent_list.append(Agent(current_state=np.array([-2,-2,0]).reshape([-1,1]), goal_state=np.array([4,4,0]).reshape([-1,1]), index=2))
	# agent_list.append(Agent(current_state=np.array([2,-2,np.pi]).reshape([-1,1]), goal_state=np.array([-4,4,np.pi]).reshape([-1,1]), index=3))
	# '''
	agent_list.append(Agent(current_state=np.array([-2,2,0,0]).reshape([-1,1]), goal_state=np.array([2,-2,0,0]).reshape([-1,1]), index=0))
	agent_list.append(Agent(current_state=np.array([2,-2,0,0]).reshape([-1,1]), goal_state=np.array([-2,2,0,0]).reshape([-1,1]), index=1))
	agent_list.append(Agent(current_state=np.array([2,2,0,0]).reshape([-1,1]), goal_state=np.array([-2,-2,0,0]).reshape([-1,1]), index=2))
	agent_list.append(Agent(current_state=np.array([0,-2,0,0]).reshape([-1,1]), goal_state=np.array([0,2,0,0]).reshape([-1,1]), index=3))

	# agent_list.append(Agent(current_state=np.array([2,2,2,0,0,0]).reshape([-1,1]), goal_state=np.array([-2,-2,2,0,0,0]).reshape([-1,1]), index=0))
	# agent_list.append(Agent(current_state=np.array([-2,-2,2,0,0,0]).reshape([-1,1]), goal_state=np.array([2,2,2,0,0,0]).reshape([-1,1]), index=1))
	# agent_list.append(Agent(current_state=np.array([2,-2,-2,0,0,0]).reshape([-1,1]), goal_state=np.array([-2,2,2,0,0,0]).reshape([-1,1]), index=2))
	# agent_list.append(Agent(current_state=np.array([-2,2,2,0,0,0]).reshape([-1,1]), goal_state=np.array([2,-2,-2,0,0,0]).reshape([-1,1]), index=3))

	# '''

	n_agents = len(agent_list)
	n = agent_list[0].n
	d = agent_list[0].d
	horizon = agent_list[0].horizon

	# initialize dmpc controller
	dmpc = DMPC(n_agents=n_agents, n=n, d=d, horizon=horizon)

	# compute initial guess solution
	# z0 = []
	# for i in range(n_agents):

	# q1 = ax.quiver(agent_list[0].current_state[0,0], agent_list[0].current_state[1,0], np.cos(agent_list[0].current_state[2,0]), np.sin(agent_list[0].current_state[2,0]),color='red')
	# q2 = ax.quiver(agent_list[1].current_state[0,0], agent_list[1].current_state[1,0], np.cos(agent_list[1].current_state[2,0]), np.sin(agent_list[1].current_state[2,0]),color='green')
	# q3 = ax.quiver(agent_list[2].current_state[0,0], agent_list[2].current_state[1,0], np.cos(agent_list[2].current_state[2,0]), np.sin(agent_list[2].current_state[2,0]),color='blue')
	# q4 = ax.quiver(agent_list[3].current_state[0,0], agent_list[3].current_state[1,0], np.cos(agent_list[3].current_state[2,0]), np.sin(agent_list[3].current_state[2,0]),color='black')

	# ax.plot(agent_list[0].current_state[0,0],agent_list[0].current_state[1,0], marker='s', color='red')
	# ax.plot(agent_list[1].current_state[0,0],agent_list[1].current_state[1,0], marker='s', color='blue')
	# ax.plot(agent_list[2].current_state[0,0],agent_list[2].current_state[1,0], marker='s', color='green')
	# ax.plot(agent_list[3].current_state[0,0],agent_list[3].current_state[1,0], marker='s', color='black')

	# plt.pause(0.01)

	# initial guess dict
	initial_guess = {"condition":False,
					 "value":None}

	# iter: timesteps or cost
	time = 0
	cl_data_z = [[] for i in range(n_agents)]
	cl_data_u = [[] for i in range(n_agents)]
	while time <= 50:
		# dmpc.solve
		dmpc.solve(agent_list, initial_guess=initial_guess)		

		# update agents
		for i in range(n_agents):
			agent_list[i].update_state(u=agent_list[i].u_stack[i*d*horizon:i*d*horizon+d])
			
			# store closed loop data
			cl_data_z[i].append(agent_list[i].current_state)
			cl_data_u[i].append(dmpc.u_average[i*d*n_agents:i*d*n_agents+d])

		time = time + 1
		print(time)

		# update initial guess
		initial_guess["condition"] = True
		initial_guess["value"] = dmpc.z_average

		# plot vectors
		# q1 = ax.quiver(agent_list[0].current_state[0,0], agent_list[0].current_state[1,0], np.cos(agent_list[0].current_state[2,0]), np.sin(agent_list[0].current_state[2,0]),color='red')
		# q2 = ax.quiver(agent_list[1].current_state[0,0], agent_list[1].current_state[1,0], np.cos(agent_list[1].current_state[2,0]), np.sin(agent_list[1].current_state[2,0]),color='green')
		# q3 = ax.quiver(agent_list[2].current_state[0,0], agent_list[2].current_state[1,0], np.cos(agent_list[2].current_state[2,0]), np.sin(agent_list[2].current_state[2,0]),color='blue')
		# q4 = ax.quiver(agent_list[3].current_state[0,0], agent_list[3].current_state[1,0], np.cos(agent_list[3].current_state[2,0]), np.sin(agent_list[3].current_state[2,0]),color='black')

		# ax.plot(agent_list[0].current_state[0,0], agent_list[0].current_state[1,0], 's',color='red')
		# ax.plot(agent_list[1].current_state[0,0], agent_list[1].current_state[1,0], 's',color='blue')
		# ax.plot(agent_list[2].current_state[0,0], agent_list[2].current_state[1,0], 's',color='green')
		# ax.plot(agent_list[3].current_state[0,0], agent_list[3].current_state[1,0], 's',color='black')		

		# plt.pause(0.01)
	plt.cla()
	# ax = plt.axes(projection="3d")
	# ax.set_xlim([-3,3])
	# ax.set_ylim([-3,3])
	# ax.set_zlim([-3,3])
	colors = ['red','blue','green','black']
	for i in range(n_agents):
		x = [state[0,0] for state in cl_data_z[i]]
		y = [state[1,0] for state in cl_data_z[i]]
		# z = [state[2,0] for state in cl_data_z[i]]
		# ax.plot3D(x,y,z,color=colors[i])
		plt.plot(x,y, color=colors[i])

	plt.show()


if __name__ == '__main__':
	start()