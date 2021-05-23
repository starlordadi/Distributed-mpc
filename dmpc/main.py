import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
from dmpc import DMPC
import imageio
import os

def start():
	# plotting params
	# fig, ax = plt.subplots()
	# ax.set_xlim([-2,2])
	# ax.set_ylim([-2,2])

	# initialize agents
	agent_list = []
	# agent_list.append(Agent(current_state=np.array([2,2,np.pi]).reshape([-1,1]), goal_state=np.array([-4,-4,np.pi]).reshape([-1,1]), index=0))
	# agent_list.append(Agent(current_state=np.array([-2,2,0]).reshape([-1,1]), goal_state=np.array([4,-4,0]).reshape([-1,1]), index=1))
	# agent_list.append(Agent(current_state=np.array([-2,-2,0]).reshape([-1,1]), goal_state=np.array([4,4,0]).reshape([-1,1]), index=2))
	# agent_list.append(Agent(current_state=np.array([2,-2,np.pi]).reshape([-1,1]), goal_state=np.array([-4,4,np.pi]).reshape([-1,1]), index=3))
	# '''
	agent_list.append(Agent(current_state=np.array([-3,-2,0,0]).reshape([-1,1]), goal_state=np.array([1,3,0,0]).reshape([-1,1]), index=0))
	agent_list.append(Agent(current_state=np.array([-1,-2,0,0]).reshape([-1,1]), goal_state=np.array([-1,1,0,0]).reshape([-1,1]), index=1))
	agent_list.append(Agent(current_state=np.array([1,-2,0,0]).reshape([-1,1]), goal_state=np.array([1,1,0,0]).reshape([-1,1]), index=2))
	agent_list.append(Agent(current_state=np.array([3,-2,0,0]).reshape([-1,1]), goal_state=np.array([-1,3,0,0]).reshape([-1,1]), index=3))

	# agent_list.append(Agent(current_state=np.array([3,3,3,0,0,0]).reshape([-1,1]), goal_state=np.array([-3,-3,3,0,0,0]).reshape([-1,1]), index=0))
	# agent_list.append(Agent(current_state=np.array([-3,-3,3,0,0,0]).reshape([-1,1]), goal_state=np.array([3,3,3,0,0,0]).reshape([-1,1]), index=1))
	# agent_list.append(Agent(current_state=np.array([3,-3,-3,0,0,0]).reshape([-1,1]), goal_state=np.array([-3,3,3,0,0,0]).reshape([-1,1]), index=2))
	# agent_list.append(Agent(current_state=np.array([-3,3,3,0,0,0]).reshape([-1,1]), goal_state=np.array([3,-3,-3,0,0,0]).reshape([-1,1]), index=3))

	# agent_list.append(Agent(current_state=np.array([3,0,0,0,0,0]).reshape([-1,1]), goal_state=np.array([-3,0,3,0,0,0]).reshape([-1,1]), index=0))
	# agent_list.append(Agent(current_state=np.array([1,0,0,0,0,0]).reshape([-1,1]), goal_state=np.array([0,3,3,0,0,0]).reshape([-1,1]), index=1))
	# agent_list.append(Agent(current_state=np.array([-1,0,0,0,0,0]).reshape([-1,1]), goal_state=np.array([0,-3,3,0,0,0]).reshape([-1,1]), index=2))
	# agent_list.append(Agent(current_state=np.array([-3,0,0,0,0,0]).reshape([-1,1]), goal_state=np.array([3,0,3,0,0,0]).reshape([-1,1]), index=3))

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

	# plot initial state
	colors = ['red','blue','green','black']

	# plt.xlabel("x")
	# plt.ylabel("y")

	# plt.xlim([-4,4])
	# plt.ylim([-4,4])

	# for i in range(n_agents):
	# # 	label = "agent " + str(i+1)
	# 	plt.plot(agent_list[i].current_state[0,0], agent_list[i].current_state[1,0], marker='s', markersize=12, color=colors[i])

	# # for i in range(n_agents):
	# 	plt.plot(agent_list[i].goal_state[0,0], agent_list[i].goal_state[1,0], marker='^', markersize=10, color=colors[i])

	# plt.legend()
	# plt.show()

	# initial guess dict
	initial_guess = {"condition":False,
					 "value":None}

	# iter: timesteps or cost
	time = 0
	cl_data_z = [[] for i in range(n_agents)]
	cl_data_u = [[] for i in range(n_agents)]
	cost = [[] for i in range(n_agents)]
	while time <= 80:
		# dmpc.solve
		dmpc.solve(agent_list, initial_guess=initial_guess)		

		# update agents
		for i in range(n_agents):
			agent_list[i].update_state(u=agent_list[i].u_stack[i*d*horizon:i*d*horizon+d])
			
			# store closed loop data
			cl_data_z[i].append(agent_list[i].current_state)
			cl_data_u[i].append(dmpc.u_average[i*d*n_agents:i*d*n_agents+d])
			cost[i].append(((agent_list[i].current_state-agent_list[i].goal_state).T @ agent_list[i].Q @ (agent_list[i].current_state-agent_list[i].goal_state))[0,0])

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
	print(len(dmpc.plot_data))
	print(len(dmpc.plot_data[0]))
	print(len(dmpc.plot_data[0][0]))
	print(len(dmpc.plot_data[0][0][0]))
	# plt.cla()
	# ax = plt.axes(projection="3d")
	# ax.set_xlim([-3,3])
	# ax.set_ylim([-3,3])
	# ax.set_zlim([-3,3])

	# plt.xlabel("x")
	# plt.ylabel("y")

	# plt.xlim([-4,4])
	# plt.ylim([-4,4])
	
	# for i in range(n_agents):
	# 	label = "agent "+str(i+1)
	# 	x = [state[0,0] for state in cl_data_z[i]]
	# 	y = [state[1,0] for state in cl_data_z[i]]
	# 	# z = [state[2,0] for state in cl_data_z[i]]
	# 	# ax.plot3D(x,y,z,color=colors[i])
	# 	plt.plot(x,y, color=colors[i], label=label)
	# 	# plt.plot(np.arange(len(cost[i])), cost[i], color=colors[i], label=label)

	# plt.legend()
	# plt.show()

	plt.cla()
	# plt.plot()
	# plt.show()
	time = 0
	filenames = []
	while time <= 80:
		plt.cla()

		plt.xlim([-3,3])
		plt.ylim([-3,3])
		# ax = plt.axes(projection="3d")
		# ax.set_xlim([-3,3])
		# ax.set_ylim([-3,3])
		# ax.set_zlim([-3,3])
		for i in range(n_agents):
			plt.scatter(dmpc.plot_data[time][i][0], dmpc.plot_data[time][i][1], marker='.', color=colors[i])
			# ax.scatter3D(dmpc.plot_data[time][i][0], dmpc.plot_data[time][i][1], dmpc.plot_data[time][i][2], color=colors[i])

		filename = 'time'+str(time)+'.png'
		plt.savefig(filename)
		filenames.append(filename)
		plt.pause(0.1)
		time = time+1

	# with imageio.get_writer('2d.gif', mode='I') as writer:
	images = []
	for filename in filenames:
		images.append(imageio.imread(filename))
		# writer.append_data(image)
	imageio.mimsave('2dformation_1.gif', images, format='GIF', fps=10)

	for filename in set(filenames):
		os.remove(filename)


if __name__ == '__main__':
	start()