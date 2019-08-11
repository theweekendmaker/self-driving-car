from pynput.keyboard import Key, Controller 
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import pprint
import datetime as datetime
d = datetime.datetime.now()
d = "_".join(str(d).split())

keyboard_auto = Controller()
update_count = 0
episods = 0

WIDTH = 700
HEIGHT = 600
SPEED = 6
car = Actor("racecar")
car.pos = 250,500
trackLeft = []
trackRight = []
trackCount = 0
trackPosition = 250
trackWidth = 120
trackDirection = False
gameStatus = 0
learning_cycles = 12500
trackSpaceState = [None] * 8
state_space = 5

def build_state(features):
	return features[0]*(7**0) + features[1]*(7**1) + features[2]*(7**2) + features[3]*(7**3) + features[4]*(7**4) 

def to_bin(value,bins):
	return np.digitize(x=[value], bins=bins)[0]

class FeatureTransformer:
	def __init__(self):
		self.b2l_bin = np.linspace(0, 200, 6)
		self.b3l_bin = np.linspace(0, 200, 6)
		self.b4l_bin = np.linspace(0, 200, 6)
		self.b5l_bin = np.linspace(0, 200, 6)
		self.b6l_bin = np.linspace(0, 200, 6)
	
	def transform(self, observation):
		# returns an int
		b2l, b3l, b4l, b5l, b6l = observation
		return build_state([ to_bin(b2l, self.b2l_bin), to_bin(b3l, self.b3l_bin), to_bin(b4l, self.b4l_bin), to_bin(b5l, self.b5l_bin), to_bin(b6l, self.b6l_bin)])

class Model:
	def __init__(self, state_space, feature_transformer):
		self.feature_transformer = feature_transformer
		num_states = 7**state_space
		num_actions = 3 # either left or right or nomove
		self.Q = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))
		#pprint.pprint(self.Q)
		#print(self.Q[0,0])

	def predict(self, s):
		x = self.feature_transformer.transform(s)
		return self.Q[x]

	def update(self, s, a, G):
		x = self.feature_transformer.transform(s)
		self.Q[x,a] += 1e-2*(G - self.Q[x,a])

	def sample_action(self, s, eps):
		if np.random.random() < eps:
			return np.random.choice([0,1,2])  #0 = left, 1 = right and 2 = nomove this is our action space
		else:
			p = self.predict(s)
			return np.argmax(p)

ft = FeatureTransformer()
model = Model(state_space, ft)
eps = 1.0/np.sqrt(episods+1.0)
gamma = 0.9

totalrewards = np.empty(learning_cycles)

def draw():
	global gameStatus, trackCount, trackLeft, trackRight, trackSpaceState, car, episods, trackPosition, learning_cycles, eps, gamma, totalrewards
	screen.fill((128,128,128))
	if gameStatus == 0:
		car.draw()
		b = 0
		while b < len(trackLeft):
			trackLeft[b].draw()
			trackRight[b].draw()
			b += 1
	if gameStatus == 1:
		# Red Flag
		print(str(episods)+" Episode done with collision:"+ str(trackCount))
		screen.blit('rflag', (318, 268))
		if episods == learning_cycles:
			plt.plot(totalrewards)
			plt.savefig("Rewards_"+d+".jpg")
			N = len(totalrewards)
			running_avg = np.empty(N)
			for t in range(N):
				running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
			plt.plot(running_avg)
			plt.title("Running Average")
			plt.savefig("Running_Average_"+d+".jpg")
			exit()
		else:
			totalrewards[episods] = trackCount
			episods = episods + 1
			gameStatus = 0
			car.pos = 250,500
			trackLeft = []
			trackRight = []
			trackCount = 0
			trackPosition = 250
			trackSpaceState = [None] * 8
			eps = 1.0/np.sqrt(episods+1.0)
			makeTrack()
	if gameStatus == 2:
		# Chequered Flag
		print(str(episods)+" Episode completed")
		screen.blit('cflag', (318, 268))
		if episods == learning_cycles:
			plt.plot(totalrewards)
			plt.savefig("Rewards_"+d+".jpg")
			N = len(totalrewards)
			running_avg = np.empty(N)
			for t in range(N):
				running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
			plt.plot(running_avg)
			plt.title("Running Average")
			plt.savefig("Running_Average_"+d+".jpg")
			exit()
		else:
			totalrewards[episods] = trackCount
			episods = episods + 1
			gameStatus = 0
			car.pos = 250,500
			trackLeft = []
			trackRight = []
			trackCount = 0
			trackPosition = 250
			trackSpaceState = [None] * 8
			eps = 1.0/np.sqrt(episods+1.0)
			makeTrack()

def update():
	global gameStatus , trackCount, update_count, trackSpaceState, model, eps
	if gameStatus == 0:
		if len(trackLeft) >= 13:
			for b in range(8):
				trackSpaceState[b] = [car.midleft[0] - trackLeft[b].midright[0]]
			action =  model.sample_action([trackSpaceState[2][0],trackSpaceState[3][0],trackSpaceState[4][0],trackSpaceState[5][0],trackSpaceState[6][0]], eps)
			#print(action, eps)
			previous_observation = [trackSpaceState[2][0],trackSpaceState[3][0],trackSpaceState[4][0],trackSpaceState[5][0],trackSpaceState[6][0]]
			
			if action == 0:
				keyboard_auto.press(Key.left)
			elif action == 1:
				keyboard_auto.press(Key.right)

			if keyboard.left:
				car.x = car.x - 16
				keyboard_auto.release(Key.left)
			if keyboard.right:
				car.x = car.x + 16
				keyboard_auto.release(Key.right)

			updateTrack()
			
			observation, reward = [[trackSpaceState[2][0],trackSpaceState[3][0],trackSpaceState[4][0],trackSpaceState[5][0],trackSpaceState[6][0]], trackCount]
			if gameStatus == 1:
				reward = -300
		
			#print(observation, reward)
			G = reward + gamma*np.max(model.predict(observation))
			model.update(previous_observation, action, G)
		else:
			updateTrack()
		
	if trackCount > 200: 
		gameStatus = 2
	#print("update called",update_count)
	#update_count = update_count + 1

def makeTrack(): # Function to make a new section of track
	global trackCount, trackLeft, trackRight, trackPosition, trackWidth
	trackLeft.append(Actor("barrier", pos = (trackPosition-trackWidth,0)))
	trackRight.append(Actor("barrier", pos = (trackPosition+trackWidth,0)))
	trackCount += 1

def updateTrack(): # Function to update where the track blocks appear
	global trackCount, trackPosition, trackDirection, trackWidth, gameStatus
	b = 0
	while b < len(trackLeft):
		if car.colliderect(trackLeft[b]) or car.colliderect(trackRight[b]):
			gameStatus = 1  # Red flag state
		trackLeft[b].y += SPEED
		trackRight[b].y += SPEED
		b += 1
	if trackLeft[0].y > 600:
		del trackLeft[0]
		del trackRight[0]
	if trackLeft[len(trackLeft)-1].y > 32:
		if trackDirection == False: trackPosition += 16
		if trackDirection == True: trackPosition -= 16
		if randint(0, 4) == 1: trackDirection = not trackDirection
		if trackPosition > 700-trackWidth: trackDirection = True
		if trackPosition < trackWidth: trackDirection = False
		makeTrack()

makeTrack()
