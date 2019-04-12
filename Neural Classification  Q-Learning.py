import numpy as np
import random as r
import matplotlib.pyplot as plt
#wall = 1
#start = 2
#finish = 3
#punishment = -x
r.seed(1)
class NeuralNet:
    def __init__(self, starting,num1Neurons, num2Neurons,output):
        self.syn1 = np.random.random((starting,num1Neurons))*2-1
        self.syn2 = np.random.random((num1Neurons,num2Neurons))*2-1
        self.syn3 = np.random.random((num2Neurons,output))*2-1
        self.err = 0
        #print(self.inputs.shape, self.syn1.shape,self.syn2.shape)
    #initialize weights and biases
        

    def sigmoid(self, x):
        x = 1 / (np.exp(-x) + 1)
        return x

    def sigmoid_dev(self, x):
       return x * ( 1 - (x))
    
    def relu(self,x):
        return 0.5*(x+np.abs(x))

    def relu_dev(self, x):
        y = self.relu(x)
        for i in range(len(y)):
            for r in range(len(y[i])):
                if y[i][r] != 0:
                    y[i][r] = 1
        return y
        
    def think(self,inputs):
        
        self.layer1 = self.sigmoid(np.dot(inputs,self.syn1))
        self.layer2 = self.sigmoid(np.dot(self.layer1,self.syn2))
        self.layer3 = self.sigmoid(np.dot(self.layer2,self.syn3))
        
    def rthink(self,inputs):
        
        self.layer1 = self.relu(np.dot(inputs,self.syn1))
        self.layer2 = self.relu(np.dot(self.layer1,self.syn2))
        self.layer3 = self.relu(np.dot(self.layer2,self.syn3))
        
    def guess(self, inputs):
        self.think(inputs)
        return self.layer3
    
    def train(self,inputs,answer):
        self.think(inputs)
        error3 = answer - self.layer3
        #print(error3)
        delta3 = error3*self.sigmoid_dev(self.layer3)
        error2 = np.dot(delta3,self.syn3.T)
        delta2 = error2*self.sigmoid_dev(self.layer2)
        error1 = np.dot(delta2,self.syn2.T)
        delta1 = error1*self.sigmoid_dev(self.layer1)
        self.syn1 += np.dot(inputs.T,delta1)
        self.syn2 += np.dot(self.layer1.T,delta2)
        self.syn3 += np.dot(self.layer2.T,delta3)
        self.err = 0
        for i in range(error3.shape[0]):
            for s in range(error3.shape[1]):
                self.err += abs(error3[i][s])
        self.err /= error3.shape[0]*error3.shape[1]
        
    def rtrain(self,inputs,answer):
        self.rthink(inputs)
        error3 = answer - self.layer3
        #print(error3)
        delta3 = error3*self.relu_dev(self.layer3)
        error2 = np.dot(delta3,self.syn3.T)
        delta2 = error2*self.relu_dev(self.layer2)
        error1 = np.dot(delta2,self.syn2.T)
        delta1 = error1*self.relu_dev(self.layer1)
        self.syn1 += np.dot(inputs.T,delta1)
        self.syn2 += np.dot(self.layer1.T,delta2)
        self.syn3 += np.dot(self.layer2.T,delta3)
        self.err = 0
        for i in range(error3.shape[0]):
            for s in range(error3.shape[1]):
                self.err += abs(error3[i][s])
        self.err /= error3.shape[0]*error3.shape[1]
  
        
                
class maze:
    def __init__(self):
        self.maze = [[0,0,0,0,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,1],
                [1,0,1,0,0,0,0,0,0,1],
                [1,0,1,0,0,0,0,0,1,1],
                [1,0,1,0,0,0,0,0,3,1],
                [1,0,0,0,0,1,0,0,1,1],
                [1,0,0,0,0,1,0,0,1,1],
                [1,0,1,0,0,0,1,0,0,1],
                [1,2,1,0,0,0,0,0,0,1],
                [1,1,1,1,1,1,1,1,1,1]]
        self.start = (8,1)
        self.location = self.start
        self.pastLocation = self.location
        self.finish = (4,8)
        
    def move(self,move):
        self.pastLocation = self.location
        #print(move)
        #print(self.location)
        
        if move == 0:
            self.location = (self.location[0]-1,self.location[1])
        if move == 1:
            self.location = (self.location[0],self.location[1]+1)
        if move == 2:
            self.location = (self.location[0]+1,self.location[1])
        if move == 3:
            self.location = (self.location[0],self.location[1]-1)
            
        if self.maze[self.location[0]][self.location[1]] == 1:
            self.location = self.pastLocation
        #print(self.location)
        return self.location, move
    
    def finished(self):
        if self.maze[self.location[0]][self.location[1]] == 3:
            return True
        return False
    
class Q:
    def __init__(self):
        self.envi = maze()
        self.numInputs = len(self.envi.maze)* len(self.envi.maze[0])
        self.current = self.envi.start
        self.neural = NeuralNet(self.numInputs,8,8,4)
        self.currPath = [[self.envi.start,2]]
        self.esp = 1
        self.err = []
        self.states = {self.envi.start:{0:0,1:0,2:0,3:0},self.envi.finish:{1:1,2:1,3:1,0:1}}
        
    
    def maxAct(self,state):
        guess = 1
        #print(state)
        for i in range(4):
            if self.states[state][guess] < self.states[state][i]:
                guess = i
        return guess
    def makeInput(self, point):
        blank = [0]*self.numInputs
        blank[point[0]*len(self.envi.maze) + point[1]] = 1
        return blank

        
    def move(self):
        past = self.current
        if not self.current in self.states.keys():
            self.states[self.current] = {0:0,1:0,2:0,3:0}
        if r.random() > self.esp:
            self.current, tempMove = self.envi.move(self.maxAct(self.current))
        else:
            self.current, tempMove = self.envi.move(r.randint(0,3))
        self.currPath.append([past,tempMove])
    def nMove(self):
        past = self.current
        if not self.current in self.states.keys():
            self.states[self.current] = {0:0,1:0,2:0,3:0}
        if r.random() > self.esp:
            #print(self.guess(np.array([self.makeInput(self.current)])),self.current)
            trash , checkMove = self.envi.move(self.maxAct(self.current))
            self.current, tempMove = self.envi.move(self.guess(np.array([self.makeInput(self.current)])))
            if checkMove != tempMove:
                print("wrong")
        else:
            self.current, tempMove = self.envi.move(r.randint(0,3))
        self.currPath.append([past,tempMove])
        

    def adjustValues(self):
        #loop through list
        
        self.currPath.append([self.envi.finish,1])
        revStates = self.currPath[::-1]
        #print(self.states.keys())
        adjusted = []
        #print(revStates)
        for i in range(len(revStates)):
            if i != 0 and not revStates[i] in adjusted:
                adjusted.append(revStates[i])
                self.states[revStates[i][0]][revStates[i][1]] = 0.8 * self.states[revStates[i][0]][revStates[i][1]] + 0.2*0.2*self.states[revStates[i-1][0]][self.maxAct(revStates[i-1][0])]

    def trainNetwork(self,r):
        for t in range(r):
            netInputs = []
            netAnswers = []
            for i in self.states.keys():
                blank = [self.states[i][0],self.states[i][1],self.states[i][2],self.states[i][3]]
                blank[self.maxAct(i)] = 1
                netInputs.append(self.makeInput(i))
                netAnswers.append(blank)
            netInputs = np.array(netInputs)
            netAnswers = np.array(netAnswers)
            self.neural.train(netInputs,netAnswers)
            self.err.append(self.neural.err)
        print(self.err[-1])
        plt.plot(self.err)
        plt.show()
     
    def guess(self,state):
        output = [x for x in self.neural.guess(state)[0]]
        return output.index(max(output))

    def rguess(self,state):
        output = [x for x in self.neural.guess(state)[0]]
        return output
    
    def run(self):
        count = 0
        total = []
        for i in range(1000):
            while self.current != self.envi.finish:
                self.move()
                count+=1
            #print(i)
            self.adjustValues()
            self.currPath = [[self.envi.start,1]]
            total.append(count)
            count = 0
            self.esp -= 1/1000
            self.esp = max((self.esp,.1))
            self.current = self.envi.start
            self.envi.location = self.current
        self.trainNetwork(20000)
        plt.xlabel("Number of Finishes")
        plt.ylabel("Steps to Finish")
        plt.title('Explicit Q-Learning')
        plt.plot(total)
        plt.show()
    def nRun(self):
        count = 0
        total = []
        for i in range(1000):
            while self.current != self.envi.finish:
                self.nMove()
                count+=1
            #print(i)
            self.adjustValues()
            self.currPath = [[self.envi.start,1]]
            total.append(count)
            count = 0
            self.esp = .01
            self.current = self.envi.start
            self.envi.location = self.current
            if i%100 == 0:
                print(i)
        plt.xlabel("Number of Finishes")
        plt.ylabel("Steps to Finish")
        plt.title('Neural Classification Q-Learning')
        plt.plot(total)
        plt.show()
 
        
            

q = Q()
#print(q.maxAct((4,8)))
q.run()
q.nRun()

            
#for i in q.states.keys():
    #print(i,q.states[i])          
        
    
            
    
    
        
        
  
        
        
