# path_planning_deepRL

This project is about making a path planning robot that'll discover the road on it's own, and then follow it. It was done by Subash(responsible for the Kivy design) and I(DQN, neural network) under th guidance of Dr. Abhishek Sarkar from the department of Mechanical Engineering in BITS-Pilani, Hyderabad Campus).

As of now I've implemented:
1. the DQN algorithm, and ran it on the cart-pole env. 
2. DQN2 (which is the simpler version of the grokking one)
3. The DQN algorithm, and the neural network from scratch using Numpy (for some reason, I wasn't able to install kivy on Ubuntu, so I proceeded to do it on windows)
4. A very simple simulation on kivy (in no way is this the final model, we plan on making it solve a maze, for now to see if it works, we made it follow a line!)

A picture of the line-follower: (I couldn't figure out how to upload a video, sorry :( )
![line-DQN](https://github.com/anushtup-nandy/path_planning_deepRL/blob/main/Pic_line_follower.png)

THANK YOU!


## Help taken:
I took a lot of help from the book "Grokking deep RL" by Miguel Morales to be able to implement the DQN, and for learning!

## Plan:

1. simulate a car training itself to follow a road. (will use KIVY for this)
2. create the simulation env  using Kivy. 
3. implement DDQN and compare results with DQN.


## Papers Read:
[1] V. Mnih et al., “Playing Atari with Deep Reinforcement Learning.” arXiv, Dec. 19, 2013. Accessed: Oct. 17, 2022. [Online]. Available: http://arxiv.org/abs/1312.5602
