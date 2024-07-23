# B"H

Pytorch with cuda support
http://incompleteideas.net/book/RLbook2020.pdf
Chapter 3
"agent" - computer program
"environment"
"state" - agents internal representation of its environment
In each timestep (t1, t2, t3, etc)
E.g. Every 50ms
agent perceives environment
agent has "state" for that timestep
agent performs "action" at that timestep
agent receives "reward" at that timestep

"policy" - given (current_state, set_of_potential_actions)- choose from the set of actions to maximize expected long-term reward
Where long term reward is reward(current_timestep) + discount * reward(future_timestep1) + discount * reward(future_timestep2) ....
Discount is because future rewards are uncertain
0.99 or 0.9 often used for discount

Value function: expected long term reward
e.g. (current_state | take action=a1)
e.g. (current_state | take action=a2)
e.g. (current_state | take action=a3)
example: Where a1 could be stay in lane, a2 could be left lane change, a3 could be right lane change

Reinforment learning has a bunch of arcane terminology, but basically above is the jist

For this project, youre estimating the value of each state (where value is expected long term reward)
State (past 4 frames)
Action (left, stay, right)
Reward +1 on brick -1 on missed ball
For each state, have the network predict the reward (this is the action you choose – the one with highest)
example: Left: 0.5 center:2, right -0.5
"memory replay buffer" - loop: you run the model for 1000 timesteps, record the experiences, then train on the experiences
DeepMind original 2015 atari paper worth reading
This is in tensorflow, convert to pytorch
Color is 3 2d images (red, green, blue) - this gets to merged to grayscale x*red + y *green + z * blue
Input: 4d image (past 4 frames) ... It skips like 3 frames each timestep (gives you more motion of the ball)
