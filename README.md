## Project Overview
This project was created as an attempt to create modular Neural Networks.  The idea is that we can share information in each layer between multiple instances.  In the global scene, we will have 3x copies of the `world` layer, each one containing information for their specific world.  In this project, it was simply a maze world.  Then, we will have 3x copies of the `task` layer, again, each copy contianing information for that specific task.  Tasks in this project were end positions (an x,y pair).  Once an agent reached that location, it would be given a reward.  Finally, there will be 3x copies of the `agent` layer.  Each agent had 4 options of movemen: north, east, south, west.  The difference between each agent was the order, i.e. nesw, eswn, swne.

The goal is to be able to train each instance of each layer, but not all at the same time.  If we can share the weights and the learning between training instances, then previoiusly unseen combinations should be able to produce correct actions.  To explain, lets look at an simplified example of just task and agents.  The notation will be an array of two numbers: [task id, agent id].  [1,2] would repressent agent 2 performing task 1.  We will use the following pairs.

|Training combinations|Evaluating Combinations|
|---|---|
|       [1,2], [1,3]|[1,1]|
|[2,1],        [2,3]|[2,2]|
|[3,1], [3,2]|[3,3]|

In the case of the evaluating combinations, we have never trained on these specific pairs.  But we have trained the individual layers.  Because the layers were trained in other combinations, they have all figured out how to communicate with eachother, so new, unseed combinations shouldn't be a big problem.

The idea of this project is to help reduce the amount of training needed for approaching new combinations.  To me, it feels somewhat akin to transfer learning.  In the future, we were planning on applying this idea to bigger and more complex problems, so this repository stood as more of a proof of concept.

### Tensorflow setup:
* Each local learner/evaluater has it's onw copies of it's specific layers.
* Each time it evalueates a move, it uses its local copy.
* Each time it trains on experience, it uses its local copy.
 * After so many times training, it will sync with the Parameter Servers.  This means that it will
  * Calculate its local deltas on the weights
  * Send those weights tot he Parameter server to add to the global weights
  * Request the new global weights specific to its task.
  * Stash those weights to calculate the deltas for the next sync process.
  
Due to the slowness of requesting weights from the Parameter server, we dicided to cache the weights locally.  This had a significant impact on performace.
  

### Code Archetectrue:
client_runner has:
 - world
 - agent
   - dqn

### Algorithm overview:
agent gets state 
agent calls q-value
agent acts on world
world returns expereience
agent stores expereience
agent calls train


### Design chart can be seen here:
https://www.lucidchart.com/documents/view/d52e3617-3df2-46df-b3f1-ee42f73cf476 
