SystematicTrainer was a branch off idea to just use a perfect experience database (it's <400 state-action pairs) to train on.
It was found that handing an xy pair as a state was hard for the networks to learn with.  using a one-hot vector faired much better.
I believe all paths trhough the code are working now, but I might have missed an update somewhere.

screenplay.py will launch new terminal windows.  adding the `-all` flag will launch a bunch of learners and evaluateors with a server.
hmmm...  `-udud` on the client_runner will a perfect experience database for ~3000 training steps to get a jump start on network weights, but may not work when multiple agents learn different weights.
I haven't tested putting servers on different computers (one comp for world, one for tasks, etc.)

TODO:
Still need to have network archetectues be stored in a single place.  Right now you have to change it in multiple files.
Fix spelling mistakes in the readme.
Comment a little bit more.


Code Archetectrue:
client_runner has:
 - world
 - agent (has:)
   - dqn

Algorithm overview:
agent gets state 
agent calls q-value
agent acts on world
world returens expereice
agent stores expereience
agent calls train


You can probably figure out the rest




hline

archive notes:

**Paper drafts can be found in the [wiki](http://pccgit.cs.byu.edu/tetchart/modularDNN_Practice/wikis/home).**

### Design chart can be seen here:
https://www.lucidchart.com/documents/view/d52e3617-3df2-46df-b3f1-ee42f73cf476

## Requires:
- Server-Client messages are handled via `ZMQ`.
- Networks are built using `tensorflow`.
- Obviously, Nvidia `CUDA` and `cuDNN` are also required

