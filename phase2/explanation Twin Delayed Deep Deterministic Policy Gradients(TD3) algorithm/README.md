# Reinforcement-Learning-Library

## Twin Delayed Deep Deterministic Poly Gradient network (T3D)

### This github repo explains T3D algorithm

Let us start withthe code 

### Step 1: 
- Initialise the replay buffer wiht "n" random records
![image](/images/replay_buffer.jpg)
```python
class ReplayBuffer(object):
    def __init__(self, max_size = 1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) = self.max_size :
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size) :
        ind = np.random.randint(0, len(self.storage), batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy = False))
            batch_next_states.append(np.array(state, copy = False))
            batch_actions.append(np.array(state, copy = False))
            batch_rewards.append(np.array(state, copy = False))
            batch_dones.append(np.array(state, copy = False))

        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions),\
                np.array(batch_rewards).reshape(-1,1), np.array(batch_dones).reshape(-1,1)


```

### Step 2:
- Initialise Actor model and Actor target \
\
![image](/images/step2.jpg)
```python
class Actor(nn.Module):
    def __init__(self, state_dims, action_dims, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dims, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dims)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x

```
- Each actor model structure is as following image
![image](/images/actor.jpg)

### Step 3:
- Initialise Critic models and critic targets
- In our case we took 2 - Critic Models and 2 corresponding Critic Targets\
\
![image](/images/step3.jpg)
```python
class Critic(nn.Module):
    def __init__(self, state_dims, action_dims):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dims + action_dims, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dims)

        self.layer_4 = nn.Linear(state_dims + action_dims, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, action_dims)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)

        x2 = F.relu(self.layer_4(xu))
        x2 = f.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)

        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

```
- Each Critic model Structure is as following image
![image](/images/critic.jpg)
### initialization step before step 4:
```python
class T3D(object):
    def __init__(self, state_dims, action_dims, max_action):
        self.actor = Actor(state_dims, action_dims, max_action).to(device)
        self.actor_target = Actor(state_dims, action_dims, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dims, action_dims).to(device) ## how can critic output equal action dims
        self.critic_target = Critic(state_dims, action_dims, action_dims).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict)

        self.critic_optmizer = torch.optim.Adam(self.critic.parameters())
        self.max_action =  max_action

```

### Step 4:
- Import a batch of transition records from replay buffer as showin in step1
- Each record consists of (current state, next state, current action , next record, batch done)
```python
def train(self, replay_buffer, iterations, batch_size = 100, discount = 0.99, tau = 0.005, policy_noise = 0.2, noise_clip = 0.5, policy_freq = 2) :
  for it in range(iterations):
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)
```

### Step 5:
- Predict Target for next action trough Actor Target\
  \
![image](/images/step5.jpg)
```python
    next_action = self.actor_target.forward(next_state)
```
### Step 6:
- Add gaussian noise to the next action prediction
- This will help the Actor to explore new paths or stratigies to perform better
```python
    noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
    noise = noise.clamp(-noise_clip, noise_clip)
    next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
```

### Step 7:
- Predict Q values to the next state and next action from the critic Target\
  \
![image](/images/Step7.jpg)
```python
    target_Q1, target_Q2 = self.critic_target.forward(next_state, next_action)
```

### Step 8:
- Calculate minimum Q value from all the critic targets
- This prevents too optmistic estimates of the value of next states\
  \
![image](/images/step8.jpg)
```python
    target_Q = torch.min(target_Q1, target_Q2)
```

### Step 9:
- calculate the final target Q value including the reward
- Q = Reward + gamma * minimum(all target Q values)
- If we observe the above step carefully it is Bellmen ford equation we are using to rain the agent
- Observe the following diagram and formula and compare it with what is happening in current step\
  \
![image](/images/recapf.PNG)
![image](/images/recapi.PNG)
```python
    target_Q = reward + ((1 - done) * discount * target_Q).detach()
```

### Step 10:
- Calculate Critic model current Q-values from current state and action \
  \
![image](/images/Step10.jpg)
```python
    current_Q1, current_Q2 = self.critic.forward(state, action)
```

### Step 11:
- Now as we have already calculated current steps expected Max Q-value from step 9 we can calculate Loss for the Critic model just by calculating difference between Q calculated in step 9(Target Q-value) and step 10(Current Q-value) as shown in figure.
- here MSE() refers to mean square error
  \
![image](/images/Step11.jpg)
```python
    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
```

### Step 12:
- Back propagate all the Ctitic models from the loss calculated in step 11\
  \
![image](/images/Step12.jpg)
```python
    self.critic_optmizer.zero_grad()
    critic_loss.backward()
    self.critic_optmizer.step()
```

### Step 13:
- repeat step 4-12 for policy_freq nouber of iterations
- After every policy_freq number of iterations, backpropagate Actor model using the loss calculated from critic model as shown below
  \
![image](/images/Step13_1.jpg)
```python
if it % policy_freq == 0:
    actor_loss = -(self.critic.Q1(state, self.actor(state)).mean())
```

  \
  
  -------------------------------------------------------------------------------------------------------------------------\
  \
  
![image](/images/Step13_2.jpg)
```python
    self.actor_optimizer.grad_zero()
    actor_loss.backward()
    self.actor_optimizer.step()
```

### Step 14:
- Update weights of actor and critic target models using poliyak averaging
- tetha' = tou * tetha + (1 - tou) * tetha' \
  \
![image](/images/step14.jpg)
```python
    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

