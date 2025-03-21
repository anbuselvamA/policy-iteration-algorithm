# POLICY ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm.

## PROBLEM STATEMENT
The aim of this experiment is to find optimal policy for the mdp using policy iteration. Policy iteration includes policy evaluation and policy improvement where evaluation function is used to find optimal value function of each state and then improvement function is used to find best policy by comparing all the action value function as well as policy.

## POLICY ITERATION ALGORITHM
-> Step1 :
we are going to do policy evaluation of each state to get the state value function where the initial policy is defined randomly to the mdp.

-> Step2:
Once we obtain convergence in the policy evaluation then implement policy improvement where we are going to find best optimal policy until the previous and current policy are same.
</br>
</br>

## POLICY IMPROVEMENT FUNCTION
#### Name : Anbuselvam A
#### Register Number : 212222240009
```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    # Write your code here to improve the given policy
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob,next_state,reward,done in P[s][a]:
          Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
          new_pi=lambda s:{s:a for s, a in enumerate(np.argmax(Q,axis=1))}[s]
    return new_pi
```
## POLICY ITERATION FUNCTION
#### Name : Anbuselvam A
#### Register Number : 212222240009
```python
def policy_iteration(P, gamma=1.0, theta=1e-10):
   random_actions=np.random.choice(tuple(P[0].keys()),len(P))
   pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
   while True:
    old_pi = {s:pi(s) for s in range(len(P))}
    V = policy_evaluation(pi, P,gamma,theta)
    pi = policy_improvement(V,P,gamma)
    if old_pi == {s:pi(s) for s in range(len(P))}:
      break
   return V, pi
```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
![image](https://github.com/user-attachments/assets/1f532179-aa07-49cb-b1a4-2a0d9cf63d64)
![image](https://github.com/user-attachments/assets/c2ee3e96-160b-4915-afb2-fe0ad896f7d4)



### 2. Policy, Value function and success rate for the Improved Policy
![image](https://github.com/user-attachments/assets/5d365fe6-4b04-44c7-8481-d847a19352fe)
![image](https://github.com/user-attachments/assets/b9c13319-930a-48d5-9ed9-fd3e705eee31)



### 3. Policy, Value function and success rate after policy iteration
![image](https://github.com/user-attachments/assets/5d365fe6-4b04-44c7-8481-d847a19352fe)
![image](https://github.com/user-attachments/assets/2f859289-d90d-4a42-9cf8-f39898da0dcd)


## RESULT:
Thus, The Python program to find the optimal policy for the given MDP using the policy iteration algorithm is successfully executed.
