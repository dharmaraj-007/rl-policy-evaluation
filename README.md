# POLICY EVALUATION

## AIM
To evaluate and compare different policies in the Frozen Lake environment and find the best policy for reaching the goal successfully.

## PROBLEM STATEMENT
In the Frozen Lake environment, an agent must navigate from the start to the goal while avoiding holes. Movements are uncertain due to slipperiness. A policy guides the agent’s actions, but not all policies are effective. The task is to:

Evaluate a given policy (V1) using policy evaluation. Create and test a new policy (V2) to improve performance. Compare both policies based on success rate and rewards. Find the best policy for safely reaching the goal. This helps in identifying the most efficient way to complete the task.

## POLICY EVALUATION FUNCTION
```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        delta = 0
        for s in range(len(P)):
            v = 0
            a = pi(s)  # action chosen by the policy at state s
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V
```

## OUTPUT:
### POLICY 1:
<img width="848" height="239" alt="image" src="https://github.com/user-attachments/assets/e30639cc-a9aa-4ece-940a-fc3c26534403" />
<img width="765" height="163" alt="image" src="https://github.com/user-attachments/assets/0de18734-f60f-44c4-9a2f-a6edff814765" />

### POLICY 2:
<img width="848" height="239" alt="image" src="https://github.com/user-attachments/assets/e30639cc-a9aa-4ece-940a-fc3c26534403" />
<img width="787" height="179" alt="image" src="https://github.com/user-attachments/assets/3622e3b4-8e94-406b-a87e-a4c17f126cb4" />

### COMPARISON:
<img width="833" height="321" alt="image" src="https://github.com/user-attachments/assets/27ea5bf7-8d0a-42d8-979c-c2842f07a723" />

## RESULT:

Thus, The Python program to evaluate the given policy is successfully executed.
