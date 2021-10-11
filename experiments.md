#### Envs
- cp_v0:   _CartPole-v0_  
- cp_vL:   _cp_v0 -p 0.8_

#### Experiment 1:
- **Agents:**     _DQN, DDQN, DQV_  
- **Source Env:** _cp_v0_  
- **Target Env:** _cp_vL_  
- **Transfer:**   _Buffer, Model, Buffer + Model_  
- **Collect Params:**
    + DQN: -s 500 -w 200
    + DDQN: -s 500 -w 200
    + DQV: -s 500 -w 200 -S 0.5
- **Transfer Params:**
    + -r 10
    + -T 2
    + --max-eval 1000
    + No Warm-Up!

#### Experiment 2:
- **Agents:**     _DQN, DDQN, DQV_  
- **Source Env:** _cp_vL_  
- **Target Env:** _cp_v0_  
- **Transfer:**   _Buffer, Model, Buffer + Model_  
- **Collect Params:**
    + DQN: -s 500 -w 200
    + DDQN: -s 500 -w 200
    + DQV: -s 500 -w 200 -S 0.5
- **Transfer Params:**
    + -r 10
    + -T 2
    + --max-eval 1000
    + No Warm-Up!
    
#### Experiment 3:
Limited Experience Replay
