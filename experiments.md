## Agents:
- DQN
- DDQN
- DQV
- DQV2

## Networks:
- **MLP:** _4x32x32x2 (Q) and 4x32x32x1 (V, V2)_
- **Optimizer:** _Adam_
- **Loss:** _MSE_

## Envs:
- cp_v0:    _CartPole-v0_  
- cp_vL:    _cp_v0 -p 0.8_
- ac_v1:    _Acrobot-v1_
- ac_vL:    _?_
- mc_v0:    _MountainCar-v0_
- mc_vL:    _?_

## Default Agent Params:
- **Learning Rate:** _0.001_
- **Batch Size:** _32_
- **Buffer Size:** _10,000_
- **Gamma:** _0.99_
- **Epsilon Start:** _0.9_
- **Epsilon End:** _0.01_
- **Epsilon Decay:** _0.995_
- **Episodes:** _200_
- **Max Training Steps:** _None_
- **Warm Up Steps:** _0_
- **Target Update Period:** _10_

## Cart Pole Parameters:  
- **Gravity:** _9.8_
- **Mass Cart:** _1.0_
- **Mass Pole:** _0.1_
- **Pole Length:** _0.5_

#### Experiment 1:  
- **Source Env:** _cp_v0_  
- **Target Env:** _cp_vL_  
- **Transfer:**   _Buffer, Model, Buffer + Model_  
- **Episodes:** _200_  
- **Agent Params:**  
    + **DQV:** -t 5  
    + **DQV2:** -t 5  
- **Collect Params:**  
    + -w 200  
- **Transfer Params:**  
    + -r 10  
    + -T 2  
    + No Warm-Up!  

#### Experiment 2:  
- **Source Env:** _cp_vL_  
- **Target Env:** _cp_v0_  
- **Transfer:**   _Buffer, Model, Buffer + Model_  
- **Episodes:** _200_  
- **Agent Params:**  
    + **DQV:** -t 5  
    + **DQV2:** -t 5  
- **Collect Params:**  
    + -w 200   
- **Transfer Params:**  
    + -r 10  
    + -T 2  
    + No Warm-Up!  
    
#### Experiment 3:  
Limited Experience Replay  
- **Max New Trajectories:** _1,000_  
- cp_v0 --> cp_vL  
- cp_vL --> cp_v0  
- **Transfer:** _Buffer, Model, Buffer + Model_  
- **Params::** _see Exp. 1 & 2_  

#### Experiment 4:  
##### Exp. 1 & 2 for Acrobot  
- **Episodes:** _800_  
- **Collect Params:**  
    + **Warm Up:** _1,000_  
    + **DQV:** -t 5  
    + **DQV2:** -t 5  
- **Transfer Params:**  
    + -r 5  
    + -T 2  
    + No Warm-Up!  
##### Exp. 1 & 2 for Mountain Car  
