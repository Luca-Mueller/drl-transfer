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
- cp_v0:   _CartPole-v0_  
- cp_vL:   _cp_v0 -p 0.8_

## Default Agent Params:
- **Learning Rate:** _0.001_
- **Batch Size:** _32_
- **Buffer Size:** _10,000_
- **Gamma:** _0.95_
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
- **Collect Params:**
    + DQN: -w 200
    + DDQN: -w 200
    + DQV: -w 200 -S 0.5
    + DQV2: -w 200 -S 0.5 -t 5
- **Transfer Params:**
    + -r 10
    + -T 2
    + --max-eval 1000
    + No Warm-Up!

#### Experiment 2:
- **Source Env:** _cp_vL_  
- **Target Env:** _cp_v0_  
- **Transfer:**   _Buffer, Model, Buffer + Model_  
- **Collect Params:**
    + DQN: -w 200
    + DDQN: -w 200
    + DQV: -w 200 -S 0.5
    + DQV2: -w 200 -S 0.5 -t 5
- **Transfer Params:**
    + -r 10
    + -T 2
    + --max-eval 1000
    + No Warm-Up!
    
#### Experiment 3:
Limited Experience Replay
- **Max New Trajectories:** _1,000_
- cp_v0 --> cp_vL
- cp_vL --> cp_v0
- **Transfer:**   _Buffer, Model, Buffer + Model_
- **Params::** _see Exp. 1 & 2_

#### Experiment 4:
- Exp. 1 & 2 for Acrobot
- Exp. 1 & 2 for Mountain Car
