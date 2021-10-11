## Agents:
- DQN
- DDQN
- DQV

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

#### Cart Pole Parameters:
- **Gravity:** _9.8_
- **Mass Cart:** _1.0_
- **Mass Pole:** _0.1_
- **Pole Length:** _0.5_

#### Experiment 1:
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
