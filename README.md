# Learning From Demo
This is a pytorch implemention of paper "Learning Montezuma revenge from a single demonstration". In this repository, PPO or other reinforcement learning algorithms is adopted as optimizer in an imitation learning framework to learning Montezuma's Revenge and Pitfall, which are two hardest explore game in Atari 2600. 

## Env

## Usage
```python
python main.py
```

## Data stracture

Demo (dict)
-   actions
-   signed rewards
-   reward
-   lives
-   checkpoints
-   checkpoint_action_nr

## Update log
1. 写好了replay demo 和reset env相关的类
2. agent 已测试

## Reference
