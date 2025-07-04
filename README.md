# qwen3 single classify
    - src/cls_train_gpu.py
    - src/cls_eval_gpu.py
    - src/cls_train_scratch.py (from scratch)

# Qwen3 rank reward model
    - src/reward_train_gpu.py (train)
    - src/reward_eval_gpu.py (eval)


# Qwen3 multitask score model
    - reward_train_gpu.py (train)
    - reward_eval_gpu.py (eval)
    - data/score_model_* (dataset)


# llama factory train yaml
    - qwen3_dpo_reward.yaml  (using GPT4.1 dataset)
    - skywork_reward_model.yaml (skywork finetune)
