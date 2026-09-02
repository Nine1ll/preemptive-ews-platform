# import pandas as pd
# import numpy as np

# #TODO 1: 건정성이 조금 변한다
# #TODO 2: 수입/지출이 생긴다 - 건전성에 따라 다름
# # TODO 3: 잔액이 부족하면 현금서비스를 당겨쓴다 - 건전성에 따라 다음 
# # TODO 4: 공과금을 낼 때가 되면 낸거나 못낸다
# # TODO 5: 압박이 크면 연체가 발생한다. 

# for _, c in pop.iterrows():
#     # (1) 매일의 작은 흔들림 + 평균회귀
#     health = health + 0.02 * (0.5 - health) + rng.normal(0, 0.02)
#     # (2) 가끔 큰 충격 (2% 확률)
#     if rng.random() < 0.02:
#         health = health - rng.uniform(0.1, 0.3)
#     # (3) 0~1 범위 강제 (아까 배운 clip!)
#     health = float(np.clip(health, 0.0, 1.0))