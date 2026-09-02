import numpy as np
import pandas as pd

rng = np.random.default_rng(42)


# 나이대별 설정: 비율(p)와 소득분포(평균, 표준편차)를 담아둠
# 대한민국 통계청 인구구조 및 가계금융복지조사 기반 연령대별 설정
# income: 만원 단위 / std는 평균의 약 35% (data-sources.md 참조)
# p: 인구 비율 (합이 1이 되도록) — 실제 성인 인구 비율 기반 근사

AGE_BANDS = {
    "20s":  {"p": 0.18, "income_mean": 255, "income_std": 90},
    "30s":  {"p": 0.22, "income_mean": 379, "income_std": 130},
    "40s":  {"p": 0.24, "income_mean": 438, "income_std": 150},
    "50s":  {"p": 0.21, "income_mean": 415, "income_std": 145},
    "60s":  {"p": 0.15, "income_mean": 243, "income_std": 100},
}

def sample_income(mean, std, rng):
    """소득을 고려해 로그 정규 분포로 만든다."""
    mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
    sigma = np.sqrt(np.log(1 + std**2 / mean**2))
    return rng.lognormal(mu, sigma)

def make_population(n_customers):
    bands = list(AGE_BANDS.keys())
    probs = [AGE_BANDS[b]["p"] for b in bands]

    rows = []
    for cid in range(n_customers):
        # 1. 나이대를 비율에 따라 뽑기
        age = rng.choice(bands, p=probs)
        # 2. 그 나이대의 소득 분포에서 소득 뽑기
        cfg = AGE_BANDS[age]
        income = sample_income(cfg["income_mean"], cfg["income_std"], rng)
        # 3. 소득에서 잔액, 한도, 건전성 파생시키기
        balance = income * rng.uniform(0.2, 1.5)
        limit = income * rng.uniform(1.5, 4.0)
        # 건전성
        # rng.beta(2, 2): 개인 편차 => 베타분포 사용 0~1 사이에서 가운데가 많이 나오고 양 끝은 드물게 나오는 분포
        # + (income - 350) / 3000: 소득 약하게 반영 
        # np.clip(..., 0.05, 0.95): 1을 넘거나 0 이하로 가면 잡아주는 장치 
        financial_soundness = rng.beta(2, 2) + (income - 350) / 3000
        financial_soundness = float(np.clip(financial_soundness, 0.05, 0.95))
        # 4. dict로 만들어서 rows에 append
        rows.append({
            "customer_id": cid,
            "age_band": age,
            "income": round(income, 1),
            "balance": round(balance, 1),
            "credit_limit": round(limit, 1),
            "soundness": round(financial_soundness, 3),
        })

    return pd.DataFrame(rows)


def simulate(pop, days=120):
    records = []
    for _, c in pop.iterrows():
        health = c['soundness']

        for day in range(days):
            # (1) 매일의 작은 흔들림 + 평균회귀
            health = health + 0.02 * (0.5 - health) + rng.normal(0, 0.02)
            # (2) 가끔 큰 충격 (2% 확률)
            if rng.random() < 0.02:
                health = health - rng.uniform(0.1, 0.3)
            # (3) 0~1 범위 강제
            health = float(np.clip(health, 0.0, 1.0))

            records.append({
                "customer_id": c["customer_id"],
                "day": day,
                "health": round(health, 3),
            })

    assert len(records) == days * len(pop), "고객 날짜 데이터가 부족합니다."
    return pd.DataFrame(records)



if __name__ == "__main__":
    df = make_population(300)
    assert df['age_band'].isin(['20s','30s','40s','50s','60s']).all(), "나이대가 20~60대를 벗어남"
    print("나이대 범위 통과")

    assert df['soundness'].between(0,1).all(), "건전성이 0~1 범위를 벗어남"
    print("건전성 범위 통과")

    simulate(df)