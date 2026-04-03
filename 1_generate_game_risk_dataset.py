import random
from dataclasses import dataclass, asdict
from typing import List
import pandas as pd


@dataclass
class Sample:
    sample_id: int
    player_archetype: str
    enemy_reward: int
    death_penalty: int
    win_probability: float
    time_pressure: float
    recent_death: int
    player_type: float
    decision_score: float
    decision: int


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def sample_player_type() -> tuple[str, float]:
    """
    Generate player risk tendency by three groups:
    conservative / balanced / aggressive
    """
    group = random.choices(
        population=["conservative", "balanced", "aggressive"],
        weights=[0.3, 0.4, 0.3],
        k=1
    )[0]

    if group == "conservative":
        value = random.uniform(0.05, 0.35)
    elif group == "balanced":
        value = random.uniform(0.35, 0.65)
    else:
        value = random.uniform(0.65, 0.95)

    return group, round(value, 3)


def sample_time_pressure() -> float:
    """
    Most scenarios are low-to-medium pressure,
    while extreme pressure appears less often.
    """
    mode = random.choices(
        population=["low", "medium", "high"],
        weights=[0.45, 0.35, 0.20],
        k=1
    )[0]

    if mode == "low":
        value = random.uniform(0.00, 0.35)
    elif mode == "medium":
        value = random.uniform(0.35, 0.70)
    else:
        value = random.uniform(0.70, 1.00)

    return round(value, 3)


def sample_recent_death() -> int:
    """
    Recent death is a binary condition.
    """
    return random.choices([0, 1], weights=[0.7, 0.3], k=1)[0]


def sample_win_probability(player_type: float, recent_death: int) -> float:
    """
    Win probability is not purely random.
    It is influenced by player tendency and recent failure.
    Aggressive players may enter harder fights,
    while recent death may slightly reduce confidence.
    """
    base = random.uniform(0.20, 0.85)

    # Aggressive players slightly tend to challenge harder fights.
    base -= 0.10 * (player_type - 0.5)

    # Recent death slightly lowers perceived chance of success.
    if recent_death == 1:
        base -= random.uniform(0.03, 0.08)

    value = clamp(base, 0.10, 0.90)
    return round(value, 3)


def sample_enemy_reward(win_probability: float, time_pressure: float) -> int:
    """
    Reward is partially structured rather than fully random.
    More difficult battles (lower win probability) tend to give higher reward.
    High time pressure can also slightly increase urgency/reward.
    """
    base_reward = 40 + (1.0 - win_probability) * 120 + time_pressure * 20
    noise = random.uniform(-18, 18)
    value = int(round(clamp(base_reward + noise, 10, 200)))
    return value


def sample_death_penalty(win_probability: float, time_pressure: float) -> int:
    """
    Penalty is also partially structured.
    Harder fights and urgent situations may bring larger penalty.
    """
    base_penalty = 20 + (1.0 - win_probability) * 90 + time_pressure * 15
    noise = random.uniform(-15, 15)
    value = int(round(clamp(base_penalty + noise, 5, 150)))
    return value


def compute_decision_score(
    enemy_reward: int,
    death_penalty: int,
    win_probability: float,
    time_pressure: float,
    recent_death: int,
    player_type: float
) -> float:
    """
    Behavioral decision model:
    - Higher expected gain increases attack tendency
    - Higher expected loss decreases attack tendency
    - Aggressive players are more willing to take risks
    - Recent death makes players more conservative
    - Time pressure increases decision instability
    """
    expected_gain = enemy_reward * win_probability
    expected_loss = death_penalty * (1.0 - win_probability)

    score = 0.0
    score += expected_gain
    score -= expected_loss

    # Player tendency: conservative -> lower score, aggressive -> higher score
    score += (player_type - 0.5) * 60.0

    # Recent failure makes players more cautious
    if recent_death == 1:
        score -= 18.0

    # Time pressure adds instability / impulsiveness
    pressure_noise = random.uniform(-1, 1) * (8.0 + 22.0 * time_pressure)
    score += pressure_noise

    # Small general human randomness
    score += random.uniform(-8.0, 8.0)

    return round(score, 3)


def score_to_decision(score: float, threshold: float = 0.0) -> int:
    return 1 if score > threshold else 0


def generate_one_sample(sample_id: int) -> Sample:
    player_archetype, player_type = sample_player_type()
    time_pressure = sample_time_pressure()
    recent_death = sample_recent_death()

    win_probability = sample_win_probability(
        player_type=player_type,
        recent_death=recent_death
    )

    enemy_reward = sample_enemy_reward(
        win_probability=win_probability,
        time_pressure=time_pressure
    )

    death_penalty = sample_death_penalty(
        win_probability=win_probability,
        time_pressure=time_pressure
    )

    decision_score = compute_decision_score(
        enemy_reward=enemy_reward,
        death_penalty=death_penalty,
        win_probability=win_probability,
        time_pressure=time_pressure,
        recent_death=recent_death,
        player_type=player_type
    )

    decision = score_to_decision(decision_score, threshold=0.0)

    return Sample(
        sample_id=sample_id,
        player_archetype=player_archetype,
        enemy_reward=enemy_reward,
        death_penalty=death_penalty,
        win_probability=win_probability,
        time_pressure=time_pressure,
        recent_death=recent_death,
        player_type=player_type,
        decision_score=decision_score,
        decision=decision
    )


def generate_dataset(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)

    records: List[dict] = []
    for i in range(1, n_samples + 1):
        sample = generate_one_sample(i)
        records.append(asdict(sample))

    df = pd.DataFrame(records)

    # Optional balancing step if distribution is too skewed
    # This keeps the dataset more suitable for classification experiments.
    positive_ratio = df["decision"].mean()

    if positive_ratio < 0.35 or positive_ratio > 0.65:
        random.seed(seed + 1)
        records = []
        for i in range(1, n_samples + 1):
            sample = generate_one_sample(i)
            records.append(asdict(sample))
        df = pd.DataFrame(records)

    return df


def print_dataset_summary(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"Number of samples: {len(df)}")
    print()
    print("Decision distribution:")
    print(df["decision"].value_counts().sort_index())
    print()
    print("Decision ratio:")
    print(df["decision"].value_counts(normalize=True).sort_index().round(3))
    print()
    print("Player archetype distribution:")
    print(df["player_archetype"].value_counts(normalize=True).round(3))
    print()
    print("Feature ranges:")
    for col in [
        "enemy_reward",
        "death_penalty",
        "win_probability",
        "time_pressure",
        "recent_death",
        "player_type",
        "decision_score"
    ]:
        print(
            f"{col:18s} min={df[col].min():>8} "
            f"max={df[col].max():>8} "
            f"mean={df[col].mean():>8.3f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    df = generate_dataset(n_samples=500, seed=42)
    output_path = "game_risk_decision_dataset.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print_dataset_summary(df)
    print(f"Saved to: {output_path}")