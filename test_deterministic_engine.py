from deterministic_engine import DeterministicDecisionEngine


def _sample_datasets():
    import pandas as pd

    return {
        "indb": pd.DataFrame(
            [
                {"food_name": "Jowar Roti", "energy_kcal": 120, "protein_g": 4.0},
                {"food_name": "Wheat Roti", "energy_kcal": 140, "protein_g": 4.0},
                {"food_name": "Moong Dal", "energy_kcal": 105, "protein_g": 7.0},
                {"food_name": "Peanut Dal", "energy_kcal": 180, "protein_g": 8.0},
                {"food_name": "Lauki Sabzi", "energy_kcal": 70, "protein_g": 2.0},
                {"food_name": "Spinach Sabzi", "energy_kcal": 90, "protein_g": 3.0},
                {"food_name": "Curd", "energy_kcal": 80, "protein_g": 4.0},
            ]
        )
    }


def test_determinism_100_runs_identical_output():
    engine = DeterministicDecisionEngine()
    profile = {
        "age": 32,
        "gender": "female",
        "health_conditions": ["pcos"],
        "allergies": ["peanut"],
        "primary_dosha": "kapha",
        "health_goals": ["weight_loss"],
    }
    datasets = _sample_datasets()

    first = engine.recommend_meal(
        session_id="test_det",
        user_profile=profile,
        query="bloating after lunch and low energy",
        datasets=datasets,
        template_id="lunch_roti_dal_sabzi",
    ).model_dump()

    for _ in range(100):
        nxt = engine.recommend_meal(
            session_id="test_det",
            user_profile=profile,
            query="bloating after lunch and low energy",
            datasets=datasets,
            template_id="lunch_roti_dal_sabzi",
        ).model_dump()
        assert nxt == first


def test_p0_allergy_never_violated():
    engine = DeterministicDecisionEngine()
    profile = {
        "health_conditions": ["pcos"],
        "allergies": ["peanut"],
        "health_goals": ["weight_loss"],
    }

    out = engine.recommend_meal(
        session_id="test_p0",
        user_profile=profile,
        query="weight loss",
        datasets=_sample_datasets(),
        template_id="lunch_roti_dal_sabzi",
    )

    selected = " ".join([v or "" for v in out.meal.values()]).lower()
    assert "peanut" not in selected


def test_trace_fidelity_contains_selected_steps():
    engine = DeterministicDecisionEngine()
    profile = {"health_conditions": [], "allergies": []}

    out = engine.recommend_meal(
        session_id="test_trace",
        user_profile=profile,
        query="general healthy lunch",
        datasets=_sample_datasets(),
        template_id="lunch_roti_dal_sabzi",
    )

    steps = "\n".join(out.trace.optimization_steps)
    for slot, food in out.meal.items():
        if food:
            assert f"slot_{slot}_selected:{food}" in steps
