from eightd_engine.mix_profiles import FIREFLIES_8D_REFERENCE, PROFILES


def test_fireflies_reference_profile_matches_measured_rotation():
    assert FIREFLIES_8D_REFERENCE.name in PROFILES
    assert 5.7 < FIREFLIES_8D_REFERENCE.rotation_cpm < 5.9
    assert FIREFLIES_8D_REFERENCE.crossover_hz == 150.0
    assert 0.1 <= FIREFLIES_8D_REFERENCE.room_size <= 0.25
