import pytest
from studio.models import MixSettings
from studio.store import Store

def test_project_survives_reopen(tmp_path):
    db = tmp_path / 'studio.db'
    s = Store(db)
    p = s.create_project('Song', {'id': 'source', 'path': 'audio.wav'})
    s.update_settings(p['id'], MixSettings(rotation_cpm=8))
    recovered = Store(db).get_project(p['id'])
    assert recovered['title'] == 'Song'
    assert recovered['settings']['rotation_cpm'] == 8
    assert recovered['asset']['id'] == 'source'

@pytest.mark.parametrize('settings', [{'rotation_cpm': float('nan')}, {'room_size': 2}, {'motion_depth': -1}, {'crossover_hz': 1000}, {'panning_preset': 'unknown'}])
def test_settings_reject_nonfinite_and_out_of_range(settings):
    with pytest.raises(ValueError):
        MixSettings(**settings)
