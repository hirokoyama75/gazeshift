import json


def test_idle_probe_measures_and_closes_own_process(tmp_path, monkeypatch, metrics):
    from scripts import build_macos as builder
    calls, now = [], [0.]
    class Process:
        pid = 123
        def __init__(self, command, **kwargs): calls.append(command)
        def poll(self): return None
        def terminate(self): calls.append('terminate')
        def wait(self, **kwargs): calls.append('wait')
    monkeypatch.setattr(builder.subprocess, 'Popen', Process)
    samples = iter(['00:01.00 102400', '00:01.02 102400'])
    monkeypatch.setattr(builder.subprocess, 'check_output', lambda *args, **kw: next(samples))
    monkeypatch.setattr(builder.time, 'sleep', lambda seconds: now.__setitem__(0, now[0]+seconds))
    monkeypatch.setattr(builder.time, 'monotonic', lambda: now[0])
    builder.check_idle(tmp_path/'Gazeshift', tmp_path)
    report = json.loads((tmp_path/'idle.json').read_text())
    assert report['status']=='PASS' and report['rss_mib']==100
    assert abs(report['cpu_percent_of_one_core']-.4)<1e-6
    assert calls[0][0]==str(tmp_path/'Gazeshift') and calls[-2:]==['terminate','wait']
    assert '--settings' in calls[0] and '--config' in calls[0]
    metrics.update(measured_cpu_delta=True, isolated_preferences=True, own_process_terminated=True)
