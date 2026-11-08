import json
import threading
import time
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path
import pytest
from eeg_transfer.console import Console, handler


@pytest.fixture
def console(tmp_path):
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/test.json').write_text('{"value":42}')
    (tmp_path/'reports').mkdir()
    (tmp_path/'reports/rq1-manifest.json').write_text(json.dumps({'runs':[{'id':'S01-seed42-A0','plans':'data/missing.gz','output':'results/rq1-test','group':'A0'}]}))
    return Console(tmp_path)


def test_http_auth_and_file_boundaries(console):
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(console,'test-token'))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    base=f'http://127.0.0.1:{server.server_port}'
    def request(path,body=None,token='test-token'):
        return urllib.request.urlopen(urllib.request.Request(base+path,data=json.dumps(body).encode() if body is not None else None,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'}))
    try:
        with pytest.raises(urllib.error.HTTPError) as e: request('/api/overview',token='wrong')
        assert e.value.code==401
        assert request('/').status==200
        assert json.load(request('/api/file?path=configs/test.json'))['content']=='{"value":42}'
        for path in ('../../etc/passwd','/etc/passwd','data/console/jobs.json'):
            with pytest.raises(urllib.error.HTTPError) as e: request('/api/file?path='+path)
            assert e.value.code==400
        with pytest.raises(urllib.error.HTTPError): request('/api/jobs',{'action':'shell','command':'touch /tmp/nope'})
        assert not console.jobs
    finally:
        server.shutdown();server.server_close()


def test_formal_requires_explicit_single_run_confirmation(console):
    args={'action':'train','manifest':'reports/rq1-manifest.json','run_id':'S01-seed42-A0'}
    with pytest.raises(ValueError,match='正式实验需要'): console.launch(args)
    with pytest.raises(ValueError,match='计划文件未恢复'): console.launch({**args,'confirmation':'启动单个正式实验'})
    assert not console.jobs


def test_command_allowlist_no_shell_and_serial_execution(console,monkeypatch):
    import eeg_transfer.console as module
    release=threading.Event()
    captured=[]
    class Process:
        pid=12345
        def __init__(self,command,**kwargs):
            captured.append((command,kwargs));self.code=None
        def wait(self):
            release.wait(2);self.code=0;return 0
        def poll(self): return self.code
    monkeypatch.setattr(module.subprocess,'Popen',Process)
    job=console.launch({'action':'tests','command':'arbitrary ignored'})
    try:
        assert captured[0][0][-3:]==['pytest','-q','-ra']
        assert not captured[0][1].get('shell',False)
        with pytest.raises(ValueError,match='已有任务'): console.launch({'action':'tests'})
    finally: release.set()
    for _ in range(100):
        if job['state']=='complete': break
        time.sleep(.01)
    assert job['state']=='complete'
    assert Console(console.root).jobs[0]['state']=='complete'


def test_startup_interrupted_not_fake_complete(console):
    console.jobs=[{'id':'a','state':'running'}];console.save()
    assert Console(console.root).jobs[0]['state']=='interrupted'


def test_symlink_cannot_read_outside_project(console,tmp_path):
    outside=tmp_path.parent/'outside-console-test.json';outside.write_text('{}')
    link=tmp_path/'configs/linked.json';link.symlink_to(outside)
    try:
        with pytest.raises(ValueError,match='路径越界'): console.allowed('configs/linked.json',console.files())
    finally: outside.unlink()


def test_formal_rejects_tampered_plan_and_existing_output(console):
    import hashlib
    package=console.root/'data/plan.gz';package.write_bytes(b'plan')
    row={'id':'S01-seed42-A0','plans':'data/plan.gz','output':'results/rq1-test','group':'A0','plans_sha256':'wrong'}
    manifest=console.root/'reports/rq1-manifest.json'
    manifest.write_text(json.dumps({'runs':[row]}))
    args={'action':'train','manifest':'reports/rq1-manifest.json','run_id':row['id'],'confirmation':'启动单个正式实验'}
    with pytest.raises(ValueError,match='校验和'): console.launch(args)
    row['plans_sha256']=hashlib.sha256(b'plan').hexdigest()
    manifest.write_text(json.dumps({'runs':[row]}))
    (console.root/row['output']).mkdir(parents=True)
    with pytest.raises(ValueError,match='输出已存在'): console.launch(args)
    assert not console.jobs


def test_stop_only_owned_process_group(console,monkeypatch):
    import eeg_transfer.console as module
    class Process:
        pid=12345
        def poll(self): return None
    console.jobs=[{'id':'owned','state':'running'}]
    console.processes={'owned':Process()}
    calls=[]
    monkeypatch.setattr(module.os,'killpg',lambda pid,sig:calls.append((pid,sig)))
    with pytest.raises(ValueError): console.stop('foreign')
    assert calls==[]
    result=console.stop('owned')
    assert result['state']=='stopping'
    assert calls==[(12345,module.signal.SIGTERM)]
