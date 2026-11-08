"""Local project console. Run on the machine that owns the GPU/data."""
import argparse
import hmac
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).with_name('web')


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


class Console:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.state = self.root/'data/console'
        self.state.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = read_json(self.state/'jobs.json', [])
        self.processes = {}
        for job in self.jobs:
            if job['state'] in ('running', 'stopping'):
                job['state'] = 'interrupted'
        self.save()

    def save(self):
        path = self.state/'jobs.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.jobs, indent=2))
        temporary.replace(path)

    def relative(self, p):
        return str(p.relative_to(self.root))

    def manifests(self):
        return sorted(self.root.glob('reports/**/rq1-manifest.json'))

    def allowed(self, value, paths):
        # Only expose known artifacts; never accept arbitrary filesystem paths.
        mapping = {self.relative(p): p for p in paths}
        if value not in mapping:
            raise ValueError('请选择已有项目文件')
        p = mapping[value].resolve()
        if not p.is_relative_to(self.root):
            raise ValueError('路径越界')
        return p

    def files(self):
        paths = list(self.root.glob('configs/*.json'))
        paths += list(self.root.glob('reports/**/*.md'))
        paths += list(self.root.glob('reports/**/protocol-lock.json'))
        paths += list(self.root.glob('reports/**/engineering*.md'))
        for name in ('evaluation.json', 'manifest.json', 'runtime.json', 'gate.json', 'report.json'):
            paths += list(self.root.glob('results/**/'+name))
        return sorted(set(p for p in paths if p.is_file()))

    def overview(self):
        subjects = []
        for i in range(1, 10):
            audits = sorted(self.root.glob(f'reports/**/S{i:02d}.json'))
            preferred = [p for p in audits if 'resumption' in str(p)]
            report = read_json((preferred or audits)[-1], {}) if audits else {}
            subjects.append({'subject': f'S{i:02d}', 'raw': all((self.root/f'data/raw/A{i:02d}{s}.mat').exists() for s in ('T', 'E')), 'processed': (self.root/f'data/processed/S{i:02d}.npz').exists(), 'splits': report.get('split_counts', {}), 'audit': self.relative((preferred or audits)[-1]) if audits else None})
        manifests = []
        for path in self.manifests():
            m = read_json(path, {})
            runs = m.get('runs', [])
            manifests.append({'path': self.relative(path), 'runs': len(runs), 'available': sum((self.root/r['plans']).exists() for r in runs), 'completed': sum(read_json(self.root/r['output']/'status.json', {}).get('state') == 'complete' for r in runs)})
        gpu = '未检测到 NVIDIA GPU'
        try:
            gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader'], text=True, timeout=3).strip()
        except (OSError, subprocess.SubprocessError):
            pass
        return {'root': str(self.root), 'gpu': gpu, 'python': sys.version.split()[0], 'subjects': subjects, 'manifests': manifests, 'files': [self.relative(p) for p in self.files()], 'jobs': self.jobs, 'protocol': read_json(self.root/'reports/phase3/protocol-lock.json', {}), 'note': '协议锁和学习验证为历史证据；短程验证不构成负迁移结论。'}

    def launch(self, request):
        with self.lock:
            if any(j['state'] in ('running', 'stopping') for j in self.jobs):
                raise ValueError('已有任务在运行，请等待或停止后再启动')
            action = request.get('action')
            identity = time.strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(3)
            base = [sys.executable, '-u']
            output = None
            if action == 'tests':
                command = base+['-m', 'pytest', '-q', '-ra']
            elif action == 'audit':
                subject = int(request.get('subject', 1))
                if subject not in range(1, 10):
                    raise ValueError('受试者必须为1至9')
                output = f'reports/console/{identity}'
                (self.root/output).mkdir(parents=True)
                # Read existing source files only; never overwrite historical derived data.
                command = base+['-m', 'eeg_transfer.bnci', '--subject', str(subject), '--output', f'data/console/{identity}.npz', '--report', output+'/audit.json']
            elif action == 'download':
                subject = int(request.get('subject', 1))
                if subject not in range(1, 10): raise ValueError('受试者必须为1至9')
                if (self.root/f'data/processed/S{subject:02d}.npz').exists():
                    raise ValueError('已有派生数据，拒绝覆盖；请使用重新审计')
                output = f'reports/console/{identity}'
                command = base+['-m', 'eeg_transfer.console_worker', '--subject', str(subject), '--out', output]
            elif action == 'prepare':
                output = f'reports/console/{identity}'
                command = base+['-m', 'eeg_transfer.suite', '--out', f'data/plans/console-{identity}', '--reports', output, '--results-root', f'results/rq1-console-{identity}']
            elif action == 'smoke':
                output = f'results/rq1-console-smoke-{identity}'
                cfg = read_json(self.root/'configs/rq1-suite.json')['template']
                cfg.update(data_path='data/processed/S01.npz', seed=42, steps=4, evaluation_trial_limit=1, purpose='console_engineering_smoke_only')
                cfg.pop('protocol_lock', None); cfg.pop('minimum_evaluation_coverage', None)
                cfgpath = self.state/f'{identity}.json'
                cfgpath.write_text(json.dumps(cfg))
                command = base+['-m', 'eeg_transfer.console_worker', '--config', str(cfgpath), '--plans', f'data/plans/console-smoke-{identity}', '--out', output]
            elif action in ('train', 'summary'):
                manifest = self.allowed(request.get('manifest'), self.manifests())
                suite = read_json(manifest)
                if action == 'summary':
                    output = f'reports/console/{identity}'
                    (self.root/output).mkdir(parents=True)
                    command = base+['-m', 'eeg_transfer.summarize', '--manifest', str(manifest), '--out', output+'/summary.json']
                else:
                    if request.get('confirmation') != '启动单个正式实验':
                        raise ValueError('正式实验需要输入：启动单个正式实验')
                    matches = [r for r in suite['runs'] if r['id'] == request.get('run_id')]
                    if len(matches) != 1:
                        raise ValueError('请选择唯一运行项')
                    row = matches[0]
                    for key in ('plans', 'output'):
                        if not (self.root/row[key]).resolve().is_relative_to(self.root):
                            raise ValueError('清单路径越界')
                    if not (self.root/row['plans']).exists():
                        raise ValueError('此历史清单的计划文件未恢复，请选择新清单')
                    from .bnci import sha256
                    if sha256(self.root/row['plans']) != row['plans_sha256']:
                        raise ValueError('计划校验和不符，拒绝启动')
                    if (self.root/row['output']).exists():
                        raise ValueError('输出已存在，拒绝覆盖；中断实验须在新计划输出路径重启')
                    output = row['output']
                    command = base+['-m', 'eeg_transfer.experiment', 'run', '--plans', row['plans'], '--group', row['group'], '--out', output]
                    if request.get('evaluate_test') is True:
                        command.append('--evaluate-test')
            else:
                raise ValueError('不支持的操作')
            log = self.state/f'{identity}.log'
            with log.open('w') as f:
                process = subprocess.Popen(command, cwd=self.root, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            job = {'id': identity, 'action': action, 'state': 'running', 'started': time.time(), 'command': command, 'output': output, 'returncode': None}
            self.jobs.append(job); self.processes[identity] = process; self.save()
            threading.Thread(target=self.finish, args=(job, process), daemon=True).start()
            return job

    def finish(self, job, process):
        code = process.wait()
        with self.lock:
            job.update(returncode=code, ended=time.time(), state='stopped' if job['state']=='stopping' else ('complete' if code==0 else 'failed'))
            self.processes.pop(job['id'], None); self.save()

    def stop(self, identity):
        with self.lock:
            p = self.processes.get(identity)
            if not p or p.poll() is not None:
                raise ValueError('任务已结束或不属于此服务进程')
            job = next(j for j in self.jobs if j['id']==identity)
            job['state'] = 'stopping'; self.save()
            os.killpg(p.pid, signal.SIGTERM)
            return job

    def detail(self, identity):
        with self.lock:
            job = next((j.copy() for j in self.jobs if j['id']==identity), None)
        if job is None:
            raise ValueError('任务不存在')
        path = self.state/f'{identity}.log'
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size-100000)); log = f.read().decode('utf-8', 'replace')
        result = {'job': job, 'log': log, 'metrics': [], 'evaluation': None}
        if job.get('output'):
            directory = self.root/job['output']
            try:
                result['metrics'] = [json.loads(s) for s in (directory/'train.jsonl').read_text().splitlines()[-300:]]
            except (OSError, ValueError):
                pass
            result['evaluation'] = read_json(directory/'evaluation.json') or read_json(directory/'summary.json') or read_json(directory/'audit.json')
        return result


def handler(console, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, body, mime='application/json'):
            data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode() if mime=='application/json' else body
            self.send_response(status); self.send_header('Content-Type', mime+'; charset=utf-8')
            self.send_header('Cache-Control', 'no-store'); self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            if not url.path.startswith('/api/'):
                name = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}.get(url.path)
                if not name:
                    self.send(404, {'error': 'Not found'}); return
                self.send(200, (WEB/name).read_bytes(), {'index.html':'text/html','app.js':'text/javascript','style.css':'text/css'}[name]); return
            if not self.authorized(): return
            try:
                query = parse_qs(url.query)
                if url.path == '/api/overview': body = console.overview()
                elif url.path == '/api/job': body = console.detail(query.get('id', [''])[0])
                elif url.path == '/api/file':
                    p = console.allowed(query.get('path', [''])[0], console.files()+[p for p in console.root.glob('reports/**/S*.json')])
                    body = {'path': console.relative(p), 'content': p.read_text()[:200000]}
                elif url.path == '/api/runs':
                    p = console.allowed(query.get('manifest', [''])[0], console.manifests())
                    body = read_json(p)['runs']
                else: self.send(404, {'error':'Not found'}); return
                self.send(200, body)
            except (ValueError, OSError) as exc:
                self.send(400, {'error': str(exc)})

        def authorized(self):
            provided = self.headers.get('Authorization', '')
            if not hmac.compare_digest(provided, 'Bearer '+token):
                self.send(401, {'error': '请填写控制台访问令牌'}); return False
            return True

        def do_POST(self):
            if not self.authorized(): return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 8192: raise ValueError('请求过大或为空')
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict): raise ValueError('无效请求')
                path = urlparse(self.path).path
                if path == '/api/jobs': result = console.launch(body)
                elif path == '/api/stop': result = console.stop(body.get('id'))
                else: self.send(404, {'error':'Not found'}); return
                self.send(200, result)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self.send(400, {'error': str(exc)})
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=8765)
    args = p.parse_args()
    token = os.environ.get('EEG_CONSOLE_TOKEN') or secrets.token_urlsafe(32)
    server = ThreadingHTTPServer((args.host, args.port), handler(Console(), token))
    print(f'EEG Console: http://{args.host}:{args.port}', flush=True)
    print(f'Access token: {token}', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

if __name__ == '__main__': main()
