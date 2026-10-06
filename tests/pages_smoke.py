"""Check the static Pages artifact at its project prefix with Playwright Chromium."""

import functools
import hashlib
import http.server
import json
import math
import os
import pathlib
import tempfile
import threading
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parents[1]

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

with tempfile.TemporaryDirectory(prefix='kan-pages-preview-') as preview:
    (pathlib.Path(preview) / 'kan-visualizer').symlink_to(ROOT / 'web', target_is_directory=True)
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=preview))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}/kan-visualizer/'
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, executable_path=os.environ.get("KAN_CHROMIUM_PATH"), args=['--no-sandbox'])
            page = browser.new_page(viewport={'width': 1440, 'height': 1100}, device_scale_factor=1)
            errors, failed_requests, bad_responses, local_paths = [], [], [], []
            page.on('console', lambda message: (errors.append(message.text), print('Console error:', message.text)) if message.type == 'error' else None)
            page.on('pageerror', lambda error: (errors.append(str(error)), print('Page error:', error)))
            page.on('requestfailed', lambda request: failed_requests.append({'url': request.url, 'failure': request.failure}))
            page.on('response', lambda response: bad_responses.append({'url': response.url, 'status': response.status}) if response.status >= 400 else None)
            page.on('request', lambda request: local_paths.append(request.url.removeprefix(f'http://127.0.0.1:{server.server_port}')) if request.url.startswith(f'http://127.0.0.1:{server.server_port}') else None)
            page.goto(url, wait_until='networkidle')
            page.wait_for_selector('#loading-overlay', state='hidden')
            records = []
            for model in ['model_1d', 'model_2d', 'model_complex']:
                page.select_option('#model-select', model)
                page.wait_for_function('(name) => window.kanApp.currentModelName === name && window.kanApp.currentModel.metadata.target_id === ({model_1d: "1d_sine_wave", model_2d: "2d_gaussian", model_complex: "2d_complex"})[name]', arg=model)
                page.select_option('#visualization-mode', 'network')
                assert page.locator('#network-svg circle').count() > 0
                page.select_option('#visualization-mode', 'splines')
                page.wait_for_selector('#spline-plot .main-svg')
                page.select_option('#visualization-mode', 'training')
                page.wait_for_selector('#loss-plot .main-svg')
                page.wait_for_selector('#convergence-plot .main-svg')
                training = page.evaluate('({actual: document.querySelector("#loss-plot").data[0].y, expected: kanApp.currentModel.training_history.train_loss, traces: document.querySelector("#loss-plot").data.map(trace => trace.name), stats: document.querySelector("#stats-content").textContent})')
                assert training['actual'] == training['expected']
                assert len(training['actual']) > 0
                assert training['traces'] == ['training loss']
                assert 'no data' not in training['stats']
                page.select_option('#visualization-mode', 'inference')
                sliders = page.locator('#input-sliders input[type=range]')
                initial = page.locator('#current-output').inner_text()
                for index in range(sliders.count()):
                    slider = sliders.nth(index)
                    slider.focus()
                    slider.press('Home')
                    for _ in range(27 if index == 0 else 16):
                        slider.press('ArrowRight')
                page.wait_for_function('window.kanApp.inferenceEngine.currentInput[0] === 0.7')
                state = page.evaluate('({input: kanApp.inferenceEngine.currentInput, output: kanApp.inferenceEngine.currentOutput, target: kanApp.inferenceEngine.targetOutput, target_id: kanApp.currentModel.metadata.target_id, displayed_output: document.querySelector("#current-output").textContent, displayed_target: document.querySelector("#target-output").textContent})')
                x = state['input'][0]
                y = state['input'][1] if len(state['input']) > 1 else 0
                expected = {'model_1d': math.sin(3*x) + 0.3*math.cos(10*x), 'model_2d': math.sin(x)*math.exp(-y*y), 'model_complex': math.sin(x*y) + 0.5*math.tanh(x-y)}[model]
                assert abs(state['target'] - expected) < 1e-12, state
                assert state['displayed_output'] != initial, state
                assert math.isfinite(state['output'])
                records.append({'model': model, 'views': ['network', 'splines', 'training', 'inference'], 'training_points': len(training['actual']), **state})
            page.wait_for_load_state('networkidle')
            assert all(path.startswith('/kan-visualizer/') for path in local_paths), local_paths
            for filename in ['datasets.json', 'model_1d.json', 'model_2d.json', 'model_complex.json']:
                assert f'/kan-visualizer/data/{filename}' in local_paths, local_paths
            assert not errors, errors
            assert not failed_requests, failed_requests
            assert not bad_responses, bad_responses
            screenshot = ROOT / 'docs/assets/pages-inference.png'
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(screenshot), full_page=True, animations='disabled')
            report = {'url': url, 'browser': f'Playwright Chromium {browser.version}', 'headless': True, 'viewport': {'width': 1440, 'height': 1100}, 'console_errors': errors, 'failed_requests': failed_requests, 'http_errors': bad_responses, 'local_request_paths': sorted(set(local_paths)), 'models': records, 'screenshot': 'docs/assets/pages-inference.png', 'export_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((ROOT / 'web/data').glob('*.json'))}}
            (ROOT / 'results/browser_pages.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report, indent=2))
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
