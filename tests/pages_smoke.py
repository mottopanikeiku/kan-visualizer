"""I check the static demo, keyboard controls, and mobile layouts in Chromium."""

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
MODELS = ['model_1d', 'model_2d', 'model_complex']
VIEWS = ['network', 'splines', 'inference', 'training']
EXPLANATIONS = {
    'network': 'sampled edge RMS',
    'splines': 'base activation plus Gaussian RBF',
    'inference': 'actual activations',
    'training': 'not a live optimization',
}


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def keyboard_choice(locator):
    locator.focus()
    locator.press('Home')
    locator.press('ArrowDown')
    locator.press('Enter')
    locator.press('Escape')


def layout_observation(page, model, view, width):
    observation = page.evaluate('''() => {
        const panel = document.querySelector('.panel.active');
        const chart = panel.querySelector('.visualization-container > *');
        const focused = document.activeElement;
        const style = getComputedStyle(focused);
        return {
            viewport_width: innerWidth,
            document_width: document.documentElement.scrollWidth,
            body_width: document.body.scrollWidth,
            panel_width: panel.getBoundingClientRect().width,
            chart_width: chart.getBoundingClientRect().width,
            focus_outline: {style: style.outlineStyle, width: style.outlineWidth},
            focused_view: focused.dataset.view,
            explanation: document.querySelector('#view-explanation').textContent
        };
    }''')
    assert observation['viewport_width'] == width, observation
    assert observation['document_width'] <= width, observation
    assert observation['body_width'] <= width, observation
    assert 200 <= observation['chart_width'] <= width, observation
    assert observation['focus_outline']['style'] != 'none', observation
    assert float(observation['focus_outline']['width'].removesuffix('px')) >= 2, observation
    assert observation['focused_view'] == view, observation
    return {'model': model, 'view': view, **observation}


def numerical_observation(page, evaluation):
    tables = [
        ('Actual node activations',
         [f'{"Input" if layer == 0 else f"Layer {layer}"}, node {node}'
          for layer, values in enumerate(evaluation['activations']) for node in range(len(values))],
         [value for values in evaluation['activations'] for value in values]),
        ('Signed edge contributions',
         [f'Layer {layer + 1}, input {input_index}, output {output}'
          for layer, outputs in enumerate(evaluation['edges'])
          for output, inputs in enumerate(outputs) for input_index in range(len(inputs))],
         [value for outputs in evaluation['edges'] for inputs in outputs for value in inputs]),
    ]
    observations = {}
    for name, expected_labels, expected_values in tables:
        table = page.get_by_role('table', name=name, exact=True)
        assert table.count() == 1
        assert table.evaluate('(element) => element.closest(\'[role="img"]\') === null')
        labels = table.get_by_role('rowheader').all_text_contents()
        values = table.get_by_role('cell').all_text_contents()
        assert labels == expected_labels, labels
        assert len(values) == len(expected_values)
        assert all(abs(float(label) - value) <= 0.000000501
                   for label, value in zip(values, expected_values))
        snapshot = table.aria_snapshot()
        assert f'table "{name}"' in snapshot
        assert f'rowheader "{expected_labels[0]}"' in snapshot
        observations[name] = {'rows': len(values), 'first_label': labels[0],
                              'first_value': values[0], 'last_value': values[-1]}
    return observations


with tempfile.TemporaryDirectory(prefix='kan-pages-preview-') as preview:
    (pathlib.Path(preview) / 'kan-visualizer').symlink_to(ROOT / 'web', target_is_directory=True)
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=preview))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f'http://127.0.0.1:{server.server_port}'
    url = f'{origin}/kan-visualizer/'
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, executable_path=os.environ.get('KAN_CHROMIUM_PATH'), args=['--no-sandbox'])
            page = browser.new_page(viewport={'width': 1440, 'height': 1100}, device_scale_factor=1)
            errors, failed_requests, bad_responses, local_paths = [], [], [], []
            page.on('console', lambda message: errors.append(message.text) if message.type == 'error' else None)
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('requestfailed', lambda request: failed_requests.append({'url': request.url, 'failure': request.failure}))
            page.on('response', lambda response: bad_responses.append({'url': response.url, 'status': response.status}) if response.status >= 400 else None)
            page.on('request', lambda request: local_paths.append(request.url.removeprefix(origin)) if request.url.startswith(origin) else None)
            page.goto(url, wait_until='networkidle')
            page.wait_for_selector('#loading-overlay', state='hidden')
            assert page.locator('.guide-steps button').count() == 4
            assert 'illustrate connectivity' in page.locator('#motion-note').inner_text()
            assert 'real inference' in page.locator('#motion-note').inner_text()
            records, layouts, keyboard_records = [], [], []
            for width in [1440, 390, 320]:
                page.set_viewport_size({'width': width, 'height': 1100})
                for model in MODELS:
                    if page.locator('#model-select').input_value() != model:
                        page.select_option('#model-select', model)
                    page.wait_for_function('(name) => window.kanApp.currentModelName === name && window.kanApp.currentModel.metadata.target_id === ({model_1d: "1d_sine_wave", model_2d: "2d_gaussian", model_complex: "2d_complex"})[name]', arg=model)
                    for view in VIEWS:
                        guide_button = page.locator(f'.guide-steps button[data-view="{view}"]')
                        guide_button.focus()
                        guide_button.press('Enter')
                        page.wait_for_selector(f'#{view}-panel.active')
                        assert page.locator('#visualization-mode').input_value() == view
                        assert guide_button.get_attribute('aria-current') == 'step'
                        assert page.locator('.guide-steps button[aria-current]').count() == 1
                        assert EXPLANATIONS[view] in page.locator('#view-explanation').inner_text()
                        if view == 'network':
                            assert page.locator('#network-svg circle.node').count() > 0
                        elif view == 'splines':
                            page.wait_for_selector('#spline-plot .main-svg')
                        elif view == 'inference':
                            page.wait_for_selector('#activation-plot .main-svg')
                        else:
                            page.wait_for_selector('#loss-plot .main-svg')
                            page.wait_for_selector('#convergence-plot .main-svg')
                        if view == 'network':
                            node_select = page.get_by_label('Inspect node', exact=True)
                            edge_select = page.get_by_label('Inspect edge', exact=True)
                            assert node_select.locator('option').count() == page.locator('#network-svg .node').count() + 1
                            assert edge_select.locator('option').count() == page.locator('#network-svg .edge').count() + 1
                            keyboard_choice(node_select)
                            assert node_select.input_value() == '0'
                            assert 'index: 0' in page.locator('#layer-info').inner_text()
                            assert 'Node: input' in page.locator('#network-selection').inner_text()
                            keyboard_choice(edge_select)
                            page.wait_for_selector('#edge-spline-plot .main-svg')
                            assert edge_select.input_value() == '0'
                            assert page.locator('#network-svg .edge.selected').count() == 1
                            assert 'Sampled full-edge RMS' in page.locator('#network-selection').inner_text()
                            edge = page.evaluate('({weight: kanApp.networkViz.selectedEdge.weight, samples: document.querySelector("#edge-spline-plot").data[0].y})')
                            rms = math.sqrt(sum(value * value for value in edge['samples']) / len(edge['samples']))
                            assert abs(edge['weight'] - rms) < 1e-12
                            page.locator('#network-svg .node').last.click()
                            assert page.locator('#network-svg .node.selected').count() == 1
                            assert node_select.input_value() != '0'
                            keyboard_records.append({'model': model, 'viewport_width': width, 'node_selection': node_select.input_value(), 'edge_selection': edge_select.input_value(), 'sampled_edge_rms': rms, 'sample_count': len(edge['samples'])})
                        elif view == 'splines':
                            keyboard_choice(page.get_by_label('layer:', exact=True))
                            keyboard_choice(page.get_by_label('connection:', exact=True))
                            selected = page.evaluate('({layer: kanApp.splineViz.selectedLayer, connection: kanApp.splineViz.selectedConnection, plot_points: document.querySelector("#spline-plot").data[0].y.length})')
                            assert selected['layer'] == 1
                            assert selected['plot_points'] > 0
                            assert selected['connection'] == page.locator('#connection-select').input_value()
                            keyboard_records.append({'model': model, 'viewport_width': width, 'edge_function_selection': selected})
                        elif view == 'inference':
                            sliders = page.locator('#input-sliders input[type=range]')
                            initial = page.locator('#current-output').inner_text()
                            for index in range(sliders.count()):
                                slider = page.get_by_label(f'input {index}:', exact=True)
                                slider.focus()
                                slider.press('Home')
                                for _ in range(27 if index == 0 else 16):
                                    slider.press('ArrowRight')
                            page.wait_for_function('window.kanApp.inferenceEngine.currentInput[0] === 0.7')
                            state = page.evaluate('({input: kanApp.inferenceEngine.currentInput, output: kanApp.inferenceEngine.currentOutput, target: kanApp.inferenceEngine.targetOutput, target_id: kanApp.currentModel.metadata.target_id, displayed_output: document.querySelector("#current-output").textContent, displayed_target: document.querySelector("#target-output").textContent, activations: kanApp.inferenceEngine.evaluation.activations, activation_labels: [...document.querySelectorAll("#inference-svg .activation-text")].map(node => node.textContent), activation_means: document.querySelector("#activation-plot").data[0].y})')
                            x = state['input'][0]
                            y = state['input'][1] if len(state['input']) > 1 else 0
                            expected = {'model_1d': math.sin(3*x) + 0.3*math.cos(10*x), 'model_2d': math.sin(x)*math.exp(-y*y), 'model_complex': math.sin(x*y) + 0.5*math.tanh(x-y)}[model]
                            assert abs(state['target'] - expected) < 1e-12, state
                            assert state['displayed_output'] != initial, state
                            assert abs(float(state['displayed_output']) - state['output']) <= 0.000501, state
                            assert math.isfinite(state['output'])
                            flattened = [value for layer in state['activations'] for value in layer]
                            assert len(flattened) == len(state['activation_labels'])
                            assert all(abs(float(label) - value) <= 0.000501 for label, value in zip(state['activation_labels'], flattened))
                            for layer, mean in zip(state['activations'], state['activation_means']):
                                assert abs(mean - sum(abs(value) for value in layer) / len(layer)) < 1e-12
                            details = page.locator('#inference-values')
                            summary = page.get_by_text('Read numerical activations and edge contributions', exact=True)
                            summary.focus()
                            summary.press('Enter')
                            assert details.get_attribute('open') is not None
                            evaluation = page.evaluate('kanApp.inferenceEngine.evaluation')
                            accessible_before = numerical_observation(page, evaluation)
                            first_slider = page.get_by_label('input 0:', exact=True)
                            first_slider.focus()
                            first_slider.press('ArrowRight')
                            page.wait_for_function('kanApp.inferenceEngine.currentInput[0] === 0.8')
                            accessible_after = numerical_observation(page, page.evaluate('kanApp.inferenceEngine.evaluation'))
                            assert accessible_before != accessible_after
                            first_slider.press('ArrowLeft')
                            page.wait_for_function('kanApp.inferenceEngine.currentInput[0] === 0.7')
                            numerical_observation(page, page.evaluate('kanApp.inferenceEngine.evaluation'))
                            state['accessible_tables'] = {'before': accessible_before, 'after_slider_change': accessible_after}
                            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                            summary.focus()
                            summary.press('Enter')
                            assert details.get_attribute('open') is None
                            records.append({'model': model, 'viewport_width': width, 'views': VIEWS, **state})
                        else:
                            training = page.evaluate('({actual: document.querySelector("#loss-plot").data[0].y, expected: kanApp.currentModel.training_history.train_loss, traces: document.querySelector("#loss-plot").data.map(trace => trace.name), stats: document.querySelector("#stats-content").textContent})')
                            assert training['actual'] == training['expected']
                            assert len(training['actual']) > 0
                            assert training['traces'] == ['training loss']
                            assert 'no data' not in training['stats']
                            records[-1]['training_points'] = len(training['actual'])
                        guide_button.focus()
                        guide_button.press('ArrowRight')
                        layouts.append(layout_observation(page, model, view, width))
            motion_records = []
            for view in ['network', 'inference']:
                page.emulate_media(reduced_motion='no-preference')
                page.select_option('#visualization-mode', view)
                assert EXPLANATIONS[view] in page.locator('#view-explanation').inner_text()
                play = page.locator('#play-button')
                play.focus()
                play.press('Enter')
                assert play.get_attribute('aria-pressed') == 'true'
                if view == 'inference':
                    page.wait_for_function('kanApp.inferenceEngine.currentInput[0] !== 0')
                else:
                    page.wait_for_selector('#network-svg .data-particle')
                page.emulate_media(reduced_motion='reduce')
                page.wait_for_function('document.querySelector("#play-button").disabled && !kanApp.isAnimating')
                assert play.get_attribute('aria-pressed') == 'false'
                motion = page.evaluate('({view: kanApp.currentMode, reduced_motion: matchMedia("(prefers-reduced-motion: reduce)").matches, network_timer: kanApp.networkViz.animation, inference_timer: kanApp.inferenceEngine.animationInterval ?? null, particles: document.querySelectorAll(".data-particle").length, spinner_animation: getComputedStyle(document.querySelector(".spinner")).animationName})')
                assert motion['network_timer'] is None and motion['inference_timer'] is None
                assert motion['particles'] == 0 and motion['spinner_animation'] == 'none'
                if view == 'inference':
                    slider = page.get_by_label('input 0:', exact=True)
                    slider.focus()
                    slider.press('Home')
                    slider.press('ArrowRight')
                    page.wait_for_function('kanApp.inferenceEngine.currentInput[0] === -1.9')
                motion_records.append(motion)
            page.set_viewport_size({'width': 1440, 'height': 1100})
            page.select_option('#visualization-mode', 'inference')
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
            report = {'url': url, 'browser': f'Playwright Chromium {browser.version}', 'headless': True, 'viewport': {'width': 1440, 'height': 1100}, 'tested_viewport_widths': [1440, 390, 320], 'console_errors': errors, 'failed_requests': failed_requests, 'http_errors': bad_responses, 'local_request_paths': sorted(set(local_paths)), 'models': records, 'layouts': layouts, 'keyboard_selections': keyboard_records, 'reduced_motion': motion_records, 'screenshot': 'docs/assets/pages-inference.png', 'export_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((ROOT / 'web/data').glob('*.json'))}}
            (ROOT / 'results/browser_pages.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report, indent=2))
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
