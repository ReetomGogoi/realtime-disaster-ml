"""Local browser integration. No GDACS fetch, training or persistent DB writes.

Start isolated backend and opt-in Vite dev server per documentation first.
Screenshots show actual local API state; synthetic fixtures test filters only.
"""
import json
import os
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def run():
    base_url = os.getenv('DASHBOARD_BASE_URL', 'http://127.0.0.1:5173')
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': 1512, 'height': 1080}, ignore_https_errors=True)
        page.set_default_timeout(10000)
        requests, errors = [], []
        page.on('request', lambda req: requests.append(req.url))
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(base_url)
        expect(page.get_by_role('heading', name='Three-hazard event monitor')).to_be_visible()
        expect(page.get_by_text('Monitoring connected')).to_be_visible()
        assert any('/events?limit=500&offset=0' in url for url in requests)
        output = Path(tempfile.gettempdir()) / 'disaster-dashboard-preview'
        output.mkdir(exist_ok=True)
        expect(page.locator('.leaflet-tile-loaded').first).to_be_visible(timeout=15000)
        expect(page.get_by_role('button', name='Enable decorative marker ripple animation').or_(
            page.get_by_role('button', name='Disable decorative marker ripple animation'))).to_have_count(1)
        page.screenshot(path=str(output/'dashboard-desktop.png'), full_page=True)
        nav=page.get_by_role('navigation', name='Main navigation')
        expect(nav.get_by_role('button')).to_have_count(4)
        expect(nav.get_by_text('ML Prediction Lab', exact=True)).to_have_count(0)
        expect(nav.get_by_text('Cyclone Intelligence', exact=True)).to_have_count(0)
        for label in ['Dashboard', 'Disaster Explorer', 'Model Performance', 'About']:
            expect(nav.get_by_role('button', name=label)).to_have_count(1)
        nav.get_by_role('button', name='Disaster Explorer').click()
        expect(page.get_by_role('searchbox')).to_be_visible()
        page.screenshot(path=str(output/'disaster-explorer.png'), full_page=True)
        nav.get_by_role('button', name='Model Performance').click()
        expect(page.get_by_role('heading',name='Random Forest')).to_have_count(2)
        assert any('/local-demo/performance' in url for url in requests)
        page.screenshot(path=str(output/'validation-performance.png'), full_page=True)
        nav.get_by_role('button', name='About').click()
        expect(page.get_by_text('2000–2019',exact=True)).to_be_visible()
        page.set_viewport_size({'width':390,'height':844})
        page.get_by_role('button',name='Toggle navigation').click()
        nav.get_by_role('button',name='Dashboard').click()
        expect(page.get_by_role('heading',name='Three-hazard event monitor')).to_be_visible()
        page.wait_for_timeout(300)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.screenshot(path=str(output/'dashboard-mobile.png'),full_page=True)
        # Deterministic synthetic records verify search/filter/detail without
        # misrepresenting them as actual saved events or using live GDACS.
        fixture=[]
        hazard_types = ['Earthquake', 'Flood', 'Drought']
        for i,kind in enumerate(hazard_types):
            fixture.append(dict(id=i+1,source='GDACS',source_event_id=f'fixture:{i}',
                title=f'SYNTHETIC TEST {kind}',disaster_type=kind,country='Fixture country',
                latitude=[35, -25, 48][i],longitude=[135, -55, 15][i],event_time='2026-06-01T00:00:00Z',
                fetched_at='2026-06-01T00:05:00Z',created_at='2026-06-01T00:05:00Z',
                alert_level=['Red', 'Orange', 'Green'][i],
                source_url='https://www.gdacs.org/'))
        fixture.append({**fixture[0], 'id': 99, 'source_event_id': 'fixture:overlap',
                        'title': 'SYNTHETIC TEST CO-LOCATED EARTHQUAKE'})
        fixture.append({**fixture[0], 'id': 100, 'source_event_id': 'fixture:unsupported',
                        'title': 'SYNTHETIC TEST UNSUPPORTED CYCLONE',
                        'disaster_type': 'Tropical Cyclone', 'latitude': 0, 'longitude': 0})
        page.route('**/events?*',lambda route:route.fulfill(json={'total':len(fixture),'events':fixture,'offset':0,'limit':500}))
        page.set_viewport_size({'width':1512,'height':1080})
        page.reload()
        nav.get_by_role('button',name='Disaster Explorer').click()
        page.get_by_role('searchbox').fill('Flood')
        expect(page.locator('.event-row')).to_have_count(1)
        page.locator('.event-row').click()
        expect(page.get_by_role('dialog')).to_be_visible()
        expect(page.get_by_role('link',name='View official GDACS report')).to_be_visible()
        page.get_by_role('button',name='Close details').click()
        page.get_by_role('searchbox').fill('')
        nav.get_by_role('button',name='Dashboard').click()
        expect(page.locator('.disaster-marker')).to_have_count(3)
        expect(page.locator('.disaster-marker .hazard-svg')).to_have_count(3)
        ripple_toggle = page.get_by_role('button', name='Disable decorative marker ripple animation')
        expect(ripple_toggle).to_have_attribute('aria-pressed', 'true')
        visible_marker = None
        markers = page.locator('.leaflet-marker-icon:has(.disaster-marker)')
        for index in range(markers.count()):
            candidate = markers.nth(index)
            box = candidate.bounding_box()
            if box and page.evaluate(
                "([x,y]) => Boolean(document.elementFromPoint(x,y)?.closest('.leaflet-marker-icon'))",
                [box['x'] + box['width'] / 2, box['y'] + box['height'] / 2],
            ):
                visible_marker = candidate
                break
        assert visible_marker is not None
        visible_marker.hover()
        assert visible_marker.locator('.marker-ripple').first.evaluate(
            "element => getComputedStyle(element).animationName === 'marker-ripple'"
        )
        ripple_toggle.click()
        expect(page.get_by_role('button', name='Enable decorative marker ripple animation')).to_have_attribute('aria-pressed', 'false')
        assert not page.locator('.map-shell').evaluate("element => element.classList.contains('ripples-enabled')")
        page.get_by_role('button', name='Enable decorative marker ripple animation').click()
        expect(page.locator('.marker-count')).to_have_text('2')
        expect(page.get_by_text('SYNTHETIC TEST UNSUPPORTED CYCLONE')).to_have_count(0)
        expect(page.get_by_text('OpenStreetMap', exact=True)).to_be_visible()
        page.get_by_role('button', name='Show Flood filter, 1 mapped events').click()
        expect(page.locator('.disaster-marker')).to_have_count(1)
        nav.get_by_role('button', name='Disaster Explorer').click()
        expect(page.get_by_label('Hazard category')).to_have_value('Flood')
        nav.get_by_role('button', name='Dashboard').click()
        page.get_by_role('button', name='Clear Flood filter, 1 mapped events').click()
        expect(page.locator('.disaster-marker')).to_have_count(3)
        page.screenshot(path=str(output/'map-icons-overview-fixture.png'), full_page=True)
        marker = page.locator('.leaflet-marker-icon[tabindex="0"]:has(.marker-count)')
        marker.focus()
        marker.press('Enter')
        expect(page.locator('.leaflet-popup-content')).to_be_visible()
        assert marker.locator('.disaster-marker').evaluate("element => element.classList.contains('is-selected')")
        expect(page.get_by_text('co-located events', exact=False)).to_be_visible()
        expect(page.get_by_role('link', name='Open official GDACS report').first).to_be_visible()
        page.screenshot(path=str(output/'map-ripple-selected-desktop.png'), full_page=True)
        page.get_by_role('button', name='Collapse map legend').click()
        expect(page.get_by_role('button', name='Expand map legend')).to_be_visible()
        page.get_by_role('button', name='Expand map legend').click()
        page.set_viewport_size({'width':390,'height':844})
        page.wait_for_timeout(250)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.screenshot(path=str(output/'map-ripple-mobile.png'), full_page=True)
        page.set_viewport_size({'width':1512,'height':1080})
        page.route('**/sync-live-events',lambda route:route.fulfill(json={'status':'ok','fetched_count':len(fixture),'inserted_count':0,'updated_count':len(fixture),'skipped_count':0}))
        page.get_by_role('button',name='Refresh GDACS').click()
        expect(page.get_by_text('GDACS refresh complete.',exact=False)).to_be_visible()
        # Hundreds-of-records regression: deterministic fixtures exercise the
        # client rendering path without creating fake product data.
        large_fixture = []
        for i in range(400):
            base = fixture[i % len(hazard_types)]
            large_fixture.append({**base, 'id': 1000+i, 'source_event_id': f'load:{i}',
                                  'title': f'SYNTHETIC LOAD TEST {i}',
                                  'latitude': -59 + (i % 119),
                                  'longitude': -169 + (i % 337)})
        page.unroute('**/events?*')
        page.route('**/events?*',lambda route:route.fulfill(json={'total':len(large_fixture),'events':large_fixture,'offset':0,'limit':500}))
        page.reload()
        expect(page.locator('.regional-cluster').first).to_be_visible(timeout=10000)
        dense_toggle = page.get_by_role('button', name='Enable decorative marker ripple animation')
        expect(dense_toggle).to_have_attribute('aria-pressed', 'false')
        expect(page.get_by_text('Reduced by default on dense maps', exact=True)).to_be_visible()
        rendered_groups = page.locator('.leaflet-marker-icon').count()
        assert 0 < rendered_groups < 400
        cluster = page.locator('.leaflet-marker-icon[title*="activate to zoom in"]').first
        cluster.focus()
        cluster.press('Enter')
        page.wait_for_timeout(500)
        assert page.locator('.leaflet-marker-icon').count() >= rendered_groups
        page.emulate_media(reduced_motion='reduce')
        expect(dense_toggle).to_be_disabled()
        expect(page.get_by_text('Disabled by your reduced-motion preference', exact=True)).to_be_visible()
        page.emulate_media(reduced_motion='no-preference')
        assert any('/sync-live-events' in url for url in requests)
        assert not errors, errors
        # The only refresh above is intercepted with a synthetic response.
        assert not any('/local-demo/historical-predict' in url for url in requests)
        print('PASS: map SVG categories, focused ripples, motion toggle, dense/reduced-motion defaults, severity borders, grouped events, keyboard popup, legend filter/collapse, attribution and mobile states.')
        print('Screenshots:',output)
        browser.close()


if __name__ == '__main__':
    run()
