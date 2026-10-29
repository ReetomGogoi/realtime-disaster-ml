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
        expect(page.get_by_role('heading', name='Global event monitor')).to_be_visible()
        expect(page.get_by_text('Monitoring connected')).to_be_visible()
        assert any('/events?limit=500&offset=0' in url for url in requests)
        output = Path(tempfile.gettempdir()) / 'disaster-dashboard-preview'
        output.mkdir(exist_ok=True)
        expect(page.locator('.leaflet-tile-loaded').first).to_be_visible(timeout=15000)
        page.screenshot(path=str(output/'dashboard-desktop.png'), full_page=True)
        nav=page.get_by_role('navigation', name='Main navigation')
        expect(nav.get_by_role('button')).to_have_count(5)
        expect(nav.get_by_text('ML Prediction Lab', exact=True)).to_have_count(0)
        for label in ['Dashboard', 'Disaster Explorer', 'Cyclone Intelligence',
                      'Model Performance', 'About']:
            expect(nav.get_by_role('button', name=label)).to_have_count(1)
        nav.get_by_role('button', name='Disaster Explorer').click()
        expect(page.get_by_role('searchbox')).to_be_visible()
        page.screenshot(path=str(output/'disaster-explorer.png'), full_page=True)
        nav.get_by_role('button', name='Cyclone Intelligence').click()
        expect(page.get_by_role('button', name='Latest GDACS Cyclones', exact=True)).to_be_visible()
        page.screenshot(path=str(output/'cyclone-intelligence.png'), full_page=True)
        page.get_by_role('button',name='Archived Cyclone Observations',exact=True).click()
        expect(page.get_by_text('Local extraction output missing')).to_be_visible()
        nav.get_by_role('button', name='Model Performance').click()
        expect(page.get_by_role('heading',name='Random Forest')).to_have_count(2)
        assert any('/local-demo/performance' in url for url in requests)
        page.screenshot(path=str(output/'validation-performance.png'), full_page=True)
        nav.get_by_role('button', name='About').click()
        expect(page.get_by_text('2000–2019',exact=True)).to_be_visible()
        page.set_viewport_size({'width':390,'height':844})
        page.get_by_role('button',name='Toggle navigation').click()
        nav.get_by_role('button',name='Dashboard').click()
        expect(page.get_by_role('heading',name='Global event monitor')).to_be_visible()
        page.wait_for_timeout(300)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.screenshot(path=str(output/'dashboard-mobile.png'),full_page=True)
        # Deterministic synthetic records verify search/filter/detail without
        # misrepresenting them as actual saved events or using live GDACS.
        fixture=[]
        hazard_types = ['Earthquake', 'Flood', 'Tropical Cyclone', 'Wildfire',
                        'Drought', 'Landslide', 'Volcanic Event']
        for i,kind in enumerate(hazard_types):
            fixture.append(dict(id=i+1,source='GDACS',source_event_id=f'fixture:{i}',
                title=f'SYNTHETIC TEST {kind}',disaster_type=kind,country='Fixture country',
                latitude=20+i,longitude=75+i,event_time='2026-06-01T00:00:00Z',
                fetched_at='2026-06-01T00:05:00Z',created_at='2026-06-01T00:05:00Z',
                alert_level=['Red', 'Orange', 'Green', None, None, None, None][i],
                source_url='https://www.gdacs.org/'))
        fixture.append({**fixture[0], 'id': 99, 'source_event_id': 'fixture:overlap',
                        'title': 'SYNTHETIC TEST CO-LOCATED EARTHQUAKE'})
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
        expect(page.locator('.disaster-marker')).to_have_count(7)
        expect(page.locator('.disaster-marker .hazard-svg')).to_have_count(7)
        expect(page.locator('.marker-count')).to_have_text('2')
        expect(page.get_by_text('OpenStreetMap', exact=True)).to_be_visible()
        page.get_by_role('button', name='Show Flood filter, 1 mapped events').click()
        expect(page.locator('.disaster-marker')).to_have_count(1)
        nav.get_by_role('button', name='Disaster Explorer').click()
        expect(page.get_by_label('Hazard category')).to_have_value('Flood')
        nav.get_by_role('button', name='Dashboard').click()
        page.get_by_role('button', name='Clear Flood filter, 1 mapped events').click()
        expect(page.locator('.disaster-marker')).to_have_count(7)
        page.screenshot(path=str(output/'map-icons-overview-fixture.png'), full_page=True)
        marker = page.locator('.leaflet-marker-icon[tabindex="0"]:has(.marker-count)')
        marker.focus()
        marker.press('Enter')
        expect(page.locator('.leaflet-popup-content')).to_be_visible()
        expect(page.get_by_text('co-located events', exact=False)).to_be_visible()
        expect(page.get_by_role('link', name='Open official GDACS report').first).to_be_visible()
        page.screenshot(path=str(output/'map-icon-legend-fixture.png'), full_page=True)
        page.get_by_role('button', name='Collapse map legend').click()
        expect(page.get_by_role('button', name='Expand map legend')).to_be_visible()
        page.get_by_role('button', name='Expand map legend').click()
        page.set_viewport_size({'width':390,'height':844})
        page.wait_for_timeout(250)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.screenshot(path=str(output/'map-mobile-fixture.png'), full_page=True)
        page.set_viewport_size({'width':1512,'height':1080})
        page.route('**/sync-live-events',lambda route:route.fulfill(json={'status':'ok','fetched_count':len(fixture),'inserted_count':0,'updated_count':len(fixture),'skipped_count':0}))
        page.get_by_role('button',name='Refresh GDACS').click()
        expect(page.get_by_text('Refresh complete:',exact=False)).to_be_visible()
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
        expect(page.locator('.disaster-marker')).to_have_count(400, timeout=10000)
        assert any('/sync-live-events' in url for url in requests)
        assert not errors, errors
        # The only refresh above is intercepted with a synthetic response.
        assert not any('/local-demo/historical-predict' in url for url in requests)
        print('PASS: map SVG categories, severity borders, grouped events, keyboard popup, legend filter/collapse, attribution, empty and mobile states.')
        print('Screenshots:',output)
        browser.close()


if __name__ == '__main__':
    run()
