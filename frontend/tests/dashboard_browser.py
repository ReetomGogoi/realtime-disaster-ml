"""Local browser integration. No GDACS fetch, training or persistent DB writes.

Start isolated backend and opt-in Vite dev server per documentation first.
Screenshots show actual local API state; synthetic fixtures test filters only.
"""
import json
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': 1512, 'height': 1080}, ignore_https_errors=True)
        page.set_default_timeout(10000)
        requests, errors = [], []
        page.on('request', lambda req: requests.append(req.url))
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto('http://127.0.0.1:5173/')
        expect(page.get_by_role('heading', name='Global event monitor')).to_be_visible()
        expect(page.get_by_text('Monitoring connected')).to_be_visible()
        assert any('/events?limit=500&offset=0' in url for url in requests)
        output = Path(tempfile.gettempdir()) / 'disaster-dashboard-preview'
        output.mkdir(exist_ok=True)
        expect(page.locator('.leaflet-tile-loaded').first).to_be_visible(timeout=15000)
        page.screenshot(path=str(output/'dashboard-desktop.png'), full_page=True)
        nav=page.get_by_role('navigation', name='Main navigation')
        nav.get_by_role('button', name='Disaster Explorer').click()
        expect(page.get_by_role('searchbox')).to_be_visible()
        nav.get_by_role('button', name='Cyclone Intelligence').click()
        expect(page.get_by_text('Local extraction output missing')).to_be_visible()
        nav.get_by_role('button', name='ML Prediction Lab').click()
        expect(page.get_by_role('heading', name='Historical feature record')).to_be_visible()
        page.get_by_label('disaster type', exact=True).select_option('Storm')
        for name,value in [('disaster subtype','Tropical cyclone'),('iso','USA'),('region','Americas'),('subregion','Northern America')]:
            page.get_by_label(name,exact=True).fill(value)
        page.get_by_label('start month',exact=True).select_option('8')
        page.get_by_role('button', name='Run historical inference').click()
        expect(page.get_by_text('No prediction available')).to_have_count(2)
        assert any('/local-demo/historical-predict' in url for url in requests)
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
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.screenshot(path=str(output/'dashboard-mobile.png'),full_page=True)
        # Deterministic synthetic records verify search/filter/detail without
        # misrepresenting them as actual saved events or using live GDACS.
        fixture=[]
        for i,kind in enumerate(['Earthquake','Flood']):
            fixture.append(dict(id=i+1,source='GDACS',source_event_id=f'fixture:{i}',
                title=f'SYNTHETIC TEST {kind}',disaster_type=kind,country='Fixture country',
                latitude=20+i,longitude=75+i,event_time='2026-06-01T00:00:00Z',
                fetched_at='2026-06-01T00:05:00Z',created_at='2026-06-01T00:05:00Z',
                source_url='https://www.gdacs.org/'))
        page.route('**/events?*',lambda route:route.fulfill(json={'total':2,'events':fixture,'offset':0,'limit':500}))
        page.set_viewport_size({'width':1512,'height':1080})
        page.reload()
        nav.get_by_role('button',name='Disaster Explorer').click()
        page.get_by_role('searchbox').fill('Flood')
        expect(page.locator('.event-row')).to_have_count(1)
        page.locator('.event-row').click()
        expect(page.get_by_role('dialog')).to_be_visible()
        expect(page.get_by_role('link',name='View official GDACS report')).to_be_visible()
        page.get_by_role('button',name='Close details').click()
        nav.get_by_role('button',name='Dashboard').click()
        expect(page.locator('.leaflet-interactive')).to_have_count(2)
        page.route('**/sync-live-events',lambda route:route.fulfill(json={'status':'ok','fetched_count':2,'inserted_count':0,'updated_count':2,'skipped_count':0}))
        page.get_by_role('button',name='Refresh GDACS').click()
        expect(page.get_by_text('Refresh complete:',exact=False)).to_be_visible()
        assert any('/sync-live-events' in url for url in requests)
        assert not errors, errors
        # The only refresh above is intercepted with a synthetic response.
        print('PASS: actual events, cyclone, historical-predict and performance API requests; navigation; mobile layout; synthetic search/details.')
        print('Screenshots:',output)
        browser.close()


if __name__ == '__main__':
    run()
