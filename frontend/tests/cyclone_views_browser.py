"""Public navigation regression: cyclone research is preserved but not routed."""
from playwright.sync_api import sync_playwright, expect


def run():
    requests = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': 1100, 'height': 800})
        page.on('request', lambda request: requests.append(request.url))
        page.goto('http://127.0.0.1:5173/?demo=1')
        nav = page.get_by_role('navigation', name='Main navigation')
        expect(nav.get_by_role('button')).to_have_count(4)
        expect(nav.get_by_text('Cyclone Intelligence', exact=True)).to_have_count(0)
        expect(page.get_by_role('heading', name='Three-hazard event monitor')).to_be_visible()
        assert not any('/cyclones/' in url or '/local-demo/cyclones' in url for url in requests)
        print('PASS: cyclone research has no public navigation, route, or API request.')
        browser.close()


if __name__ == '__main__':
    run()
