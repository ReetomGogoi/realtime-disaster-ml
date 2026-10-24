"""Synthetic browser fixtures verify latest/archive separation without GDACS IO.

Run against the opt-in local Vite/backend setup from the project docs.
"""
from playwright.sync_api import sync_playwright, expect


def run():
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page()
        page.set_default_timeout(10000)
        state={'revision':1,'fail':False,'archive_calls':0}
        def row(identifier,kind):
            return {'snapshot_id':kind,'retrieved_at_utc':'2026-06-01T00:00:00Z',
                'source_url':'https://www.gdacs.org/','response_sha256':'a'*64,
                'country_occurrences':['A','A'],
                'fields':{'event_id':{'value':identifier,'status':'available','unit':None},
                          'population_exposure':{'value':0,'status':'available','unit':'Pop74'}}}
        def latest(route):
            route.fulfill(json={'status':'available','last_successful_refresh_utc':f'2026-06-0{state["revision"]}T00:00:00Z',
                                'records':[row(f'latest-{state["revision"]}','sync')]})
        def archive(route):
            state['archive_calls']+=1
            route.fulfill(json={'status':'available','records':[row('archive-original','archive')]})
        def refresh(route):
            if state['fail']:
                route.fulfill(status=502,json={'detail':'synthetic failure'})
            else:
                state['revision']=2
                route.fulfill(json={'status':'success'})
        page.route('**/cyclones/latest',latest)
        page.route('**/local-demo/cyclones',archive)
        page.route('**/sync-live-events',refresh)
        page.goto('http://127.0.0.1:5173/?demo=1')
        expect(page.get_by_role('heading',name='Cyclone latest-1',exact=True)).to_be_visible()
        assert state['archive_calls']==0
        page.get_by_role('button',name='Archived Cyclone Observations',exact=True).click()
        expect(page.get_by_role('heading',name='Cyclone archive-original',exact=True)).to_be_visible()
        assert state['archive_calls']==1
        page.get_by_role('button',name='Latest GDACS Cyclones',exact=True).click()
        page.get_by_role('button',name='Refresh GDACS').click()
        expect(page.get_by_role('heading',name='Cyclone latest-2',exact=True)).to_be_visible()
        expect(page.get_by_text('Last successful refresh: 2026-06-02T00:00:00Z',exact=True)).to_be_visible()
        assert state['archive_calls']==1
        state['fail']=True
        page.get_by_role('button',name='Refresh GDACS').click()
        expect(page.get_by_role('alert')).to_be_visible()
        expect(page.get_by_role('heading',name='Cyclone latest-2',exact=True)).to_be_visible()
        page.get_by_role('button',name='Archived Cyclone Observations',exact=True).click()
        expect(page.get_by_role('heading',name='Cyclone archive-original',exact=True)).to_be_visible()
        expect(page.get_by_role('heading',name='Cyclone latest-2',exact=True)).to_have_count(0)
        print('PASS: latest/archive tabs, lazy archive fetch, refresh updates latest only, failed refresh retains data, original archive preserved')
        browser.close()


if __name__=='__main__':
    run()
