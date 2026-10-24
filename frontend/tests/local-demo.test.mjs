import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile,writeFile,mkdtemp,rm} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import ts from 'typescript';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';

test('observation/provenance/missing and unavailable prediction rendering',async()=>{
  const directory=await mkdtemp(new URL('./.local-demo-',import.meta.url));
  try {
    const source=await readFile(new URL('../src/components/LocalDemo.tsx',import.meta.url),'utf8');
    const output=ts.transpileModule(source,{compilerOptions:{jsx:ts.JsxEmit.ReactJSX,module:ts.ModuleKind.ESNext}}).outputText;
    const path=`${directory}/component.mjs`;
    await writeFile(path,output);
    const {ObservationCards,PredictionResults}=await import(pathToFileURL(path));
    const html=renderToStaticMarkup(React.createElement(ObservationCards,{records:[{
      snapshot_id:'fixture',retrieved_at_utc:'2026-06-01',source_url:'https://www.gdacs.org/xml/rss_7d.xml',
      response_sha256:'a'.repeat(64),country_occurrences:['A','A'],
      fields:{population_exposure:{value:0,unit:'Pop74',status:'available'}}}]}));
    assert.match(html,/0 Pop74/); assert.match(html,/Unavailable \(missing\)/);
    assert.match(html,/2026-06-01/); assert.match(html,/Response SHA-256/);
    const unavailable=renderToStaticMarkup(React.createElement(PredictionResults,{results:{affected_range:{status:'model_unavailable',predicted_label:null}}}));
    assert.match(unavailable,/model_unavailable/); assert.match(unavailable,/No prediction available/);
    assert.match(renderToStaticMarkup(React.createElement(ObservationCards,{records:[]})),/No local cyclone/);
  } finally {await rm(directory,{recursive:true,force:true});}
});
