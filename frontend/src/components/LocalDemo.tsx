import { useState } from 'react';

type ObservedField = {value: string | number | null; unit: string | null; status: string};
export type Cyclone = {snapshot_id: string; retrieved_at_utc: string; source_url: string;
  response_sha256: string; country_occurrences: (string | null)[]; fields: Record<string, ObservedField>};
export function ObservationCards({records}: {records: Cyclone[]}) {
  return <div>{records.length === 0 && <p>No local cyclone observations available.</p>}
    {records.map((row, i) => <article key={`${row.snapshot_id}-${i}`} style={{border:'1px solid #888',padding:12,margin:8}}>
      <h3>Cyclone {row.fields.event_id?.value ?? 'ID unavailable'}</h3>
      <p>Countries as reported: {row.country_occurrences.join(', ') || 'Unavailable'}</p>
      {['episode_id','version','severity','population_exposure','vulnerability','alert_level','alert_score','latitude','longitude','publication_time','modification_time','event_start_time','event_end_time'].map(key => {
        const field = row.fields[key];
        return <p key={key}>{key}: {field?.status === 'available' ? `${field.value ?? 'Unavailable'} ${field.unit ?? ''}` : `Unavailable (${field?.status ?? 'missing'})`}</p>;
      })}
      <p>Retrieved: {row.retrieved_at_utc}</p><p>Source: {row.source_url}</p>
      <small>Response SHA-256: {row.response_sha256}</small>
    </article>)}
  </div>;
}
export function PredictionResults({results}: {results: Record<string, {status: string; predicted_label: string | null}>}) {
  return <div>{Object.entries(results).map(([target, value]) => <p key={target}>{target}: {value.status} — {value.predicted_label ?? 'No prediction available'}</p>)}</div>;
}
export default function LocalDemo({apiBase}: {apiBase: string}) {
  const [records,setRecords] = useState<Cyclone[]>([]);
  const [message,setMessage] = useState('Load the local extraction to inspect archived observations.');
  const [input,setInput] = useState('');
  const [results,setResults] = useState<Record<string,{status:string;predicted_label:string|null}>>({});
  async function load() {
    try { const response = await fetch(`${apiBase}/local-demo/cyclones`);
      if (!response.ok) throw new Error(`Observation request failed (${response.status})`);
      const data = await response.json(); setRecords(data.records); setMessage(data.reason ?? data.notice);
    } catch(error) {setMessage(String(error)); setRecords([]);}
  }
  async function predict() {
    setResults({});
    try {const response = await fetch(`${apiBase}/local-demo/historical-predict`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(JSON.parse(input))});
      if (!response.ok) throw new Error(`Historical input rejected (${response.status})`);
      const data = await response.json(); setResults(data.results); setMessage(data.notice);
    } catch(error) {setMessage(String(error));}
  }
  return <section style={{padding:20,background:'#17212f',color:'white'}}>
    <h2>Local research demonstration</h2>
    <h3>Archived GDACS observations — not ML predictions</h3>
    <p>Exposure is source-modeled exposure (including Pop74), not observed affected population. Retrieval does not establish original publication availability.</p>
    <button onClick={load}>Load local cyclone observations</button><p role="status">{message}</p>
    <ObservationCards records={records}/>
    <h3>Separate retrospective historical-model demonstration</h3>
    <p>Paste one authorized historical nine-feature JSON record. Do not paste GDACS observations or reserved test records. No mapping from the cards is performed.</p>
    <p>Required keys: disaster_type, disaster_subtype, iso, region, subregion, season, start_month, magnitude, magnitude_available.</p>
    <textarea aria-label="Historical feature JSON" rows={8} cols={70} value={input} onChange={event=>setInput(event.target.value)}/>
    <br/><button onClick={predict}>Run retrospective demonstration</button>
    <PredictionResults results={results}/>
    <p>Human-reviewed research only. Model probabilities are not verified operational confidence.</p>
  </section>;
}
