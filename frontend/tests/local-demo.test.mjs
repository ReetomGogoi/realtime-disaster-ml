import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile, writeFile, mkdtemp, rm } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import ts from "typescript";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

test("observation/provenance/missing and unavailable prediction rendering", async () => {
  const directory = await mkdtemp(new URL("./.local-demo-", import.meta.url));
  try {
    const source = await readFile(
      new URL("../src/components/LocalDemo.tsx", import.meta.url),
      "utf8",
    );
    const output = ts.transpileModule(source, {
      compilerOptions: {
        jsx: ts.JsxEmit.ReactJSX,
        module: ts.ModuleKind.ESNext,
      },
    }).outputText;
    const path = `${directory}/component.mjs`;
    await writeFile(path, output);
    const {
      ObservationCards,
      PredictionResults,
      PredictionLab,
      LocalUnavailable,
    } = await import(pathToFileURL(path));
    const form = renderToStaticMarkup(
      React.createElement(PredictionLab, { apiBase: "", enabled: true }),
    );
    assert.match(form, /Historical feature record/);
    assert.doesNotMatch(form, /<textarea/);
    assert.match(
      renderToStaticMarkup(React.createElement(LocalUnavailable)),
      /Unavailable in production/,
    );
    const html = renderToStaticMarkup(
      React.createElement(ObservationCards, {
        records: [
          {
            snapshot_id: "fixture",
            retrieved_at_utc: "2026-06-01",
            source_url: "https://www.gdacs.org/xml/rss_7d.xml",
            response_sha256: "a".repeat(64),
            country_occurrences: ["A", "A"],
            fields: {
              population_exposure: {
                value: 0,
                unit: "Pop74",
                status: "available",
              },
            },
          },
        ],
      }),
    );
    assert.match(html, /0 Pop74/);
    assert.match(html, /Unavailable \(missing\)/);
    assert.match(html, /2026-06-01/);
    assert.match(html, /Response SHA-256/);
    assert.match(html, /TROPICAL CYCLONE · ARCHIVED/);
    const latest = renderToStaticMarkup(React.createElement(ObservationCards, {
      mode: "latest", records: [{snapshot_id: "sync-1", retrieved_at_utc: "2026-06-01",
        source_url: "https://www.gdacs.org/", country_occurrences: [], fields: {}}]
    }));
    assert.match(latest, /LATEST REFRESH/);
    assert.doesNotMatch(latest, /ARCHIVED/);
    assert.doesNotMatch(latest, /Response SHA-256/);
    assert.match(latest, /View GDACS source/);

    const unavailable = renderToStaticMarkup(
      React.createElement(PredictionResults, {
        results: {
          affected_range: {
            status: "model_unavailable",
            predicted_label: null,
          },
        },
      }),
    );
    assert.match(unavailable, /model unavailable/);
    assert.match(unavailable, /No prediction available/);
    assert.match(
      renderToStaticMarkup(
        React.createElement(ObservationCards, { records: [] }),
      ),
      /No local cyclone/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
