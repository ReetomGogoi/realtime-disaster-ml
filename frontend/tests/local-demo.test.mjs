import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile, writeFile, mkdtemp, rm } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import ts from "typescript";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

test("public navigation excludes prediction and cyclone research interfaces", async () => {
  const directory = await mkdtemp(new URL("./.local-demo-", import.meta.url));
  try {
    const appSource = await readFile(
      new URL("../src/App.tsx", import.meta.url),
      "utf8",
    );
    const navigation = appSource.match(/const pages = \[([\s\S]*?)\] as const;/);
    assert.ok(navigation);
    assert.deepEqual(
      [...navigation[1].matchAll(/"([^"]+)"/g)].map((match) => match[1]),
      [
        "Dashboard",
        "Disaster Explorer",
        "Model Performance",
        "About",
      ],
    );
    assert.doesNotMatch(appSource, /ML Prediction Lab/);
    assert.doesNotMatch(appSource, /PredictionLab/);
    assert.doesNotMatch(appSource, /CycloneIntelligence/);
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
    const { ObservationCards, LocalUnavailable } = await import(
      pathToFileURL(path)
    );
    assert.doesNotMatch(source, /PredictionLab/);
    assert.doesNotMatch(source, /historical-predict/);
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
