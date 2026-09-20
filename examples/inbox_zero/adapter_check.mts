// Exercise the actual provider adapter, restricting all fetches to this loopback service.
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { performance } from "node:perf_hooks";

const [checkout, inputPath, outputPath, baseUrl] = process.argv.slice(2);
const { classifyWithSystem1 } = await import(pathToFileURL(resolve(checkout, "apps/web/utils/classifier/system1.ts")).href);
const requests = JSON.parse(readFileSync(inputPath, "utf8"));
const originalFetch = globalThis.fetch;
let localCalls = 0;
globalThis.fetch = ((url, options) => {
  if (String(url) !== `${baseUrl}/v1/classify`) throw new Error("External network request blocked");
  localCalls++;
  return originalFetch(url, options);
}) as typeof fetch;
const config = { provider: "system1", model: "email", apiKey: process.env.SYSTEM1_CLASSIFIER_TOKEN, baseUrl };
const rows = [];
for (let repeat = 0; repeat < 5; repeat++) {
  for (const request of requests) {
    const start = performance.now();
    const result = await classifyWithSystem1({ config, ...request.payload });
    rows.push({ id: request.id, repeat, latency_ms: performance.now() - start, result });
  }
}
writeFileSync(outputPath, JSON.stringify({ local_calls: localCalls, external_fetches: 0, rows }, null, 2) + "\n");
