import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { shaders } from "../src/shaders.js";
const tables = JSON.parse(
  readFileSync(new URL("../reference/tables.json", import.meta.url)),
);
test("all passes obey the 8 storage buffer limit with 2 and 3 layers", () => {
  for (const n of [2, 3])
    for (const p of Object.values(shaders(n, tables)).flat()) {
      assert.ok(p.bindings.length <= 8);
      assert.equal(new Set(p.bindings).size, p.bindings.length);
    }
});
