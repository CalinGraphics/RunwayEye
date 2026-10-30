// Copies HTMX and Alpine.js from node_modules into app/static/vendor/,
// so the pages never depend on a CDN (the demo must work offline).
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const vendorDir = join(root, "app", "static", "vendor");
mkdirSync(vendorDir, { recursive: true });

const files = [
  ["node_modules/htmx.org/dist/htmx.min.js", "htmx.min.js"],
  ["node_modules/alpinejs/dist/cdn.min.js", "alpine.min.js"],
];

for (const [src, dest] of files) {
  copyFileSync(join(root, src), join(vendorDir, dest));
  console.log(`vendor: ${dest}`);
}
