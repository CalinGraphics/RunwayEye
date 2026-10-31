import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const vendorDir = join(root, "static", "vendor");
mkdirSync(vendorDir, { recursive: true });

const files = [
  ["node_modules/htmx.org/dist/htmx.min.js", "htmx.min.js"],
  ["node_modules/alpinejs/dist/cdn.min.js", "alpine.min.js"],
];

for (const [src, dest] of files) {
  copyFileSync(join(root, src), join(vendorDir, dest));
  console.log(`vendor: ${dest}`);
}

const fontDir = join(root, "static", "fonts");
mkdirSync(fontDir, { recursive: true });
// Atkinson Hyperlegible Next for text; B612 Mono (designed for Airbus cockpit displays) for readouts.
const fonts = [
  ["@fontsource-variable/atkinson-hyperlegible-next", "atkinson-hyperlegible-next-latin-wght-normal.woff2"],
  ["@fontsource-variable/atkinson-hyperlegible-next", "atkinson-hyperlegible-next-latin-ext-wght-normal.woff2"],
  ["@fontsource/b612-mono", "b612-mono-latin-400-normal.woff2"],
  ["@fontsource/b612-mono", "b612-mono-latin-700-normal.woff2"],
];
for (const [pkg, file] of fonts) {
  copyFileSync(join(root, "node_modules", pkg, "files", file), join(fontDir, file));
  console.log(`font: ${file}`);
}
