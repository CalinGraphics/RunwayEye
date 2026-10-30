// Copies HTMX, Alpine.js and the fonts from node_modules into app/static/,
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

const fontDir = join(root, "app", "static", "fonts");
mkdirSync(fontDir, { recursive: true });
for (const [pkg, name] of [["archivo", "archivo"], ["jetbrains-mono", "jetbrains-mono"]]) {
  for (const subset of ["latin", "latin-ext"]) {
    const file = `${name}-${subset}-wght-normal.woff2`;
    copyFileSync(join(root, "node_modules", "@fontsource-variable", pkg, "files", file), join(fontDir, file));
    console.log(`font: ${file}`);
  }
}
